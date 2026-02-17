import os
import math
import importlib
from scipy.io import savemat
from pathlib import Path
import torch

import numpy as np
import matplotlib.pyplot as plt

from Asteroid_scenario.dynamics.polyhedron_model import (
    extract_unique_edges,
    build_face_edge_map,
    preprocess_geometry,
    model_opening,
)

from Modules_environment_online import (
    Units,
    Weights,
    Bounds,
    ParamsPhys,
    load_reference,
    NeuralNetwork,
    Optimization,
    AsteroidModel,
    DynamicsPropagator,
    MPC,
    performance_indices_vectors,
    Visualizer3D,
)
from linear_modules import linear_matrices, convert_mat_to_adimensional

# Note:
# If you do online training, the new network exists only on the Jetson. Therefore, in the live plot of the environment
# you always use the old model loaded on the PC to visualize the predicted trajectory. As a result, from the moment
# online learning starts onward, you’ll notice that the predicted trajectory in the live plot no longer matches the reference.

# ---------------------------------------------------------------------------
#  Connection to the environment (Client)
# ---------------------------------------------------------------------------
#JETSON_IP = "192.168.1.16"  # REAL
#JETSON_IP = "192.168.1.18"  # REAL
JETSON_IP = "127.0.0.1"      # TEST
PORT = 5027
DTYPE = np.float64
N_STATE = 6
N_CTRL  = 3 * 15

# ---------------------------------------------------------------------------
# Errors on the modelling
# ---------------------------------------------------------------------------
TILT_DEG = 3.0
AZ_DEG = 0.0
RHO_TRUE_MOD = 1
OMEGA_TRUE_MOD = 1


# ---------------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------------

# Decide if Linear or Neural Network ("Neural_Network")
ONLINE = True
SCENARIO = "Neural_Network"
#SCENARIO = "Linear"
ENVIRONMENT = "Asteroid_scenario"
SAVE = True
VISUALISATION = False
SIMULATION_TIME = 260  # number of simulation time instants (Δt = 5 minutes)

print(torch.__version__)
print(torch.version.cuda)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", DEVICE)

# ---------------------------------------------------------------------------
# SPIN AXIS #
# If want the "REAL" spin axis to be aligned with z axis --> put change_spin_axis = None
change_spin_axis = True   # set this to true if you want to have (in the real enviroment) the spin axis inclined

tilt_deg_nominal = TILT_DEG
az_deg_nominal   = AZ_DEG

spin_axis_direction = np.array(
    [
        np.sin(np.deg2rad(tilt_deg_nominal)) * np.cos(np.deg2rad(az_deg_nominal)),
        np.sin(np.deg2rad(tilt_deg_nominal)) * np.sin(np.deg2rad(az_deg_nominal)),
        np.cos(np.deg2rad(tilt_deg_nominal)),
    ]
)
#-------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SCEN_ROOT = ROOT / ENVIRONMENT



def main():
    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    model_path = SCEN_ROOT / "net_model" / "best_test_model_NET_no_rot_no_relu.pt"
    obj_path_reference = SCEN_ROOT / "dynamics" / "eros_asteroid_model_498.obj"
    obj_path_LF = SCEN_ROOT / "dynamics" / "eros_asteroid_model_95.obj"
    traj_mat = "reference_trajectories/circular_orbit_traj.mat"
    tt_mat = "reference_trajectories/circular_orbit_time.mat"
    if VISUALISATION:
        video_path = SCEN_ROOT / "video_tracking" / "tracking_3d_trajectory.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        video_path = None

    CONFIG = {
        "model_path": str(model_path),
        "obj_path_reference": str(obj_path_reference),
        "obj_path_lf": str(obj_path_LF),
        "traj_mat": str(traj_mat),
        "tt_mat": str(tt_mat),
        "T_steps_ref": SIMULATION_TIME,
        "print_level": 3,  # output standard IPOPT
        "max_iter": 2000,
    }

    # Workaround to solve a conflict (to be corrected)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # -----------------------------------------------------------------------
    # Constants asteroid
    # -----------------------------------------------------------------------
    rho_true = 2670.0
    G = 6.67430e-11
    T_eros = 5.27 * 3600.0
    Omega_true = 2 * math.pi / T_eros

    units = Units(LU=36000.0, VU=18.0, CU=0.6)

    # Constant of the LF model
    rho_LF = rho_true * 1.10
    Omega_LF = Omega_true * 1.01
    #########################

    rho_true = rho_LF * RHO_TRUE_MOD
    Omega_true = Omega_LF * OMEGA_TRUE_MOD


    # Obtaining the Omega vector
    if change_spin_axis is not None:
        Omega_true = spin_axis_direction * Omega_true

    # -----------------------------------------------------------------------
    # Horizons
    # -----------------------------------------------------------------------
    N_s = 16
    N_control = N_s - 1  # L


    # -----------------------------------------------------------------------
    # Weights
    # -----------------------------------------------------------------------
    Q_pos = 10000 * 1e-5
    Q_vel = 15000 * 1e-5
    R_value = 50 * 1e-8

    beta_z = 1.0
    Z_pos = beta_z * Q_pos
    Z_vel = beta_z * Q_vel
    w_end = 1.0
    w_control = np.linspace(1.0, w_end, N_control)
    w_pos_vel = np.linspace(1.0, 1.0, N_control)
    weights = Weights(
        Q_pos=Q_pos,
        Q_vel=Q_vel,
        Z_pos=Z_pos,
        Z_vel=Z_vel,
        R_value=R_value,
        w_control=w_control,
        w_pos_vel=w_pos_vel,
    )

    # -----------------------------------------------------------------------
    # Bounds
    # -----------------------------------------------------------------------
    u_max = 0.6 / units.CU  # adim, per componente
    lower = -u_max * np.ones(3 * N_control)
    upper = +u_max * np.ones(3 * N_control)
    bounds = Bounds(lower=lower, upper=upper)

    # -----------------------------------------------------------------------
    # Reference
    # -----------------------------------------------------------------------
    traj_phys, traj_adim, tt_vect = load_reference(
        CONFIG["traj_mat"], CONFIG["tt_mat"], units
    )
    x0 = traj_adim[0].copy()

    print("X0: ", x0.shape)

    # In online, first point is equal to the last one I remove the last point
    if ONLINE:
        T_per_orbit = float(tt_vect[-1] - tt_vect[0])
        traj_phys = traj_phys[:-1,:]
        traj_adim = traj_adim[:-1,:]
        tt_vect = tt_vect[:-1]
    else:
        T_per_orbit = None

    # -----------------------------------------------------------------------
    # Neural network
    # -----------------------------------------------------------------------
    net = NeuralNetwork(CONFIG["model_path"], device=DEVICE)
    net.load_model()

    # -----------------------------------------------------------------------
    # Optimization
    # -----------------------------------------------------------------------
    opt = Optimization(weight=weights, units=units, x_ref=traj_adim, net=net)

    # -----------------------------------------------------------------------
    # Dynamics
    # -----------------------------------------------------------------------
    # If using "general_dynamics" (spin axis inclined wrt z axis) --> Omega_true --> vector indicating spin axis direction and magnitude
    params = ParamsPhys(rho_real=rho_true, omega_real=Omega_true, G=G)

    vertices, faces = model_opening(CONFIG["obj_path_reference"])
    vertices = vertices * 1000.0  # km-->m
    model = AsteroidModel(params=params, vertices=vertices, faces=faces)
    dyn = DynamicsPropagator(options={}, units=units, model=model, change_spin_axis=change_spin_axis)

    #####################################################################
    vertices_LF, faces_LF = model_opening(CONFIG["obj_path_lf"])
    vertices_LF = vertices_LF * 1000.0  # km-->m

    # Pre-processing:
    edges_LF = extract_unique_edges(faces_LF)
    edge_faces_LF = build_face_edge_map(faces_LF, edges_LF)
    face_normals_LF, face_centroids_LF = preprocess_geometry(vertices_LF, faces_LF)

    ######################################################################

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------
    if VISUALISATION:
        zoom_in_meters = 23_000
        zoom_in_meters_follow = 16_000
        vis = Visualizer3D(
            units=units,
            zoom_range_phys=zoom_in_meters,
            zoom_range_phys_follow=zoom_in_meters_follow,
            fps=4,
            video_out=str(video_path),
            win_back=10,
            win_ahead=30,
            follow=False,  # True: the live plot follows the current state
        )

    # -----------------------------------------------------------------------
    # Computation Matrices Linear Model
    # -----------------------------------------------------------------------

    A_vect, B_vect = linear_matrices(
        traj_phys,
        tt_vect,
        Omega_LF,
        vertices_LF,
        faces_LF,
        edges_LF,
        edge_faces_LF,
        face_normals_LF,
        face_centroids_LF,
        rho_LF,
        G,
    )

    A_vect_AD, B_vect_AD = convert_mat_to_adimensional(
        traj_phys, A_vect, B_vect, units.LU, units.VU, units.CU
    )

    # -----------------------------------------------------------------------
    # MPC setting
    # -----------------------------------------------------------------------
    options_solv = {
        "max_iter": CONFIG["max_iter"],
        "print_level": CONFIG["print_level"],
    }
    mpc = MPC(
        N_s=N_s,
        N_control=N_control,
        bounds=bounds,
        weight=weights,
        net=net,
        opt=opt,
        options_solv=options_solv,
        units=units,
        dyn=dyn,
        scenario=SCENARIO,
        JETSON_IP = JETSON_IP,
        PORT = PORT,
        DTYPE = DTYPE,
        N_STATE = N_STATE,
        N_CTRL = N_CTRL,
        T_per_orbit=T_per_orbit
    )

    # -----------------------------------------------------------------------
    # MPC run
    # -----------------------------------------------------------------------
    if VISUALISATION:
        try:
            traj_followed_adim, controls_hist_adim = mpc.run(
                CONFIG["T_steps_ref"],
                x0,
                traj_adim,
                tt_vect,
                vis=vis,
                A_vect_AD=A_vect_AD,
                B_vect_AD=B_vect_AD,
            )

        finally:
            vis.close()
    else:
        traj_followed_adim, controls_hist_adim = mpc.run(
            CONFIG["T_steps_ref"],
            x0,
            traj_adim,
            tt_vect,
            A_vect_AD=A_vect_AD,
            B_vect_AD=B_vect_AD,
        )

    # -----------------------------------------------------------------------
    # Metrics and plots
    # -----------------------------------------------------------------------
    controls_hist_phys = units.control_to_phys(controls_hist_adim)
    traj_followed_phys = units.traj_to_phys(traj_followed_adim)

    # SAVING TRAJECTORY FOLLOWED AND THE CONTROLS
    if SAVE:
        if SCENARIO == "Neural_Network":
            savemat(
                "evaluation_results/trajectory_followed/traj_followed_net.mat",
                {"traj_followed": traj_followed_phys},
            )
            savemat(
                "evaluation_results/fuel_results/fuel_consumption_net.mat",
                {"fuel_consumption_each_instant": controls_hist_phys},
            )
        else:
            savemat(
                "evaluation_results/trajectory_followed/traj_followed_linear.mat",
                {"traj_followed": traj_followed_phys},
            )
            savemat(
                "evaluation_results/fuel_results/fuel_consumption_linear.mat",
                {"fuel_consumption_each_instant": controls_hist_phys},
            )

    if not ONLINE:
        end_win = min(len(traj_followed_phys), len(traj_phys))

        (
            APE_pos,
            APE_vel,
            MPE_pos,
            RPE_pos,
            MPE_vel,
            RPE_vel,
            VAR_pos,
            VAR_vel,
        ) = performance_indices_vectors(
            traj_followed_phys, traj_phys[:end_win], 1, end_win, moving_window=8
        )
        time_min = (tt_vect[:end_win] - tt_vect[0]) / 60.0
    else:
        n_sim = len(traj_followed_phys)
        n_ref = len(traj_phys)
        reps = int(np.ceil(n_sim / n_ref))

        traj_followed_phys_cmp = traj_followed_phys[:n_sim]
        traj_phys_cmp = np.tile(traj_phys, (reps, 1))[:n_sim]  # reference estesa (wrap)

        dt = float(tt_vect[1] - tt_vect[0]) if len(tt_vect) > 1 else 0.0
        tt_cmp = np.arange(n_sim, dtype=float) * dt  # copre tutta la simulazione
        end_win = n_sim

        time_min = (tt_cmp - tt_cmp[0]) / 60.0

        (
            APE_pos,
            APE_vel,
            MPE_pos,
            RPE_pos,
            MPE_vel,
            RPE_vel,
            VAR_pos,
            VAR_vel,
        ) = performance_indices_vectors(
            traj_followed_phys_cmp, traj_phys_cmp[:end_win], 1, end_win, moving_window=100
        )

    ############# DEBUG #############################
    mean_MPE_pos = float(np.mean(MPE_pos))
    print(f"\nMPE pos: {mean_MPE_pos}\n")
    #################################################

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    # time_min = (tt_vect[:end_win] - tt_vect[0]) / 60.0
    axs[0].plot(time_min, APE_pos, label="APE pos")
    axs[0].plot(time_min, MPE_pos, label="MPE pos")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylabel("Error [m]")
    axs[0].set_title("Absolute & Mean Position Error")

    axs[1].plot(time_min, RPE_pos)
    axs[1].grid(True)
    axs[1].set_ylabel("Error [m]")
    axs[1].set_title("Relative Position Error (RPE)")

    axs[2].plot(time_min, VAR_pos)
    axs[2].grid(True)
    axs[2].set_xlabel("Time [min]")
    axs[2].set_ylabel("Variance [m^2]")
    axs[2].set_title("Position Error Variance")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()