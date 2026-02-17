import os
import math
import importlib
from scipy.io import savemat
from pathlib import Path

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import sys

from dynamics.polyhedron_model import (
    extract_unique_edges,
    build_face_edge_map,
    preprocess_geometry,
)

from Modules_Montecarlo import (
    Units,
    Weights,
    Bounds,
    ParamsPhys,
    load_reference,
    NeuralNetwork,
    #Optimization,
    AsteroidModel,
    DynamicsPropagator,
    MPC,
    performance_indices_vectors,
    Visualizer3D,
)

from Modules_Montecarlo import Optimization, Algorithm

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from linear_modules import linear_matrices, convert_mat_to_adimensional
from dynamics.polyhedron_model import model_opening

# ---------------------------------------------------------------------------
# Montecarlo
# ---------------------------------------------------------------------------
NUMBER_OF_SIMULATIONS = 5
SEED = 42
DIST = "uniform"

SIMULATION_EXECUTION_TIME = True

# Uncertainty definitions
RHO_PCT   = 0.30      # +/-30%
OMEGA_PCT = 0.05      # +/-5%

TILT_MIN_DEG = 0.0
TILT_MAX_DEG = 6.0

AZ_MIN_DEG   = 0.0
AZ_MAX_DEG   = 360.0

np.random.seed(SEED)

# =========================
# Sampling helpers
# =========================
def sample_uniform(low, high, n):
    return np.random.uniform(low=low, high=high, size=n)

def sample_gaussian_3sigma(mean, half_range, n):
    """Gaussian with half_range = 3*sigma."""
    sigma = half_range / 3.0
    return np.random.normal(loc=mean, scale=sigma, size=n)

# ---------------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------------

# Decide if Linear or Neural Network ("Neural_Network")
SCENARIO = "Neural_Network"
#SCENARIO = "Linear"
ENVIRONMENT = "Asteroid_scenario"
SAVE = False
VISUALISATION = False
VISUALISATION_EACH_SIM = True
SIMULATION_TIME = 100  # number of simulation time instants (Δt = 5 minutes)
change_spin_axis = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", DEVICE)

ROOT = Path(__file__).resolve().parent.parent
SCEN_ROOT = ROOT / ENVIRONMENT



def main():

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    model_path = SCEN_ROOT / "net_model" / "best_test_model_NET_no_rot_no_relu.pt"
    obj_path_reference = SCEN_ROOT / "dynamics" / "eros_asteroid_model_498.obj"
    obj_path_LF = SCEN_ROOT / "dynamics" / "eros_asteroid_model_95.obj"
    traj_mat = SCEN_ROOT / "reference_trajectories" / "trajectory_reference.mat"
    tt_mat = SCEN_ROOT / "reference_trajectories" / "time_reference.mat"
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

    #########################
    rho_reference = rho_true * 1.10
    Omega_reference = Omega_true * 1.01
    #########################

    #####################
    # rho --> +/- 30%
    # omega --> +/- 5%
    # tilt --> da 0° a 6°
    # az --> da 0 a 360°

    # --- 1) Scale factors (dimensionless) ---
    rho_scale_low = 1.0 - RHO_PCT
    rho_scale_high = 1.0 + RHO_PCT
    omega_scale_low = 1.0 - OMEGA_PCT
    omega_scale_high = 1.0 + OMEGA_PCT

    if DIST == "uniform":
        rho_scale = sample_uniform(rho_scale_low, rho_scale_high, NUMBER_OF_SIMULATIONS)
        omega_scale = sample_uniform(omega_scale_low, omega_scale_high, NUMBER_OF_SIMULATIONS)
    elif DIST == "gaussian":
        # centered at 1.0, with +/-pct interpreted as 3-sigma bounds
        rho_scale = sample_gaussian_3sigma(mean=1.0, half_range=RHO_PCT, n=NUMBER_OF_SIMULATIONS)
        omega_scale = sample_gaussian_3sigma(mean=1.0, half_range=OMEGA_PCT, n=NUMBER_OF_SIMULATIONS)
    else:
        raise ValueError("DIST must be 'uniform' or 'gaussian'")

    rho_mc = rho_scale * rho_reference
    omega_mc = omega_scale * Omega_reference

    # --- 2) Angles (degrees) ---
    tilt_deg = sample_uniform(TILT_MIN_DEG, TILT_MAX_DEG, NUMBER_OF_SIMULATIONS)  # [0, 6] deg
    az_deg = sample_uniform(AZ_MIN_DEG, AZ_MAX_DEG, NUMBER_OF_SIMULATIONS)  # [0, 360] deg

    # =========================
    # Pack into Monte Carlo matrix
    # =========================
    # Each row is one MC scenario: [rho, omega, tilt_deg, az_deg]
    MC_matrix = np.column_stack([rho_mc, omega_mc, tilt_deg, az_deg])

    print("MC_matrix shape:", MC_matrix.shape)
    print("rho   range:", rho_mc.min(), rho_mc.max())
    print("omega range:", omega_mc.min(), omega_mc.max())
    print("tilt  range:", tilt_deg.min(), tilt_deg.max())
    print("az    range:", az_deg.min(), az_deg.max())
    ################################################################################################################

    # Prendiamo questo come reference
    # #########################
    # rho_reference = rho_true * 1.10
    # Omega_reference = Omega_true * 1.01
    # #########################

    Parameters_by_scenario = {}

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

    # for k, (scenario_name, scenario_considered) in enumerate(scenarios_dict.items(), start=1):
    for k in range(NUMBER_OF_SIMULATIONS):
        print(f"--- SIMULATION NUMBER {k} ---")

        # START TIME
        if SIMULATION_EXECUTION_TIME == True:
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()


        # Extracting parameter
        # rho_scale
        # omega_scale
        # tilt_deg
        # az_deg
        # spin_axis
        # scenario_id
        # rho_abs
        # omega_abs

        rho_scenario, Omega_scenario_val, tilt_deg_scenario, az_deg_scenario = (
            MC_matrix[k]
        )

        spin_axis_direction = np.array(
            [
                np.sin(np.deg2rad(tilt_deg_scenario)) * np.cos(np.deg2rad(az_deg_scenario)),
                np.sin(np.deg2rad(tilt_deg_scenario)) * np.sin(np.deg2rad(az_deg_scenario)),
                np.cos(np.deg2rad(tilt_deg_scenario)),
            ]
        )
        #################################################################################################

        # Print:
        print(f"rho: {rho_scenario}")
        print(f"omega: {Omega_scenario_val}")
        print(f"tilt deg: {tilt_deg_scenario}")
        print(f"az deg: {az_deg_scenario}")

        # Setting parameters
        #########################
        # Omega_scenario_val = Omega_reference * omega_scale
        # rho_scenario = rho_reference * rho_scale
        #########################

        # Obtaining the Omega vector
        if change_spin_axis is not None:
            Omega_scenario = spin_axis_direction * Omega_scenario_val

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
        params = ParamsPhys(rho_real=rho_scenario, omega_real=Omega_scenario, G=G)

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
        # Computation Matrices Linear Model
        # -----------------------------------------------------------------------

        A_vect, B_vect = linear_matrices(
            traj_phys,
            tt_vect,
            Omega_scenario_val,      # <---- changed here
            vertices_LF,
            faces_LF,
            edges_LF,
            edge_faces_LF,
            face_normals_LF,
            face_centroids_LF,
            rho_scenario,         # <---- changed here
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

        algorithm = Algorithm(units=units, N_s=N_s,N_control=N_control,weights=weights,net=net,bounds=bounds,opt=opt,traj_phys=traj_phys,traj_adim=traj_adim,tt_vect=tt_vect,scenario=SCENARIO)

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
            algorithm=algorithm
        )

        # -----------------------------------------------------------------------
        # MPC run
        # -----------------------------------------------------------------------
        if VISUALISATION:
            vis.reset()

            traj_followed_adim, controls_hist_adim = mpc.run(
                CONFIG["T_steps_ref"],
                x0,
                traj_adim,
                tt_vect,
                vis=vis,
                A_vect_AD=A_vect_AD,
                B_vect_AD=B_vect_AD,
            )

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

        #################################################
        # ROBUSTNESS METRICS FOR THE K SCENARIO
        #################################################

        # Mean Position Error
        mean_MPE_pos = float(np.mean(MPE_pos))
        mean_MPE_vel = float(np.mean(MPE_vel))

        # Worst-case (massimo errore)
        max_APE_pos = float(np.max(APE_pos))
        max_APE_vel = float(np.max(APE_vel))

        # Percentile 95% della APE positional
        p95_APE_pos = float(np.percentile(APE_pos, 95))
        p95_APE_vel = float(np.percentile(APE_vel, 95))

        # Percentile 90% della APE positional
        p90_APE_pos = float(np.percentile(APE_pos, 90))
        p90_APE_vel = float(np.percentile(APE_vel, 90))

        # Percentile 80% della APE positional
        p80_APE_pos = float(np.percentile(APE_pos, 80))
        p80_APE_vel = float(np.percentile(APE_vel, 80))

        print(f"rho: {rho_scenario}")
        print(f"omega: {Omega_scenario_val}")
        print(f"tilt deg: {tilt_deg_scenario}")
        print(f"az deg: {az_deg_scenario}")
        print(f"\nMPE pos: {mean_MPE_pos}\n")

        # Salva TUTTE le metriche + parametri scenario
        Parameters_by_scenario[k] = {
            # parametri dello scenario
            "rho": float(rho_scenario),
            "omega": float(Omega_scenario_val),
            "tilt_deg": float(tilt_deg_scenario),
            "az_deg": float(az_deg_scenario),
            # metriche principali
            "mean_MPE_pos": mean_MPE_pos,
            "mean_MPE_vel": mean_MPE_vel,
            # worst-case
            "max_APE_pos": max_APE_pos,
            "max_APE_vel": max_APE_vel,
            # percentile
            "p95_APE_pos": p95_APE_pos,
            "p95_APE_vel": p95_APE_vel,
            # percentile 2
            "p90_APE_pos": p90_APE_pos,
            "p90_APE_vel": p90_APE_vel,
            # percentile 3
            "p80_APE_pos": p80_APE_pos,
            "p80_APE_vel": p80_APE_vel,
        }

        ########### TIME ###############################
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print("Execution time: ", (t1 - t0))
        #################################################

        ###############################################################################################
        if VISUALISATION_EACH_SIM:
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

            fig, axs = plt.subplots(3, 1, figsize=(8, 8))
            time_min = (tt_vect[:end_win] - tt_vect[0]) / 60.0
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


    if VISUALISATION:
        vis.close()
        plt.close(vis.fig)


    # opzionale: salvarlo in file
    np.save("Parameters_by_scenario.npy", Parameters_by_scenario)




if __name__ == "__main__":
    main()