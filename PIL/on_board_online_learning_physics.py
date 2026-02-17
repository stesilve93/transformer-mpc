from dataclasses import dataclass
import numpy as np
import torch
import cyipopt
from pathlib import Path
from scipy.io import loadmat, savemat
import socket
import struct
import time
import copy
import random as random
from collections import deque
from typing import Deque, Tuple
import matplotlib.pyplot as plt
import math

from on_board_modules import  *
from sh_accelerations import PhysicsParams
from net_model.pinnsformer import PINNsformer, get_positional_encoding


# ---------------------------------------------------------------------------
#  Online learning
# ---------------------------------------------------------------------------
ONLINE_TRAINING = True
ONLINE_UPDATE_EVERY_STEPS = 8      # how many steps between network updates
START_INSTANT_TRAINING = 250       # after how many time instants the first update starts
ONLINE_BATCH_SIZE = 250            # size of the batch that is created (1 batch consists of batch_size trajectories)
ONLINE_STEPS_PER_UPDATE = 20       # number of network updates performed at each online learning step
ONLINE_REPLAY_CAPACITY = 1024      # buffer size (sliding queue)
ONLINE_LR = 1e-6

# Physics-------------------------------
ONLINE_LR_OMEGA = 1e-6
ONLINE_LR_MU = 5e7
ONLINE_LR_TILT = 1
ONLINE_LR_AZ = 1
# --------------------------------------

ONLINE_SAVE_PATH = "net_model/online_learning/NET_online.pt"
ONLINE_SAVE = False   # if you want to save the model (after online learning)
FAST_BUT_REP = False

# ---------------------------------------------------------------------------
#  Connection to the environment (Server)
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
#HOST = "0.0.0.0"
PORT = 5023
DTYPE = np.float64
N_STATE = 6

# ---------------------------------------------------------------------------
#  Parameters
# ---------------------------------------------------------------------------
SIMULATION_TIME = 260      # Has to be the same of the "environment" code
EARLY_STOPPING = True
PATIENCE = 50
NS = 16
SAVE_K_ITERATION = SIMULATION_TIME - NS     # specify the iteration at which the times list is saved
INFERENCE_TIME = False
SOLVE_OPT_TIME = False
ONLINE_TRAINING_TIME = False

PRINT_LEVEL = 0

# ---------------------------------------------------------------------------
#  Physics
# ---------------------------------------------------------------------------
START_INSTANT_TRAINING_PHYSICS = 20       # time step in which the estimation of the asteroid physical parameters starts
ONLINE_UPDATE_EVERY_STEPS_PHYSICS = 8     # number of time steps between two consecutive updates of the parameter estimation process

# Starting values of the physical parameters in the estimation process
MU_REFERENCE = 245829.45401648662
OMEGA_REFERENCE = 0.000301044457317428
TILT_REFERENCE = 12
AZ_REFERENCE = 0

# ---------------------------------------------------------------------------
#  Residuals
# ---------------------------------------------------------------------------
residuals_phys_vect = []

# ---------------------------------------------------------------------------
#  Variables to save time values
# ---------------------------------------------------------------------------
infer_times_ms = []
solve_times_ms = []
online_times_ms = []

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", DEVICE)


def main():
    scenario = "Neural_Network"
    global ONLINE_TRAINING
    # ----------------------------------------
    # Parameters
    # ----------------------------------------
    rho_true = 2670.0
    G = 6.67430e-11
    T_eros = 5.27 * 3600.0
    Omega_true = 2 * math.pi / T_eros

    Omega_true = Omega_true * 1.01

    mu_true = 473305.9704232581  # rho = 1.10

    # TODO: check if uncomment this
    # ---------------------------------------------------------------------------
    # # SPIN AXIS
    # # If want the "REAL" spin axis to be aligned with z axis --> put change_spin_axis = None
    # change_spin_axis = True   # set this to true if you want to have (in the real enviroment) the spin axis inclined
    #
    # tilt_deg_nominal = 0.0
    # #tilt_deg_nominal = 12.0   # <-----------------rimetti questo
    # az_deg_nominal   = 0.0
    #
    # spin_axis_direction = np.array(
    #     [
    #         np.sin(np.deg2rad(tilt_deg_nominal)) * np.cos(np.deg2rad(az_deg_nominal)),
    #         np.sin(np.deg2rad(tilt_deg_nominal)) * np.sin(np.deg2rad(az_deg_nominal)),
    #         np.cos(np.deg2rad(tilt_deg_nominal)),
    #         ]
    # )
    #
    # Omega_true = spin_axis_direction * Omega_true
    #-------------------------------------------------------------------------------

    # ----------------------------------------
    # Units
    # ----------------------------------------
    LU = 36000.0
    VU = 18.0
    CU = 0.6

    units = Units(LU=LU, VU=VU, CU=CU)

    # ----------------------------------------
    # Reference trajectory loading
    # ----------------------------------------
    ROOT = Path(__file__).resolve().parent

    traj_mat = ROOT / "reference_trajectories" / "circular_orbit_traj.mat"
    tt_mat = ROOT / "reference_trajectories" / "circular_orbit_time.mat"

    traj_phys, traj_adim, tt_vect = load_reference(
        traj_mat, tt_mat, units
    )

    # First point is equal to the last one I remove the last point
    traj_phys = traj_phys[:-1,:]
    traj_adim = traj_adim[:-1,:]
    tt_vect = tt_vect[:-1]

    # -----------------------------------------------------------------------
    # Options solver
    # -----------------------------------------------------------------------
    max_iter = 200
    print_level = PRINT_LEVEL
    options_solv = {
        "max_iter": max_iter,
        "print_level": print_level,
    }

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

    phys = PhysicsParams(omega=Omega_true, mu=mu_true).to(
        DEVICE
    )

    # -----------------------------------------------------------------------
    # Neural network
    # -----------------------------------------------------------------------
    model_path = ROOT / "net_model" / "best_test_model_NET_no_rot_no_relu.pt"
    net = NeuralNetwork(model_path, device=DEVICE, phys=phys, units=units)
    net.load_model()

    # -----------------------------------------------------------------------
    # Bounds
    # -----------------------------------------------------------------------
    u_max = 0.6 / units.CU  # adim, for componente
    lower = -u_max * np.ones(3 * N_control)
    upper = +u_max * np.ones(3 * N_control)
    bounds = Bounds(lower=lower, upper=upper)

    # -----------------------------------------------------------------------
    # Optimization
    # -----------------------------------------------------------------------
    opt = Optimization(weight=weights, units=units, x_ref=traj_adim, net=net, INFERENCE_TIME=INFERENCE_TIME)

    L = N_control
    U_guess = np.zeros((L, 3), dtype=float)

    # -----------------------------------------------------------------------
    # Replay and data saved lists
    # -----------------------------------------------------------------------
    replay = ReplayBuffer(ONLINE_REPLAY_CAPACITY)
    traj_followed = []
    controls_hist = []
    loss_online_single_step = []
    loss_online = []
    MPE_current_vect = []
    MPE_current_mean_vect = []

    omega_vect = []
    mu_vect = []
    tilt_vect = []
    az_vect = []

    number_iterations = 0
    best_state_dict = copy.deepcopy(net.model.state_dict())  # initial snapshot (model pre-trained)
    BEST_MPE = float("inf")


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[JETSON] Listening on {HOST}:{PORT}")

        # ---------------------------------------
        # Reception of data from the environment
        # --------------------------------------
        conn, addr = s.accept()

        # ---------------------------------------
        # Definition of the IPOPT problem
        # --------------------------------------
        U_guess = np.zeros((L, 3), dtype=float)
        x_k= traj_adim[0].copy()

        prob = IpoptProblem(
            x_k,  # x_k
            U_guess,  # U0_vec
            bounds,
            opt,
            scenario=scenario,
            Ns=N_s,
        )

        nlp = cyipopt.Problem(
            n=(15 * 3),  # U0_vec.size
            m=prob.m,
            problem_obj=prob,
            lb=bounds.lower,
            ub=bounds.upper,
        )
        # IPOPT options
        nlp.add_option("tol", 1e-6)
        nlp.add_option("max_iter", int(options_solv.get("max_iter", 100)))
        nlp.add_option("hessian_approximation", "limited-memory")
        nlp.add_option("print_level", int(options_solv.get("print_level", 5)))

        with conn:
            print(f"[JETSON] Connected with {addr}")
            while True:
                # 1) Receive state from the "spacecraft"
                k = recv_int(conn)
                x_k = recv_array(conn, N_STATE)
                print("[JETSON] k =", k)
                print("[JETSON] Received state x =", x_k)

                x_k = x_k.astype(np.float32, copy=True)
                prob.x0 = x_k

                # Saving the state
                traj_followed.append(x_k.copy())

                x_ref_window = cyclic_window(traj_adim, start=k, length=L)  # shape (L,6)

                opt.x_ref = x_ref_window
                opt.set_x_ref()

                U0_vec = U_guess.reshape(-1)

                # ---------------------------------------------------------------------------------------
                # ONLINE LEARNING (AFTER A CERTAIN NUMBER OF ORBITS)
                # ---------------------------------------------------------------------------------------

                if ONLINE_TRAINING:
                    Lw = L  # equal to N_control
                    if len(controls_hist) >= Lw:
                        i0 = (
                            len(controls_hist) - Lw
                        )  # starting index of the most recent window
                        x0_win = traj_followed[
                            i0
                        ].copy()  # initial state of the window

                        # Control sequence of the considered trajectory
                        U_win = cyclic_window(
                            np.asarray(controls_hist), start=i0, length=Lw
                        )

                        # State sequence of the considered trajectory
                        X_seq = cyclic_window(
                            np.asarray(traj_followed), start=i0, length=Lw + 1
                        )

                        # Step-by-step state differences of the trajectories
                        target_deltas = X_seq[1:] - X_seq[:-1]  # (L,6)
                        replay.add(x0_win, U_win, target_deltas)


                    if (k >= START_INSTANT_TRAINING_PHYSICS) and (
                        k - START_INSTANT_TRAINING_PHYSICS
                    ) % (ONLINE_UPDATE_EVERY_STEPS_PHYSICS) and (k < START_INSTANT_TRAINING):
                        x0_b, U_b, Yd_b = replay.sample(
                            ONLINE_BATCH_SIZE, device=DEVICE
                        )

                        if not hasattr(net, "optimizer"):
                            net.make_optimizer_2(lr=float(ONLINE_LR), lr_omega=ONLINE_LR_OMEGA, lr_mu=ONLINE_LR_MU, lr_tilt=ONLINE_LR_TILT, lr_azimuth=ONLINE_LR_AZ, weight_decay=0.0)

                        # physical update
                        metrics_phys = net.train_step_physics(
                            x0_b, U_b, Yd_b,
                            dt=300.0,
                            grad_clip=1.0,
                        )

                        # DEBUG -----------------------------------------------------------------------------
                        print("\n[DEBUG] Physics optimizer param groups:")
                        for i, g in enumerate(net.optimizer_physics.param_groups):
                            lr = g["lr"]
                            params = g["params"]
                            name = ""
                            if params[0] is net.phys.omega_raw:
                                name = "omega"
                            elif params[0] is net.phys.mu_raw:
                                name = "mu"
                            elif params[0] is net.phys.tilt_raw:
                                name = "tilt"
                            elif params[0] is net.phys.azim_raw:
                                name = "azimuth"
                            print(f"  group {i}: param={name}, lr={lr}")

                    # ---------------------------------------------------------------------------------------

                        omega_vect.append(metrics_phys.get("omega"))
                        mu_vect.append(metrics_phys.get("mu"))
                        tilt_vect.append(metrics_phys.get("tilt_deg"))
                        az_vect.append(metrics_phys.get("az_deg"))

                        print("[PHYS]", metrics_phys)

                    # ---------------------------------------------------------------------------------------------------------------

                    # Update every "_update_every" step
                    if (
                        (k >= START_INSTANT_TRAINING)
                        and ((k - START_INSTANT_TRAINING) % (ONLINE_UPDATE_EVERY_STEPS))
                        == 0
                        and len(replay) > 0
                    ):
                        print("\n\n\n UPDATING!!")
                        # Making trainable the last layer
                        if not hasattr(net, "_head_unfrozen"):
                            net.set_trainable_head()
                            net._head_unfrozen = True

                        # if is the first update, initialise the optimizer
                        if not hasattr(net, "optimizer"):
                            net.make_optimizer_2(lr=float(ONLINE_LR), lr_omega=ONLINE_LR_OMEGA, lr_mu=ONLINE_LR_MU, lr_tilt=ONLINE_LR_TILT, lr_azimuth=ONLINE_LR_AZ, weight_decay=0.0)

                        if not FAST_BUT_REP:
                            batches = replay.iter_batches(
                                batch_size=ONLINE_BATCH_SIZE,
                                max_batches=ONLINE_STEPS_PER_UPDATE,
                                shuffle=True,
                            )

                            # ----------- START TIME ONLINE LEARNING ------------------
                            if ONLINE_TRAINING_TIME == True:
                                if opt.dev.type == "cuda":
                                    torch.cuda.synchronize()
                                t0 = time.perf_counter()
                            # -----------------------------------------------------------

                            for x0_b, U_b, Yd_b in batches:
                                loss_online_iteration = net.train_step_2(x0_b, U_b, Yd_b, fast_but_rep=FAST_BUT_REP,grad_clip=1.0)
                                loss_online_single_step.append(loss_online_iteration)

                            # ---------------------------------------------------------
                            if ONLINE_TRAINING_TIME == True:
                                if opt.dev.type == "cuda":
                                    torch.cuda.synchronize()
                                t1 = time.perf_counter()
                                online_times_ms.append(1e3 * (t1 - t0))  # conversion to milliseconds
                            # ----------- END TIME ONLINE LEARNING ------------------

                            loss_online.append(np.mean(loss_online_single_step))

                        else:
                            # ----------- START TIME ONLINE LEARNING ------------------
                            if ONLINE_TRAINING_TIME == True:
                                if opt.dev.type == "cuda":
                                    torch.cuda.synchronize()
                                t0 = time.perf_counter()
                            # -----------------------------------------------------------

                            for _ in range(ONLINE_STEPS_PER_UPDATE):
                                x0_b, U_b, Yd_b = replay.sample(ONLINE_BATCH_SIZE, device=net.device)
                                loss_online_iteration = net.train_step_2(x0_b, U_b, Yd_b)
                                loss_online_single_step.append(loss_online_iteration)

                            # ---------------------------------------------------------
                            if ONLINE_TRAINING_TIME == True:
                                if opt.dev.type == "cuda":
                                    torch.cuda.synchronize()
                                t1 = time.perf_counter()
                                online_times_ms.append(1e3 * (t1 - t0))  # conversion to milliseconds
                            # ----------- END TIME ONLINE LEARNING ------------------

                            loss_online.append(np.mean(loss_online_single_step))


                        # Saving the new model
                        if ONLINE_SAVE:
                            try:
                                net.save(ONLINE_SAVE_PATH)
                            except Exception as e:
                                print(f"[OnlineTraining] Warning: couldn't save model: {e}")

                # --------------------------------------------------------------------------------------------

                # ---------------------------------------
                # Algorithm
                # ---------------------------------------
                prob.x0 = x_k
                prob._cache_u = None  # reset cache
                prob._cache_loss = None
                prob._cache_grad = None
                prob._cache_has_grad = False

                if SOLVE_OPT_TIME == True:
                    if opt.dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    U_opt, info = nlp.solve(U0_vec)

                    if opt.dev.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    solve_times_ms.append(1e3 * (t1 - t0))  # conversion to milliseconds
                else:
                    U_opt, info = nlp.solve(U0_vec)

                status = int(info.get("status", -1))
                if status < 0:
                    U_opt = U0_vec
                    print("Optimum not found!")

                # First control
                U_seq = U_opt.reshape(L, 3)
                U_guess = U_seq

                # ---------------------------------------
                # End algorithm
                # ---------------------------------------
                print("[JETSON] Control u =", U_opt)

                # Saving the controls
                u0 = U_seq[0]
                controls_hist.append(u0.copy())

                # 3) mando il controllo indietro
                send_array(conn, U_seq)

                ####### MPE #########################
                if (k >= 101):
                    traj_followed_phys = units.traj_to_phys(np.vstack(traj_followed))

                    moving_window = 100
                    MPE_current = performance_indices_computation(
                        traj_followed_phys,
                        traj_phys,
                        tt_vect,
                        moving_window=moving_window,
                    )
                    print(
                        f"\n\n MPE_current (moving window {moving_window}): {MPE_current}"
                    )
                    MPE_current_vect.append(MPE_current)

                    if k >= 120:
                        moving_window_2 = 20
                        MPE_pos_2 = np.asarray(
                            MPE_current_vect[-moving_window_2:]
                        ).mean()
                        MPE_current_mean_vect.append(MPE_pos_2)

                    else:
                        MPE_current_mean_vect.append(0)

                else:
                    MPE_current_vect.append(0)
                    MPE_current_mean_vect.append(0)
                ###########################################################

                ############### EARLY STOPPING ############################
                if EARLY_STOPPING:
                    print("MPE_current_mean_vect[-1]: ", MPE_current_mean_vect[-1])
                    if k >= (START_INSTANT_TRAINING + 250):
                        if MPE_current_mean_vect[-1] < BEST_MPE:
                            BEST_MPE = MPE_current_mean_vect[-1]
                            number_iterations = 0

                            # SNAPSHOT del best model
                            best_state_dict = copy.deepcopy(net.model.state_dict())

                        else:
                            number_iterations += 1
                            print("Number of iterations: ", number_iterations)
                            if number_iterations == PATIENCE:
                                print(
                                    f"Early stopping! Iteration number: {k} | Time [min]: {k * 5}"
                                )
                                ckpt = {"model": best_state_dict}
                                torch.save(ckpt, ONLINE_SAVE_PATH)
                                print(f"Saved best model to: {ONLINE_SAVE_PATH}")
                                ONLINE_TRAINING = False

                ###########################################################

                if (SAVE_K_ITERATION) == k:
                    # ============================================================
                    # PLOT PARAMETRI FISICI A FINE SIMULAZIONE
                    # ============================================================

                    if len(omega_vect) > 0:
                        it = np.arange(len(omega_vect))

                        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

                        # Omega
                        axs[0].plot(it, omega_vect, label="ω estimated")
                        axs[0].axhline(OMEGA_REFERENCE, linestyle="--", linewidth=1.5,
                                       label="ω reference")
                        axs[0].set_ylabel("ω [rad/s]")
                        axs[0].grid(True)
                        axs[0].legend()

                        # Mu
                        axs[1].plot(it, mu_vect, label="μ estimated", color="orange")
                        axs[1].axhline(MU_REFERENCE, linestyle="--", linewidth=1.5,
                                       color="orange", label="μ reference")
                        axs[1].set_ylabel("μ [m³/s²]")
                        axs[1].grid(True)
                        axs[1].legend()

                        # Tilt
                        axs[2].plot(it, tilt_vect, label="tilt angle estimated", color="green")
                        axs[2].axhline(TILT_REFERENCE, linestyle="--", linewidth=1.5,
                                       color="green", label="tilt angle reference")
                        axs[2].set_ylabel("Tilt [deg]")
                        axs[2].grid(True)
                        axs[2].legend()

                        # Azimuth
                        axs[3].plot(it, az_vect, label="azimuth angle estimated", color="red")
                        axs[3].axhline(AZ_REFERENCE, linestyle="--", linewidth=1.5,
                                       color="red", label="azimuth angle reference")
                        axs[3].set_ylabel("Azimuth [deg]")
                        axs[3].set_xlabel("Physics update step")
                        axs[3].grid(True)
                        axs[3].legend()

                        fig.suptitle("Online learning – Physical parameters evolution", fontsize=14)
                        plt.tight_layout()
                        plt.show()


                    savemat(
                        "evaluation_results/trajectory_followed/traj_followed_net_online.mat",
                        {"traj_followed_net": traj_followed},
                    )
                    savemat(
                        "evaluation_results/fuel_results/fuel_consumption_net_online.mat",
                        {"fuel_consumption_net": controls_hist},
                    )

                    indexes = range(np.size(residuals_phys_vect, 0))
                    print("indexes: ", indexes)
                    plt.figure(figsize=(12, 4))
                    plt.plot(indexes, residuals_phys_vect)
                    plt.title("residuals")
                    plt.xlabel("indeces")
                    plt.ylabel("")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

                    indexes = range(np.size(MPE_current_vect, 0))
                    print("indexes: ", indexes)
                    indexes = [i * 5 for i in range(np.size(MPE_current_vect, 0))]
                    plt.figure(figsize=(12, 4))
                    plt.plot(indexes, MPE_current_vect, label=f"MPE (window: {moving_window})")
                    plt.plot(indexes, MPE_current_mean_vect, label=f"MPE mean (window: {moving_window_2})")
                    plt.title("Mean Position Error (MPE)")
                    plt.xlabel("Time [min]")
                    plt.ylabel("MPE [m]")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

                    if INFERENCE_TIME == True:
                        savemat(
                            "evaluation_results/times/inference_time.mat",
                            {"inference_time": infer_times_ms},
                        )
                        print("Saved Inference Times!")

                    if SOLVE_OPT_TIME == True:
                        savemat(
                            "evaluation_results/times/solve_opt_time.mat",
                            {"solve_opt_time": solve_times_ms},
                        )
                        print("Saved Optimization Times!")

                    if ONLINE_TRAINING_TIME == True:
                        savemat(
                            "evaluation_results/times/online_time.mat",
                            {"online_time": online_times_ms},
                        )
                        print("Saved Online Training Times!")

                    # Saving the loss plot of the online learning
                    savemat(
                        "evaluation_results/online_learning/loss_online.mat",
                        {"loss_online": np.array(loss_online)}
                    )
                    plt.figure(figsize=(8, 4))
                    plt.plot(loss_online, marker="o")
                    plt.grid(True)
                    plt.xlabel("Online update index")
                    plt.ylabel("Mean training loss")
                    plt.title("Online Learning Loss (mean per update)")

                    # === Save plot ===
                    plt.savefig(
                        "evaluation_results/online_learning/loss_online.png",
                        dpi=200,
                        bbox_inches="tight"
                    )
                    plt.close()


if __name__ == "__main__":
    main()