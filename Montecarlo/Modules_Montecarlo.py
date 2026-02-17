from dataclasses import dataclass
import numpy as np
import torch
import cyipopt
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.io import loadmat
import time

from Asteroid_scenario.dynamics.polyhedron_model import (
    extract_unique_edges,
    build_face_edge_map,
    preprocess_geometry,
    dynamics_propagator_body_fast,
    dynamics_propagator_body_fast_general
)

from Asteroid_scenario.net_model.pinnsformer import (
    PINNsformer,
    get_positional_encoding,
)

from linear_modules import linear_propagation
# -------------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------------
INFERENCE_TIME = False
SOLVE_OPT_TIME = False

PRINT_LEVEL = 0

infer_times_ms = []
solve_times_ms = []

# ---------------------------------------------------------------------------
#  Code parameters classes
# ---------------------------------------------------------------------------
@dataclass
class ParamsPhys:
    rho_real: float  # kg/m^3
    omega_real: float  # rad/s
    G: float  # m^3 kg^-1 s^-2


@dataclass
class Units:
    LU: float  # length unit (m)
    VU: float  # velocity unit (m/s)
    CU: float  # control unit (m/s)

    def state_to_phys(self, x_adim: np.ndarray):
        x = np.array(x_adim, dtype=float).copy()
        x[..., 0:3] *= self.LU
        x[..., 3:6] *= self.VU
        return x

    def state_to_adim(self, x_phys: np.ndarray):
        x = np.array(x_phys, dtype=float).copy()
        x[..., 0:3] /= self.LU
        x[..., 3:6] /= self.VU
        return x

    def traj_to_phys(self, traj_adim: np.ndarray):
        return self.state_to_phys(traj_adim)

    def traj_to_adim(self, traj_phys: np.ndarray):
        return self.state_to_adim(traj_phys)

    def control_to_adim(self, u_phys: np.ndarray):
        return np.array(u_phys, dtype=float) / self.CU

    def control_to_phys(self, u_adim: np.ndarray):
        return np.array(u_adim, dtype=float) * self.CU


@dataclass
class Weights:
    # weight matrices values
    Q_pos: float
    Q_vel: float
    Z_pos: float
    Z_vel: float
    R_value: float
    w_control: (
        np.ndarray
    )  # (L,) time-varying weights for control effort term (one per step)
    w_pos_vel: (
        np.ndarray
    )  # (L,)  time-varying weights for position/velocity tracking term (one per step)


@dataclass
class Bounds:
    lower: np.ndarray  # (3L,)
    upper: np.ndarray  # (3L,)


# ---------------------------------------------------------------------------
#  Asteroid model
# ---------------------------------------------------------------------------
@dataclass
class AsteroidModel:
    params: ParamsPhys
    vertices: np.ndarray  # (Nv,3) in meters
    faces: np.ndarray  # (Nf,3) idx int


# ---------------------------------------------------------------------------
#  Dynamics propagator
# ---------------------------------------------------------------------------
class DynamicsPropagator:
    def __init__(self, options, units: Units, model: AsteroidModel, change_spin_axis=None):
        self.options = options  # dictionary
        self.units = units
        self.model = model
        self.change_spin_axis = change_spin_axis

        # Pre-processing:
        self.edges = extract_unique_edges(self.model.faces)
        self.edge_faces = build_face_edge_map(self.model.faces, self.edges)
        self.face_normals, self.face_centroids = preprocess_geometry(
            self.model.vertices, self.model.faces
        )

    def compute(self, x0_adim: np.ndarray, U_vec_adim: np.ndarray, tt_win: np.ndarray):
        """
        Propagate physically from adimensional state x0 and adimensional controls.
        Returns adimensional trajectory with shape (L+1, 6) including x0.
        L --> number of propagation instants
        tt_win --> finestra di istanti di tempo in cui abbiamo la propagazione
        """
        L = len(U_vec_adim) // 3
        U = U_vec_adim.reshape(L, 3)
        traj = [x0_adim.astype(float)]
        x = x0_adim.astype(float).copy()

        x = self.units.state_to_phys(x)

        for k in range(L):
            u_phys = self.units.control_to_phys(U[k, :])
            x[3:6] = x[3:6] + u_phys

            t_span = tt_win[k : k + 2]

            if self.change_spin_axis is None:
                _, x = dynamics_propagator_body_fast(
                    x,
                    t_span,
                    self.model.params.omega_real,
                    self.model.vertices,
                    self.model.faces,
                    self.edges,
                    self.edge_faces,
                    self.face_normals,
                    self.face_centroids,
                    self.model.params.rho_real,
                    self.model.params.G,
                )
            else:
                # General Dynamics (spin axis non aligned with z-axis)
                _, x = dynamics_propagator_body_fast_general(
                    x,
                    t_span,
                    self.model.params.omega_real,
                    self.model.vertices,
                    self.model.faces,
                    self.edges,
                    self.edge_faces,
                    self.face_normals,
                    self.face_centroids,
                    self.model.params.rho_real,
                    self.model.params.G,
                )
                ########################################################

            x_ad = self.units.state_to_adim(x.copy())
            traj.append(x_ad.copy())

        return np.vstack(traj)


# ---------------------------------------------------------------------------
#  Neural network
# ---------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self, model_path: str, device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self, time_instants: int = 16):
        """
        Builds the PINNsformer architecture with predefined hyperparameters.
        """
        d_model = 512
        Pos_src = get_positional_encoding(time_instants - 1, d_model).to(self.device)
        net = PINNsformer(
            d_model=d_model,
            d_hidden=4 * d_model,
            d_emb_input=9,
            d_final=6,
            N=2,
            heads=8,
            dropout=0,
            Pos_src=Pos_src,
        ).to(self.device)
        return net

    def load_model(self):
        """
        loads model weights from the pre-trained model file and sets the network to evaluation mode.
        """
        net = self.build_model(time_instants=16)
        reference_model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        net.load_state_dict(reference_model["model"])
        net.eval().to(self.device)
        self.model = net

    @torch.no_grad()
    def predict_trajectory(self, x0: np.ndarray, U_vec: np.ndarray, TIME=16):
        """
        Performs forward inference without gradient tracking, returning the full predicted trajectory (including x0) as a NumPy array.
        """
        assert self.model is not None, "Call load_model() first"
        xb = torch.as_tensor(x0, dtype=torch.float32, device=self.device).view(1, 6)

        U = torch.as_tensor(U_vec.copy(), dtype=torch.float32, device=self.device).view(
            1, TIME - 1, 3
        )
        deltas = self.model(xb, U)  # (1,L,6) predicts Δx
        traj = torch.cat(
            [xb.unsqueeze(1), xb.unsqueeze(1) + deltas.cumsum(dim=1)], dim=1
        )
        return traj.squeeze(0).cpu().numpy()  # (TIME,6)

    def predict_for_grad(self, x0: torch.Tensor, U_flat: torch.Tensor):
        """
        Performs forward inference with gradient tracking enabled, returning a torch.Tensor used for gradient-based optimization (does not include x0).
        """
        assert self.model is not None, "Call load_model() first"
        L = U_flat.numel() // 3
        U = U_flat.view(1, L, 3)
        deltas = self.model(x0, U)
        pred = x0.unsqueeze(1) + deltas.cumsum(dim=1)
        return pred  # (1,L,6)

# --------------------------------------------------------------------------------

class Optimization:
    def __init__(
        self,
        weight: Weights,
        units: Units,
        x_ref: np.ndarray,
        net: NeuralNetwork,
        # gamma_du: float = 0.0,  # --> not used currently
    ):
        self.weight = weight
        self.units = units
        self.x_ref = x_ref
        self.net = net
        # self.gamma_du = gamma_du

        # Initialisation of the parameters
        self.dev = torch.device(self.net.device)
        self.dtype = torch.float32
        self.L = 15

        # Weights
        self.wpos = torch.as_tensor(
            self.weight.w_pos_vel, dtype=self.dtype, device=self.dev
        ).view(1, self.L, 1)
        self.wctl = torch.as_tensor(
            self.weight.w_control, dtype=self.dtype, device=self.dev
        ).view(1, self.L, 1)

        self.Q = torch.diag(
            torch.tensor(
                [self.weight.Q_pos] * 3 + [self.weight.Q_vel] * 3,
                dtype=self.dtype,
                device=self.dev,
            )
        )
        self.Z = torch.diag(
            torch.tensor(
                [self.weight.Z_pos] * 3 + [self.weight.Z_vel] * 3,
                dtype=self.dtype,
                device=self.dev,
            )
        )
        self.R = torch.diag(
            torch.tensor([self.weight.R_value] * 3, dtype=self.dtype, device=self.dev)
        )

        self.q_vec = torch.tensor(
            [self.weight.Q_pos] * 3 + [self.weight.Q_vel] * 3,
            device=self.dev,
            dtype=self.dtype,
        ).view(1, 1, 6)
        self.z_vec = torch.tensor(
            [self.weight.Z_pos] * 3 + [self.weight.Z_vel] * 3,
            device=self.dev,
            dtype=self.dtype,
        )
        self.r_vec = torch.tensor(
            [self.weight.R_value] * 3, device=self.dev, dtype=self.dtype
        ).view(1, 1, 3)

    def set_x_ref(self):
        self.xref_t = torch.as_tensor(
            self.x_ref, dtype=self.dtype, device=self.dev
        ).view(1, self.L, 6)


class IpoptProblem:
    def __init__(
        self,
        x0: np.ndarray,
        U0: np.ndarray,
        bounds: Bounds,
        opt: Optimization,
        scenario: str,
        Ns: int,
    ):
        self.x0 = x0.astype(float)
        self.U = U0.astype(float)
        self.bounds = bounds
        self.opt = opt
        self.Ns = Ns
        self.m = 0  # no constraints
        self.scenario = scenario

        # cache
        self._cache_u = None  # np.ndarray
        self._cache_loss = None  # float
        self._cache_grad = None  # np.ndarray
        self._cache_has_grad = False  # bool

    def _u_equal(self, u):
        return self._cache_u is not None and np.array_equal(u, self._cache_u)

    def _compute(self, u, need_grad: bool):
        if self.scenario == "Neural_Network":
            return self.opt.loss_and_grad(self.x0, u, need_grad=need_grad)
        else:
            assert "Error in the selection of the scenario variable!"

    def _ensure_cache(self, u, want_grad: bool):
        """
        Garantisce che in cache ci sia almeno la loss, e se want_grad=True anche il grad.
        """
        if self._u_equal(u):
            # if i want grad and i don't have it, I compute it one time
            if want_grad and not self._cache_has_grad:
                loss, grad = self._compute(u, need_grad=True)
                self._cache_loss = float(loss)
                self._cache_grad = np.asarray(grad, dtype=np.float64).copy()
                self._cache_has_grad = True
            return

        # u different: reset cache and compute what is needed
        self._cache_u = np.asarray(u, dtype=np.float64).copy()
        if want_grad:
            loss, grad = self._compute(u, need_grad=True)
            self._cache_loss = float(loss)
            self._cache_grad = np.asarray(grad, dtype=np.float64).copy()
            self._cache_has_grad = True
        else:
            loss, _ = self._compute(u, need_grad=False)
            self._cache_loss = float(loss)
            self._cache_grad = None
            self._cache_has_grad = False

    # Objective
    def objective(self, u):
        self._ensure_cache(u, want_grad=False)
        return self._cache_loss

    # Gradient
    def gradient(self, u):
        self._ensure_cache(u, want_grad=True)
        return self._cache_grad

    # Constraints (none)
    def constraints(self, u):
        return np.zeros((0,), dtype=np.float64)

    def jacobian(self, u):
        return np.zeros((0,), dtype=np.float64)

    def jacobianstructure(self):
        return (np.array([], dtype=int), np.array([], dtype=int))

    def hessianstructure(self):
        return (np.array([], dtype=int), np.array([], dtype=int))

    def hessian(self, u, lagrange, obj_factor):
        return np.zeros((0,), dtype=np.float64)


class Algorithm:
    def __init__(
        self,
        units,
        N_s,
        N_control,
        weights,
        net,
        bounds,
        opt,
        traj_phys,
        traj_adim,
        tt_vect,
        scenario,
    ):
        self.units = units
        self.N_s = N_s
        self.N_control = N_control
        self.weights = weights
        self.net = net
        self.bounds = bounds
        self.opt = opt
        self.traj_phys = traj_phys
        self.traj_adim = traj_adim
        self.tt_vect = tt_vect
        self.scenario = scenario

    def initialisation(self):
        max_iter = 2000
        print_level = PRINT_LEVEL
        options_solv = {
            "max_iter": max_iter,
            "print_level": print_level,
        }
        # ---------------------------------------
        # Definition of the IPOPT problem
        # --------------------------------------
        U_guess = np.zeros((self.N_control, 3), dtype=float)
        self.U_guess = U_guess
        x_k = self.traj_adim[0].copy()

        prob = IpoptProblem(
            x_k,  # x_k
            U_guess,  # U0_vec
            self.bounds,
            self.opt,
            scenario=self.scenario,
            Ns=self.N_s,
        )

        nlp = cyipopt.Problem(
            n=(15 * 3),  # U0_vec.size
            m=prob.m,
            problem_obj=prob,
            lb=self.bounds.lower,
            ub=self.bounds.upper,
        )
        # IPOPT options
        nlp.add_option("tol", 1e-6)
        nlp.add_option("max_iter", int(options_solv.get("max_iter", 2000)))
        nlp.add_option("hessian_approximation", "limited-memory")
        nlp.add_option("print_level", int(options_solv.get("print_level", 5)))

        self.prob = prob
        self.nlp = nlp

    def run(self, x_k, k, U_guess):
        self.prob.x0 = x_k

        x_ref_window = self.traj_adim[k : (k + self.N_s - 1), :]  # (L,6)
        self.opt.x_ref = x_ref_window
        self.opt.set_x_ref()

        # U0_vec = self.U_guess.reshape(-1)
        U0_vec = U_guess.reshape(-1)

        # ---------------------------------------
        # Algorithm
        # ---------------------------------------
        self.prob.x0 = x_k
        self.prob._cache_u = None  # reset cache
        self.prob._cache_loss = None
        self.prob._cache_grad = None
        self.prob._cache_has_grad = False

        # print("x0: ", self.prob.x0)

        if SOLVE_OPT_TIME == True:
            if self.opt.dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            U_opt, info = self.nlp.solve(U0_vec)

            if self.opt.dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            solve_times_ms.append(1e3 * (t1 - t0))  # conversion to milliseconds
        else:
            U_opt, info = self.nlp.solve(U0_vec)

        status = int(info.get("status", -1))
        if status < 0:
            U_opt = U0_vec
            print("Optimum not found!")

        # First control
        U_seq = U_opt.reshape(self.N_control, 3)
        #self.U_guess = U_seq
        return U_seq, U_opt



# -----------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  MPC main class
# ---------------------------------------------------------------------------
class MPC:
    def __init__(
        self,
        N_s: int,
        N_control: int,
        bounds: Bounds,
        weight: Weights,
        net: NeuralNetwork,
        opt: Optimization,
        options_solv,  # dictionary
        units: Units,
        dyn: DynamicsPropagator,
        scenario: str,
        algorithm: Algorithm,

    ):
        self.N_s = N_s
        self.N_control = N_control
        self.bounds = bounds
        self.weight = weight
        self.net = net
        self.opt = opt
        self.options_solv = options_solv
        self.units = units
        self.dyn = dyn
        self.scenario = scenario

        algorithm.initialisation()
        self.algorithm=algorithm


    def state_rotation(self, x: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Rotate the state (6,) or control (3,) using matrix R (3x3).

        - State:  x = [x, y, z, vx, vy, vz]
                  => [R @ r, R @ v]
        - Control: x = [ux, uy, uz]
                  =>  R @ x
        """
        x = np.asarray(x, float)
        R = np.asarray(R, float)

        if R.shape != (3, 3):
            raise ValueError("R must be a 3x3 matrix.")

        if x.shape == (6,):
            r = R @ x[:3]
            v = R @ x[3:]
            return np.hstack((r, v))

        elif x.shape == (3,):
            return R @ x

        else:
            raise ValueError(
                f"state_rotation supports only (6,) or (3,) vectors, but got {x.shape}"
            )


    def run(
        self,
        T_steps_ref: int,
        x0: np.ndarray,
        trajectory_adim: np.ndarray,
        tt_vect: np.ndarray,
        vis=None,
        A_vect_AD=None,
        B_vect_AD=None,
        R_mat=None,
    ):
        """
        Executes the full MPC loop over the reference trajectory.
        For each time step:
        • Defines a prediction window of length N_s based on the reference trajectory.
        • Solves the constrained optimization problem (via IPOPT) to find the optimal
        control sequence U_opt that minimizes the defined cost function.
        • Applies the first control action to the current state and propagates
        the system forward one step using the real physical dynamics.
        • Optionally updates a 3D visualizer to animate the tracking process.
        Returns:
        - traj_followed_adim: the sequence of followed states (adimensional)
        - controls_history_adim: the applied control inputs (adimensional)
        """
        L = self.N_control
        U_guess = np.zeros((L, 3), dtype=float)
        x_k = x0.copy()
        traj_followed = [x_k.copy()]
        controls_hist = []

        for k in range(1, T_steps_ref + 1):
            # for k in range(0, T_steps_ref + 1):
            if k + self.N_s > T_steps_ref:
                break

            print("Iteration: ", k)

            U_seq, U_opt = self.algorithm.run(x_k, k, U_guess)
            U_guess = U_seq

            ####################################################################################
            #  SIMULATION & VISUALISATION SECTION
            #  The following block is NOT part of the flight algorithm
            #  It is included for simulation and visualisation purposes
            ####################################################################################

            x_prev = x_k.copy()

            if self.scenario == "Neural_Network":
                traj_net = self.net.predict_trajectory(x_k, U_opt)
            else:
                # time_instant_index --> index of the current time instant
                # N_prop --> number of instant in which we have to propagate (no initial state)
                time_instant_index = k
                traj_lin = linear_propagation(
                    x_k,
                    U_seq.T,
                    time_instant_index,
                    self.N_control,
                    A_vect_AD,
                    B_vect_AD,
                )

            # Apply first control and propagate
            u0 = U_seq[0]

            x_tmp = x_k.copy()

            # Rotation of the state and control (go back to "real" frame)
            if R_mat is not None:
                x_tmp = self.state_rotation(x_tmp, R_mat.T)
                u0 = self.state_rotation(u0, R_mat.T)

            seq = tt_vect[k - 1 : k + 1]
            x_step_traj = self.dyn.compute(x_tmp, u0, seq)
            x_k = x_step_traj[-1]

            # Saving control and state in "real" frame
            traj_followed.append(x_k.copy())
            controls_hist.append(u0.copy())

            # Updating the live plot
            if vis is not None:
                if self.scenario == "Neural_Network":
                    vis.update(
                        step=k,
                        x_curr=x_prev,
                        traj_ref_full=trajectory_adim,
                        traj_net_window=traj_net,
                    )
                else:
                    vis.update(
                        step=k,
                        x_curr=x_prev,
                        traj_ref_full=trajectory_adim,
                        traj_net_window=traj_lin,
                    )
            ####################################################################################

        return np.vstack(traj_followed), (np.vstack(controls_hist))


# ---------------------------------------------------------------------------
#  Utilities: loaders and metrics
# ---------------------------------------------------------------------------


def load_reference(traj_mat_path: str, tt_mat_path: str, units: Units):
    m_traj = loadmat(traj_mat_path)
    m_tt = loadmat(tt_mat_path)
    trajectory_phys = np.array(m_traj.get("xx_final"), dtype=float)
    tt_vect = np.array(m_tt.get("tt")).reshape(-1)
    # adimensionalise
    traj_adim = units.traj_to_adim(trajectory_phys)
    return trajectory_phys, traj_adim, tt_vect


def performance_indices_vectors(
    traj_pred_phys: np.ndarray,
    traj_real_phys: np.ndarray,
    start_win: int,
    end_win: int,
    moving_window: int,
):
    E = traj_real_phys - traj_pred_phys
    e_pos = E[start_win - 1 : end_win, 0:3]
    e_vel = E[start_win - 1 : end_win, 3:6]
    APE_pos = np.linalg.norm(e_pos, axis=1)
    APE_vel = np.linalg.norm(e_vel, axis=1)

    # moving mean/variance
    def movmean(x, w):
        out = np.zeros_like(x)
        for i in range(len(x)):
            a = max(0, i - (w // 2))
            b = min(len(x), i + (w // 2) + 1)
            out[i] = x[a:b].mean()
        return out

    def movvar(x, w):
        out = np.zeros_like(x)
        for i in range(len(x)):
            a = max(0, i - (w // 2))
            b = min(len(x), i + (w // 2) + 1)
            out[i] = x[a:b].var(ddof=0)
        return out

    MPE_pos = movmean(APE_pos, moving_window)
    MPE_vel = movmean(APE_vel, moving_window)
    VAR_pos = movvar(APE_pos, moving_window)
    VAR_vel = movvar(APE_vel, moving_window)
    RPE_pos = APE_pos - MPE_pos
    RPE_vel = APE_vel - MPE_vel
    return APE_pos, APE_vel, MPE_pos, RPE_pos, MPE_vel, RPE_vel, VAR_pos, VAR_vel


# ---------------------------------------------------------------------------
#  Trajectory tracking animation
# ---------------------------------------------------------------------------
class Visualizer3D:
    def __init__(
        self,
        units: Units,
        title="MPC Tracking",
        zoom_range_phys_follow=300.0,
        zoom_range_phys=10_000,
        fps=4,
        video_out=None,  # if you want to save the video, you need to specify here the name of the file where the video will be saved
        win_back=10,
        win_ahead=30,
        follow=False,
    ):
        self.units = units
        self.zoom = float(zoom_range_phys)  # [m]
        self.zoom_follow = float(zoom_range_phys_follow)  # [m]
        self.video_out = video_out
        self.fps = int(fps)
        self.win_back = int(win_back)
        self.win_ahead = int(win_ahead)
        self.follow = follow

        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(title)
        self.ax.grid(True)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")

        self.line_ref_win = None
        self.line_followed = None
        self.line_net = None
        self.scatter_sat = None

        self.writer = None
        if self.video_out is not None:
            self.writer = FFMpegWriter(fps=self.fps)
            self.writer.setup(self.fig, self.video_out, dpi=150)

    def _to_phys_pos_only(self, arr_adim_6):
        arr = np.asarray(arr_adim_6, dtype=float)
        out = arr.copy()
        out[..., 0:3] = out[..., 0:3] * self.units.LU
        return out[..., 0:3]

    def update(
        self,
        step,
        x_curr,
        traj_ref_full,
        traj_net_window,
    ):
        #  Building the reference window
        N = len(traj_ref_full)
        i0 = max(0, step - self.win_back)
        i1 = min(N - 1, step + self.win_ahead)
        ref_slice = traj_ref_full[i0 : i1 + 1, :]

        pos_ref_win = self._to_phys_pos_only(ref_slice)  # [m]
        pos_net = (
            self._to_phys_pos_only(traj_net_window)
            if traj_net_window is not None
            else None
        )
        pos_curr = self._to_phys_pos_only(x_curr.reshape(1, 6))[0]

        # Initialisation
        if self.line_ref_win is None:
            (self.line_ref_win,) = self.ax.plot(
                [], [], [], "--", lw=1.3, color=(0.3, 0.3, 0.3), label="Reference"
            )

        if self.line_net is None:
            (self.line_net,) = self.ax.plot(
                [], [], [], "-", lw=1.6, label="NET prediction"
            )
        if self.scatter_sat is None:
            self.scatter_sat = self.ax.scatter(
                [], [], [], s=40, color="C1", label="Current state"
            )

        # Updating
        self.line_ref_win.set_data(pos_ref_win[:, 0], pos_ref_win[:, 1])
        self.line_ref_win.set_3d_properties(pos_ref_win[:, 2])
        if pos_net is not None:
            self.line_net.set_data(pos_net[:, 0], pos_net[:, 1])
            self.line_net.set_3d_properties(pos_net[:, 2])
        else:
            self.line_net.set_data([], [])
            self.line_net.set_3d_properties([])

        self.scatter_sat._offsets3d = (
            np.array([pos_curr[0]]),
            np.array([pos_curr[1]]),
            np.array([pos_curr[2]]),
        )

        # Zoom and window center
        if pos_net is not None and len(pos_net) > 0:
            cent_pos = max(0, min(len(pos_net) - 1, len(pos_net) // 2 - 2))
            center = pos_net[cent_pos]
        else:
            center = pos_curr

        if self.follow:
            xlim = (center[0] - self.zoom, center[0] + self.zoom)
            ylim = (center[1] - self.zoom, center[1] + self.zoom)
            zlim = (center[2] - self.zoom, center[2] + self.zoom)
        else:
            xlim = (-self.zoom, +self.zoom)
            ylim = (-self.zoom, +self.zoom)
            zlim = (-self.zoom, +self.zoom)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)

        if step == 1:
            self.ax.legend(loc="best")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.writer is not None:
            self.writer.grab_frame()
        plt.pause(0.001)

    def close(self):
        if self.writer is not None:
            self.writer.finish()
            print(f"Video saved in: {self.video_out}")

    def reset(self):
        self.ax.cla()
        self.ax.set_title("MPC Tracking")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.ax.set_box_aspect([1, 1, 1])
        self.line_ref_win = None
        self.line_net = None
        self.scatter_sat = None