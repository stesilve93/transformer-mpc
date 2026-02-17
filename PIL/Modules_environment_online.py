from dataclasses import dataclass
from Asteroid_scenario.net_model.pinnsformer import (
    PINNsformer,
    get_positional_encoding,
)
from Asteroid_scenario.dynamics.polyhedron_model import (
    extract_unique_edges,
    build_face_edge_map,
    preprocess_geometry,
    dynamics_propagator_body_fast,
    dynamics_propagator_body_fast_general
)
from linear_modules import *
import numpy as np
import torch
import cyipopt
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.io import loadmat
import socket
import struct


# ---------------------------------------------------------------------------
def cyclic_window(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    n = arr.shape[0]
    if n == 0 or length <= 0:
        return arr[:0, :]  # edge cases
    start = start % n
    end = start + length
    if end <= n:
        return arr[start:end, :]
    part1 = arr[start:, :]
    part2 = arr[:(end % n), :]
    return np.vstack([part1, part2])

# ---------------------------------------------------------------------------
#  Connection to JETSON
# ---------------------------------------------------------------------------
def send_int(sock, value: int):
    sock.sendall(struct.pack("!i", int(value)))

def recv_int(sock) -> int:
    data = b""
    while len(data) < 4:
        chunk = sock.recv(4 - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving int")
        data += chunk
    return struct.unpack("!i", data)[0]

def recv_all(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("connessione chiusa dal server")
        data += chunk
    return data

def recv_array(sock, dim_expected, DTYPE):
    header = recv_all(sock, 8)
    (n_bytes,) = struct.unpack("!Q", header)
    payload = recv_all(sock, n_bytes)
    arr = np.frombuffer(payload, dtype=DTYPE)
    if arr.size != dim_expected:
        raise ValueError(f"attesi {dim_expected} elementi, ricevuti {arr.size}")
    return arr

def send_array(sock, arr, DTYPE):
    arr = np.asarray(arr, dtype=DTYPE)
    payload = arr.tobytes()
    n_bytes = len(payload)
    sock.sendall(struct.pack("!Q", n_bytes))
    sock.sendall(payload)

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
                # GENERAL DYNAMICS (INCLINATION OF SPIN AXIS)#################
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


# ---------------------------------------------------------------------------
#  Optimization (objective and constraints)
# ---------------------------------------------------------------------------
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

    def loss_and_grad(self, x0: np.ndarray, U_vec: np.ndarray, need_grad: bool = True):
        """
        Computes the total cost J and its gradient with respect to the control sequence.
        The cost includes:
            • State tracking term (position/velocity error weighted by Q)
            • Control effort term (control magnitude weighted by R)
            • Control smoothness term (Δu penalization weighted by gamma_du) --> not considered currently
            • Terminal state deviation term (final error weighted by Z)
        If need_grad=True, gradients are computed using PyTorch autograd.
        """
        dev = torch.device(self.net.device)
        dtype = torch.float32
        L = U_vec.size // 3  # computing length of the sequence

        # Tensors
        x0_t = torch.as_tensor(x0, dtype=dtype, device=dev).view(1, 6)
        U_t = torch.tensor(U_vec, dtype=dtype, device=dev, requires_grad=need_grad)
        xref_t = torch.as_tensor(self.x_ref, dtype=dtype, device=dev).view(1, L, 6)
        wpos = torch.as_tensor(self.weight.w_pos_vel, dtype=dtype, device=dev).view(
            1, L, 1
        )
        wctl = torch.as_tensor(self.weight.w_control, dtype=dtype, device=dev).view(
            1, L, 1
        )

        # Prediction
        pred = self.net.predict_for_grad(x0_t, U_t)  # (1,L,6) x1...xL
        x_err = xref_t - pred

        # Weights
        Q = torch.diag(
            torch.tensor(
                [self.weight.Q_pos] * 3 + [self.weight.Q_vel] * 3,
                dtype=dtype,
                device=dev,
            )
        )
        Z = torch.diag(
            torch.tensor(
                [self.weight.Z_pos] * 3 + [self.weight.Z_vel] * 3,
                dtype=dtype,
                device=dev,
            )
        )
        R = torch.diag(torch.tensor([self.weight.R_value] * 3, dtype=dtype, device=dev))

        # Tracking
        Jx_rows = ((x_err @ Q) * x_err).sum(dim=2, keepdim=True)  # (1,L,1)
        Jx = (wpos * Jx_rows).sum()

        # Control effort
        U_seq = U_t.view(1, L, 3)
        Ju_rows = ((U_seq @ R) * U_seq).sum(dim=2, keepdim=True)
        Ju = (wctl * Ju_rows).sum()

        # Control smoothness
        # U_pad = torch.cat([U_seq[:, 0:1, :], U_seq], dim=1)
        # dU = U_pad[:, 1:, :] - U_pad[:, :-1, :]
        # Jdu = self.gamma_du * ((dU @ R) * dU).sum()

        # Terminal
        x_fin = x_err[:, -1, :].view(6)
        Jfin = x_fin @ Z @ x_fin
        # J = Jx + Ju + Jdu + Jfin
        J = Jx + Ju + Jfin

        if need_grad:
            J.backward()
            grad = U_t.grad.detach().cpu().numpy().astype(np.float64)
        else:
            grad = np.zeros_like(U_vec, dtype=np.float64)
        return float(J.detach().cpu().numpy()), grad  # Tuple[float, np.ndarray]


    def loss_and_grad_linear(
        self,
        x0: np.ndarray,
        U_vec: np.ndarray,
        A_vect_AD: np.ndarray,
        B_vect_AD: np.ndarray,
        time_instant_current: int,
        Ns: int,
        need_grad: bool = True,
    ):

        x0 = np.asarray(x0, dtype=float).reshape(
            6,
        )
        U_vec = np.asarray(U_vec, dtype=float)
        L = U_vec.size // 3
        if U_vec.size != 3 * L:
            raise ValueError(
                f"U_vec.size deve essere multiplo di 3, trovato {U_vec.size}"
            )
        U = U_vec.reshape(L, 3)

        # x_ref already windowed in MPC.run: must be (L,6)
        xref = np.asarray(self.x_ref, dtype=float)
        if xref.shape != (L, 6):
            raise ValueError(
                f"x_ref deve avere shape (L,6) = ({L},6); trovato {xref.shape}"
            )

        # WEIGHTS
        Q = np.diag([self.weight.Q_pos] * 3 + [self.weight.Q_vel] * 3)
        Z = np.diag([self.weight.Z_pos] * 3 + [self.weight.Z_vel] * 3)
        R = np.diag([self.weight.R_value] * 3)


        def objective_only(U_flat: np.ndarray) -> float:
            UU = U_flat.reshape(L, 3)
            x_sim = np.zeros((L, 6), dtype=float)
            x_j = x0.copy()

            start_idx = int(time_instant_current) - 1  # 1-based -> 0-based
            for j in range(L):
                idx = start_idx + j
                if idx < 0 or idx >= A_vect_AD.shape[2]:
                    raise IndexError(
                        f"Indice A/B fuori range: idx={idx}, max={A_vect_AD.shape[2] - 1}"
                    )
                A_k = A_vect_AD[:, :, idx]
                B_k = B_vect_AD[:, :, idx]
                x_j = A_k @ x_j + B_k @ UU[j]
                x_sim[j, :] = x_j


            x_err = xref - x_sim

            Jx_rows = np.sum((x_err @ Q) * x_err, axis=1)
            Jx = float(np.sum(Jx_rows))

            Ju_rows = np.sum((UU @ R) * UU, axis=1)
            Ju = float(np.sum(Ju_rows))

            x_err_final = x_sim[-1, :] - xref[-1, :]
            Jfin = float(x_err_final @ Z @ x_err_final)

            return Jx + Ju + Jfin

        J0 = objective_only(U_vec)

        if not need_grad:
            return J0, np.zeros_like(U_vec, dtype=float)

        # finite difference gradient
        g = np.zeros_like(U_vec, dtype=float)
        eps0 = 1e-6
        for i in range(U_vec.size):
            ui = U_vec[i]
            eps = eps0 * max(1.0, abs(ui))
            U_pert = U_vec.copy()
            U_pert[i] = ui + eps
            Ji = objective_only(U_pert)
            g[i] = (Ji - J0) / eps

        return J0, g


# ---------------------------------------------------------------------------
#  IPOPT problem
# ---------------------------------------------------------------------------
class IpoptProblem:
    def __init__(
        self,
        x0: np.ndarray,
        U0: np.ndarray,
        bounds: Bounds,
        opt: Optimization,
        scenario: str,
        current_instant: int,
        A_vect_AD: np.ndarray,
        B_vect_AD: np.ndarray,
        Ns: int,
    ):
        self.x0 = x0.astype(float)
        self.U = U0.astype(float)
        self.bounds = bounds
        self.opt = opt
        self.Ns = Ns
        self.m = 0  # no constraints
        self.scenario = scenario
        self.current_instant = current_instant
        self.A_vect_AD = A_vect_AD
        self.B_vect_AD = B_vect_AD

    # Objective
    def objective(self, u):
        if self.scenario == "Neural_Network":
            loss, _ = self.opt.loss_and_grad(self.x0, u, need_grad=False)
        else:
            loss, _ = self.opt.loss_and_grad_linear(
                self.x0,
                u,
                A_vect_AD=self.A_vect_AD,
                B_vect_AD=self.B_vect_AD,
                time_instant_current=self.current_instant,
                Ns=self.Ns,
                need_grad=False,
            )
        return float(loss)



    def gradient(self, u):
        if self.scenario == "Neural_Network":
            _, g = self.opt.loss_and_grad(self.x0, u, need_grad=True)
        else:
            _, g = self.opt.loss_and_grad_linear(
                self.x0,
                u,
                A_vect_AD=self.A_vect_AD,
                B_vect_AD=self.B_vect_AD,
                time_instant_current=self.current_instant,
                Ns=self.Ns,
                need_grad=True,
            )
        return g

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

        #  Building the reference window (cyclic around current step)
        N = len(traj_ref_full)
        s = step % N
        idxs = [(s + i) % N for i in range(-self.win_back, self.win_ahead + 1)]
        ref_slice = traj_ref_full[idxs, :]


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
        JETSON_IP: str,
        PORT: int,
        DTYPE: np.dtype,
        N_STATE: int,
        N_CTRL: int,
        T_per_orbit=None,
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
        self.JETSON_IP = JETSON_IP
        self.PORT = PORT
        self.DTYPE = DTYPE
        self.N_STATE = N_STATE
        self.N_CTRL = N_CTRL
        self.T_per_orbit = T_per_orbit

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

        ################################################################################################
        # Connection with JETSON
        ################################################################################################
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"[PC] Connecting {self.JETSON_IP}:{self.PORT}...")
            s.connect((self.JETSON_IP, self.PORT))
            print("[PC] Connected!")

            for k in range(1, T_steps_ref + 1):
                # for k in range(0, T_steps_ref + 1):
                if k + self.N_s > T_steps_ref:
                    break

                # Modifying the reference trajectory inside the optimizer
                x_ref_window = cyclic_window(trajectory_adim, start=k, length=L)  # shape (L,6)
                self.opt.x_ref = x_ref_window

                U0_vec = U_guess.reshape(-1)


                # Rotation of the initial state (only if R_mat not "None")
                if R_mat is not None:
                    x_k = self.state_rotation(x_k, R_mat)

                ################################################################################################
                # Interaction with JETSON
                ################################################################################################
                print(f"\n[PC] Environment state {k}, x_k =", x_k)

                # 1) sending the state to the jetson
                send_int(s, k)
                send_array(s, x_k, self.DTYPE)

                # 2) receiving the control from Jetson
                U_opt = recv_array(s, self.N_CTRL, self.DTYPE)
                print("[PC] Control received =", U_opt)


                ###############################################################################################
                U_seq = U_opt.reshape(L, 3)
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

                N_T = len(tt_vect)

                i0 = (k - 1) % N_T
                i1 = k % N_T

                t0 = float(tt_vect[i0])
                t1 = float(tt_vect[i1])

                # number of orbits completed (needed when k goes beyond N)
                wrap0 = (k - 1) // N_T
                wrap1 = k // N_T

                t0 = t0 + wrap0 * self.T_per_orbit
                t1 = t1 + wrap1 * self.T_per_orbit
                print("t0=",t0)
                print("t1=",t1)


                seq = np.array([t0, t1], dtype=float)


                print("k", k, "seq", seq, "len", len(seq), "dt", (seq[1]-seq[0] if len(seq)==2 else None), flush=True)

                x_step_traj = self.dyn.compute(x_tmp, u0, seq)
                x_k = x_step_traj[-1]

                # Saving control and state in "real" frame
                traj_followed.append(x_k.copy())
                controls_hist.append(u0.copy())


                if k % 500 == 0:
                    traj_followed_plot = self.units.traj_to_phys(np.vstack(traj_followed))
                    traj_phys_plot = self.units.traj_to_phys(trajectory_adim)
                    plot_and_save_performance_indices(traj_followed_phys=traj_followed_plot, traj_phys=traj_phys_plot, tt_vect=tt_vect)

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


def plot_and_save_performance_indices(
    traj_followed_phys,
    traj_phys,
    tt_vect,
    n_sim=None,
    end_win=None,
    moving_window=8,
    save_path="performance_indices.png",
    dpi=200,
    show=True,
):
    """
    Calcola gli indici di performance (APE/MPE/RPE/VAR) confrontando una traiettoria seguita con una reference
    (ripetuta in wrap se più corta) e salva un plot con 3 subplot (posizione).

    Parameters
    ----------
    traj_followed_phys : (N, m) array-like
        Traiettoria seguita in unità fisiche.
    traj_phys : (K, m) array-like
        Reference in unità fisiche (può essere più corta di N, viene ripetuta).
    tt_vect : (T,) array-like
        Vettore tempo della reference (serve per stimare dt).
    performance_indices_vectors : callable
        Funzione che ritorna (APE_pos, APE_vel, MPE_pos, RPE_pos, MPE_vel, RPE_vel, VAR_pos, VAR_vel)
    n_sim : int or None
        Lunghezza simulazione da considerare. Default: len(traj_followed_phys).
    end_win : int or None
        Finestra finale (numero campioni) da usare nel confronto. Default: n_sim.
    moving_window : int
        Moving window per gli indici.
    save_path : str
        Path dove salvare l'immagine.
    dpi : int
        DPI del file salvato.
    show : bool
        Se True, fa anche plt.show().

    Returns
    -------
    fig, axs, metrics : (matplotlib Figure, Axes array, dict)
        metrics contiene gli 8 vettori di output.
    """
    traj_followed_phys = np.asarray(traj_followed_phys)
    traj_phys = np.asarray(traj_phys)
    tt_vect = np.asarray(tt_vect)

    if n_sim is None:
        n_sim = len(traj_followed_phys)
    n_sim = int(n_sim)

    if end_win is None:
        end_win = n_sim
    end_win = int(end_win)

    if n_sim <= 0 or end_win <= 0:
        raise ValueError("n_sim ed end_win devono essere > 0")
    if end_win > n_sim:
        raise ValueError("end_win non può essere maggiore di n_sim")

    # Reference wrap (tile) per coprire n_sim
    n_ref = len(traj_phys)
    if n_ref == 0:
        raise ValueError("traj_phys è vuota")
    reps = int(np.ceil(n_sim / n_ref))

    traj_followed_phys_cmp = traj_followed_phys[:n_sim]
    traj_phys_cmp = np.tile(traj_phys, (reps, 1))[:n_sim]

    # dt dal vettore tempo reference
    dt = float(tt_vect[1] - tt_vect[0]) if len(tt_vect) > 1 else 0.0
    tt_cmp = np.arange(n_sim, dtype=float) * dt
    time_min = (tt_cmp - tt_cmp[0]) / 60.0

    # Indici
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
        traj_followed_phys_cmp, traj_phys_cmp[:end_win], 1, end_win, moving_window=moving_window
    )

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))

    axs[0].plot(time_min[:len(APE_pos)], APE_pos, label="APE pos")
    axs[0].plot(time_min[:len(MPE_pos)], MPE_pos, label="MPE pos")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylabel("Error [m]")
    axs[0].set_title("Absolute & Mean Position Error")

    axs[1].plot(time_min[:len(RPE_pos)], RPE_pos)
    axs[1].grid(True)
    axs[1].set_ylabel("Error [m]")
    axs[1].set_title("Relative Position Error (RPE)")

    axs[2].plot(time_min[:len(VAR_pos)], VAR_pos)
    axs[2].grid(True)
    axs[2].set_xlabel("Time [min]")
    axs[2].set_ylabel("Variance [m^2]")
    axs[2].set_title("Position Error Variance")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return