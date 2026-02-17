# ---------------------------------------------------------------------------
# MPC MODULES
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from Python_MPC.Asteroid_scenario.net_model.pinnsformer import (
    PINNsformer,
    get_positional_encoding,
)
from Python_MPC.Asteroid_scenario.dynamics.polyhedron_model import (
    extract_unique_edges,
    build_face_edge_map,
    preprocess_geometry,
    dynamics_propagator_body_fast,
    dynamics_propagator_body_fast_general
)
from Python_MPC.Montecarlo.on_board_fast import Algorithm, IpoptProblem
from Python_MPC.linear_modules import *
import numpy as np
import torch
import cyipopt
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.io import loadmat
import socket
import struct


# ---------------------------------------------------------------------------
#  Connection to JETSON
# ---------------------------------------------------------------------------
def send_int(sock, value: int):
    sock.sendall(struct.pack("!i", int(value)))  # network order int32

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
                # NEW DYNAMICS (INCLINATION OF SPIN AXIS)#################
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

        # NOTE: METTO LA COPIA DI U_vec perchè essendo passata da un buffer (dopo essere stata inviata da jetson)
        # è una variabile read-only, se poi proviamo a modificarla abbiamo un comportamento indefinito
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

    # def loss_and_grad_linear(
    #         self,
    #         x0: np.ndarray,
    #         U_vec: np.ndarray,
    #         A_vect_AD: np.ndarray,
    #         B_vect_AD: np.ndarray,
    #         time_instant_current: int,
    #         Ns: int,
    #         need_grad: bool = True,
    # ):
    #     """
    #     Versione Python della objFun MATLAB, con gradiente per differenze finite.
    #
    #     Parametri
    #     ---------
    #     x0 : (6,) stato iniziale (vettore riga o colonna, verrà reshaped)
    #     U_vec : (3*Ns,) vettore dei controlli concatenati
    #     A_vect_AD : (6,6, N-1) matrici A adimensionali discretizzate
    #     B_vect_AD : (6,3, N-1) matrici B adimensionali discretizzate
    #     time_instant_current : int (1-based, come in MATLAB)
    #     Ns : int, numero di step di predizione
    #     need_grad : bool, se True calcola il gradiente per differenze finite
    #
    #     Ritorna
    #     -------
    #     J : float
    #     grad : np.ndarray shape (3*Ns,), gradiente rispetto a U_vec
    #     """
    #
    #     # TODO: ADDED A WORKAROUND Ns = Ns-1 --> questo dovrebbe essere = Nc
    #     Ns = Ns - 1
    #     # Ns = Ns
    #
    #     # --- setup e pesi (Q, R, Z) come in loss_and_grad ---
    #     Q_diag = np.array(
    #         [self.weight.Q_pos] * 3 + [self.weight.Q_vel] * 3, dtype=float
    #     )
    #     Z_diag = np.array(
    #         [self.weight.Z_pos] * 3 + [self.weight.Z_vel] * 3, dtype=float
    #     )
    #     R_diag = np.array([self.weight.R_value] * 3, dtype=float)
    #
    #     Q = np.diag(Q_diag)
    #     Z = np.diag(Z_diag)
    #     R = np.diag(R_diag)
    #
    #     # --- reference window: come in MATLAB, xref deve coprire Ns righe ---
    #     # TODO: controlla che l'indice sia 0-based o 1-based e correggi di conseguenza
    #     # time_instant_current è 1-based -> indice Python 0-based:
    #     print(f"time_instant_current: {time_instant_current}")
    #     t0 = int(time_instant_current) - 1
    #     print(f"t0: {t0}")
    #     t1 = t0 + Ns
    #     # if t1 > self.x_ref.shape[0]:
    #     #     raise ValueError(
    #     #         f"x_ref non copre l'orizzonte richiesto: serve almeno fino a indice {t1 - 1} (0-based), "
    #     #         f"ma ha shape {self.x_ref.shape}"
    #     #     )
    #     print(f"size: {self.x_ref.shape[0]}")
    #     xref = np.asarray(self.x_ref[t0:t1, :], dtype=float)  # (Ns,6)
    #
    #     x0 = np.asarray(x0, dtype=float).reshape(6, )
    #     U_vec = np.asarray(U_vec, dtype=float)
    #     # if U_vec.size != 3 * Ns:
    #     #     raise ValueError(f"U_vec.size deve essere 3*Ns (= {3 * Ns}), trovato {U_vec.size}")
    #
    #     # MATLAB usa Uvec (3 x Ns); qui uso (Ns x 3) per comodità di indexing:
    #     U = U_vec.reshape(Ns, 3)
    #
    #     # --- funzione obiettivo (solo J), replica 1:1 MATLAB objFun ---
    #     def objective_only(U_flat: np.ndarray) -> float:
    #         UU = U_flat.reshape(Ns, 3)
    #
    #         # Propagazione lineare (adimensionale)
    #         x_sim = np.zeros((Ns, 6), dtype=float)  # pred_traj^T in MATLAB
    #         x_j = x0.copy()
    #
    #         for j in range(Ns):
    #             # Indice matrici in Python: (time_instant_current + j) - 2
    #             # (deriva da MATLAB: (time_instant_current+j)-1 in 1-based)
    #             idx = (time_instant_current + (j + 1)) - 2
    #             A_k = A_vect_AD[:, :, idx]
    #             B_k = B_vect_AD[:, :, idx]
    #             x_new = A_k @ x_j + B_k @ UU[j]
    #             x_sim[j, :] = x_new
    #             x_j = x_new
    #
    #         print(f"xref: {xref.shape}")
    #         print(f"x_sim: {x_sim.shape}")
    #         # Errori (xref - pred_traj(:,1:end)') in MATLAB → (Ns,6) - (Ns,6)
    #         x_err = xref - x_sim  # (Ns,6)
    #
    #         # Jx = sum over j of x_err_j * Q * x_err_j^T
    #         # (x_err @ Q) * x_err, somma su asse=1 poi somma totale
    #         Jx_rows = np.sum((x_err @ Q) * x_err, axis=1)  # (Ns,)
    #         Jx = float(np.sum(Jx_rows))
    #
    #         # Ju = somma su step di u_j * R * u_j^T
    #         Ju_rows = np.sum((UU @ R) * UU, axis=1)  # (Ns,)
    #         Ju = float(np.sum(Ju_rows))
    #
    #         # Terminal cost: x_err_final = pred_traj(:,end)' - xref(end,:)
    #         x_err_final = x_sim[-1, :] - xref[-1, :]
    #         Jfin = float(x_err_final @ Z @ x_err_final)
    #
    #         return Jx + Ju + Jfin
    #
    #     # Valore obiettivo
    #     J0 = objective_only(U_vec)
    #
    #     if not need_grad:
    #         return J0, np.zeros_like(U_vec, dtype=float)
    #
    #     # --- Gradiente per differenze finite (forward) ---
    #     g = np.zeros_like(U_vec, dtype=float)
    #     eps0 = 1e-6
    #     for i in range(U_vec.size):
    #         ui = U_vec[i]
    #         eps = eps0 * max(1.0, abs(ui))
    #         U_pert = U_vec.copy()
    #         U_pert[i] = ui + eps
    #         Ji = objective_only(U_pert)
    #         g[i] = (Ji - J0) / eps
    #
    #     return J0, g

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
        # --- lunghezze coerenti ---
        x0 = np.asarray(x0, dtype=float).reshape(
            6,
        )
        U_vec = np.asarray(U_vec, dtype=float)
        L = U_vec.size // 3  # numero di comandi = numero di transizioni
        if U_vec.size != 3 * L:
            raise ValueError(
                f"U_vec.size deve essere multiplo di 3, trovato {U_vec.size}"
            )
        U = U_vec.reshape(L, 3)

        # x_ref è già finestrato in MPC.run: deve essere (L,6)
        xref = np.asarray(self.x_ref, dtype=float)
        if xref.shape != (L, 6):
            raise ValueError(
                f"x_ref deve avere shape (L,6) = ({L},6); trovato {xref.shape}"
            )

        # pesi
        Q = np.diag([self.weight.Q_pos] * 3 + [self.weight.Q_vel] * 3)
        Z = np.diag([self.weight.Z_pos] * 3 + [self.weight.Z_vel] * 3)
        R = np.diag([self.weight.R_value] * 3)

        # funzione obiettivo
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

            # ora le shape combaciano: (L,6) - (L,6)
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

        # gradiente per differenze finite
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

    # def loss_and_grad_linear(self, x0: np.ndarray, U_vec: np.ndarray, need_grad: bool = True):
    #     L = U_vec.size // 3
    #     U = U_vec.reshape(L, 3)
    #
    #     # Weights
    #     Q_diag = np.array([self.weight.Q_pos] * 3 + [self.weight.Q_vel] * 3, dtype=float)
    #     Z_diag = np.array([self.weight.Z_pos] * 3 + [self.weight.Z_vel] * 3, dtype=float)
    #     R_diag = np.array([self.weight.R_value] * 3, dtype=float)
    #
    #     wpos = np.asarray(self.weight.w_pos_vel, dtype=float).reshape(L)
    #     wctl = np.asarray(self.weight.w_control, dtype=float).reshape(L)
    #
    #     # Dynamics
    #     Phi = self.dyn.Phi  # 6x6
    #     Gam = self.dyn.Gamma # 6x3
    #
    #     def objective_only(U_flat: np.ndarray) -> float:
    #         UU = U_flat.reshape(L, 3)
    #
    #         # forward pass: x1..xL
    #         x = np.zeros((L + 1, 6), dtype=float)
    #         x[0] = x0.astype(float)
    #         for k in range(L):
    #             x[k + 1] = Phi @ x[k] + Gam @ UU[k]
    #
    #         # reference
    #         xref = np.asarray(self.x_ref, dtype=float)
    #         if xref.shape[0] < L or xref.shape[1] != 6:
    #             raise ValueError(f"x_ref shape must be (>=L,6). Got {xref.shape}, L={L}")
    #
    #         err = x[1:] - xref[:L]  # (L,6)
    #         Jx_rows = (err ** 2) @ Q_diag  # (L,)
    #         Ju_rows = (UU ** 2) @ R_diag  # (L,)
    #         Jx = np.sum(wpos * Jx_rows)
    #         Ju = np.sum(wctl * Ju_rows)
    #         Jfin = float(((x[L] - xref[L - 1]) ** 2) @ Z_diag)
    #         return float(Jx + Ju + Jfin)
    #
    #     J0 = objective_only(U_vec)
    #
    #     if not need_grad:
    #         return J0
    #
    #     # Forward difference gradient
    #     g = np.zeros_like(U_vec, dtype=float)
    #
    #     eps0 = 1e-6
    #     for i in range(U_vec.size):
    #         ui = U_vec[i]
    #         eps = eps0 * max(1.0, abs(ui))
    #         U_pert = U_vec.copy()
    #         U_pert[i] = ui + eps
    #         Ji = objective_only(U_pert)
    #         g[i] = (Ji - J0) / eps
    #
    #     return J0, g


# ---------------------------------------------------------------------------
#  IPOPT problem
# ---------------------------------------------------------------------------
# class IpoptProblem:
#     def __init__(
#         self,
#         x0: np.ndarray,
#         U0: np.ndarray,
#         bounds: Bounds,
#         opt: Optimization,
#         scenario: str,
#         current_instant: int,
#         A_vect_AD: np.ndarray,
#         B_vect_AD: np.ndarray,
#         Ns: int,
#     ):
#         self.x0 = x0.astype(float)
#         self.U = U0.astype(float)
#         self.bounds = bounds
#         self.opt = opt
#         self.Ns = Ns
#         self.m = 0  # no constraints
#         self.scenario = scenario
#         self.current_instant = current_instant
#         self.A_vect_AD = A_vect_AD
#         self.B_vect_AD = B_vect_AD
#
#     # Objective
#     def objective(self, u):
#         if self.scenario == "Neural_Network":
#             loss, _ = self.opt.loss_and_grad(self.x0, u, need_grad=False)
#         else:
#             # TODO: CAPISCI SE self.n va bene al posto di N_s? nella seguente funzione
#             loss, _ = self.opt.loss_and_grad_linear(
#                 self.x0,
#                 u,
#                 A_vect_AD=self.A_vect_AD,
#                 B_vect_AD=self.B_vect_AD,
#                 time_instant_current=self.current_instant,
#                 Ns=self.Ns,
#                 need_grad=False,
#             )
#         return float(loss)
#
#     # loss_and_grad_linear(
#     #             self,
#     #             x0: np.ndarray,
#     #             U_vec: np.ndarray,
#     #             A_vect_AD: np.ndarray,
#     #             B_vect_AD: np.ndarray,
#     #             time_instant_current: int,
#     #             Ns: int,
#     #             need_grad: bool = True,
#
#     # TODO: controlla se servono o meno le funzioni del gradiente su facciamo approssimazioni
#     def gradient(self, u):
#         if self.scenario == "Neural_Network":
#             _, g = self.opt.loss_and_grad(self.x0, u, need_grad=True)
#         else:
#             # TODO: CAPISCI SE self.n va bene al posto di N_s? nella seguente funzione
#             _, g = self.opt.loss_and_grad_linear(
#                 self.x0,
#                 u,
#                 A_vect_AD=self.A_vect_AD,
#                 B_vect_AD=self.B_vect_AD,
#                 time_instant_current=self.current_instant,
#                 Ns=self.Ns,
#                 need_grad=True,
#             )
#         return g
#
#     # Constraints (none)
#     def constraints(self, u):
#         return np.zeros((0,), dtype=np.float64)
#
#     def jacobian(self, u):
#         return np.zeros((0,), dtype=np.float64)
#
#     def jacobianstructure(self):
#         return (np.array([], dtype=int), np.array([], dtype=int))
#
#     def hessianstructure(self):
#         return (np.array([], dtype=int), np.array([], dtype=int))
#
#     def hessian(self, u, lagrange, obj_factor):
#         return np.zeros((0,), dtype=np.float64)
#

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
        # JETSON_IP: str,
        # PORT: int,
        # DTYPE: np.dtype,
        # N_STATE: int,
        # N_CTRL: int,
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
        # self.JETSON_IP = JETSON_IP
        # self.PORT = PORT
        # self.DTYPE = DTYPE
        # self.N_STATE = N_STATE
        # self.N_CTRL = N_CTRL

    def state_rotation(self, x: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Ruota uno stato (6,) oppure un controllo (3,) usando la matrice R (3x3).

        - Stato:  x = [x, y, z, vx, vy, vz]
                  => [R @ r, R @ v]
        - Controllo: x = [ux, uy, uz]
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
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     print(f"[PC] Connecting {self.JETSON_IP}:{self.PORT}...")
        #     s.connect((self.JETSON_IP, self.PORT))
        #     print("[PC] Connected!")

        for k in range(1, T_steps_ref + 1):
            # for k in range(0, T_steps_ref + 1):
            if k + self.N_s > T_steps_ref:
                break

            print("Iteration: ", k)

            ###############################

            # # Qua modifichiamo la traiettoria di riferimento dentro l'optimizer
            # x_ref_window = trajectory_adim[k : (k + self.N_s - 1), :]  # (L,6)
            # self.opt.x_ref = x_ref_window
            #
            # U0_vec = U_guess.reshape(-1) # TODO: CONTROLLA CHE LA GUESS SIA IMPLEMENTATA DIRETTAMENTE SULLA JETSON
            #
            # # TODO: controlla che ci sia una coerenza negli istanti di tempo, tipo che l'istante che passiamo alla
            # # funzione sia effettivamente quello che stiamo valutando
            # # TODO: PER RENDERE IL TUTTO PIù VELOCE SI POTREBBE PENSARE DI METTERE LA DEFINIZIONE DEL PROBLEMA FUORI DAL FOR
            #
            # # Rotation of the initial state (only if R_mat not "None")
            # if R_mat is not None:
            #     # x_k: (6,)
            #     # R_mat: (3, 3)
            #     # print(f"x_k: {x_k.shape}")
            #     # print(f"R_mat: {R_mat.shape}")
            #     x_k = self.state_rotation(x_k, R_mat)
            #
            # ################################################################################################
            # # Interaction with JETSON
            # ################################################################################################
            # print(f"\n[PC] Environment state {k}, x_k =", x_k)
            #
            # # 1) mando lo stato alla Jetson
            # send_int(s, k)
            # send_array(s, x_k, self.DTYPE)
            #
            # # 2) ricevo il controllo dalla Jetson
            # U_opt = recv_array(s, self.N_CTRL, self.DTYPE)
            # print("[PC] Control received =", U_opt)
            #
            #
            # ###############################################################################################
            # ###############################################################################################
            # U_seq = U_opt.reshape(L, 3)
            # U_guess = U_seq

            ###############################

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
                # TODO: ADD THE LINEAR PROPAGATION
                # time_instant_index --> indice del istante di tempo corrente all'interno
                # N_prop --> numero di istanti di tempo per cui si deve propagare (NON compreso lo stato iniziale)
                time_instant_index = k
                traj_lin = linear_propagation(
                    x_k,
                    U_seq.T,
                    time_instant_index,
                    self.N_control,
                    A_vect_AD,
                    B_vect_AD,
                )
                # linear_propagation(x_k, U_opt, time_instant_index, N_prop, A_vect_AD, B_vect_AD)

            # Apply first control and propagate
            u0 = U_seq[0]
            # controls_hist.append(u0.copy())    # TOGLI, DOPPIO SALVATAGGIO CONTROLLO

            x_tmp = x_k.copy()

            # Rotation of the state and control (go back to "real" frame)
            if R_mat is not None:
                # x_tmp: (6,)
                # u0: (3,)
                # print(f"x_tmp: {x_tmp.shape}")
                # print(f"u0: {u0.shape}")
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