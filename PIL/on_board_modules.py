import torch
from sh_accelerations import PhysicsParams, sh_gravity_accel_cart_phi_torch_batch
import numpy as np
from typing import Deque, Tuple
import random
from collections import deque
import struct
from dataclasses import dataclass
import time
from scipy.io import loadmat

from net_model.pinnsformer import PINNsformer, get_positional_encoding

# Global parameters ----------------------------------------------
DTYPE = np.float64

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

# ----------------------------------------------------------------

def accel_fd_with_impulses_batch(
    V: torch.Tensor, dV: torch.Tensor, dt: float
) -> torch.Tensor:
    """
    V  : (B, L, 3)  velocities without control
    dV : (B, L, 3)  impulsive Δv applied at node i
    Returns:
    a_fd : (B, L-1, 3)  average acceleration over the intervals [i, i+1] including Δv(i)
    """
    # Compute only over full intervals [i, i+1], i = 0..L-2
    a_fd = (V[:, 1:, :] - (V[:, :-1, :] + dV[:, :-1, :])) / dt
    return a_fd


def physics_residuals_with_impulses_general_check(
    pred_traj_phys: torch.Tensor,
    dV: torch.Tensor,
    phys: PhysicsParams,
    dt: float,
    device,
):
    """
    pred_traj_phys : (B, L, 6)  -> [x,y,z,vx,vy,vz] in physical units (body frame)
    dV             : (B, L, 3)  -> Δv per time instant (if (B, L-1, 3), pad one zero row at the end)
    phys           : PhysicsParams
    dt             : time step [s]
    device         : torch.device

    Returns:
    rx, ry, rz  : (B, L-1) dimensionless residuals per component
    loss_per_t  : (B, L-1) squared-sum per time instant
    loss_mse    : ()       mean over batch and time of the squared-sum
    """
    pred_traj_phys = pred_traj_phys.to(device)
    dV = dV.to(device)

    B, L, _ = pred_traj_phys.shape

    #  if dV is (B, L-1, 3) -> pad one zero row to get (B, L, 3)
    if dV.shape[1] == L - 1:
        pad = torch.zeros(B, 1, 3, dtype=dV.dtype, device=dV.device)
        dV = torch.cat([dV, pad], dim=1)

    pos = pred_traj_phys[..., 0:3]  # (B, L, 3)
    V = pred_traj_phys[..., 3:6]  # (B, L, 3)
    V_node = V + dV  # (B, L, 3) velocity post-Δv

    a_fd = accel_fd_with_impulses_batch(V, dV, dt)

    # Align pos, V_node (B, L-1, ...)
    pos_mid = pos[:, :-1, :]  # (B, L-1, 3)
    Vnode_mid = V_node[:, :-1, :]  # (B, L-1, 3)

    # a_T (gravity) (B, L-1, 3)
    C, S = phys.get_CS()
    Bm, Lm1 = pos_mid.shape[:2]
    aT = sh_gravity_accel_cart_phi_torch_batch(
        pos_mid.reshape(-1, 3), phys.mu, phys.a_e, C, S, phys.N
    ).reshape(Bm, Lm1, 3)  # (B, L-1, 3)

    Om = phys.omega
    vx, vy, vz = Vnode_mid[..., 0], Vnode_mid[..., 1], Vnode_mid[..., 2]
    x, y, z = pos_mid[..., 0], pos_mid[..., 1], pos_mid[..., 2]

    # Position and velocity vectors
    r = torch.stack((x, y, z), dim=-1)
    v = torch.stack((vx, vy, vz), dim=-1)


    Om = phys.omega.to(device=device, dtype=v.dtype)

    # broadcast to (B, L-1, 3)
    omega_vec = Om.view(1, 1, 3).expand_as(v)


    # Rotating-frame terms:
    # Coriolis:      -2 ω × v
    # Centrifugal:   -ω × (ω × r)
    # print("Om tensor: ", Om.shape)

    a_coriolis = -2.0 * torch.cross(omega_vec, v, dim=-1)
    a_centrifugal = -torch.cross(omega_vec,
                             torch.cross(omega_vec, r, dim=-1),
                             dim=-1)

    a_total = aT + a_coriolis + a_centrifugal  # (B, L-1, 3)

    rx_phys = a_fd[..., 0] - a_total[..., 0]
    ry_phys = a_fd[..., 1] - a_total[..., 1]
    rz_phys = a_fd[..., 2] - a_total[..., 2]

    # Absolute value
    # rx = rx_phys.abs()
    # ry = ry_phys.abs()
    # rz = rz_phys.abs()

    rx = rx_phys
    ry = ry_phys
    rz = rz_phys

    loss_per_t = rx**2 + ry**2 + rz**2  # (B, L-1)
    #residuals = rx + ry + rz  # (B, L-1)
    residuals = rx**2 + ry**2 + rz**2
    loss_mse = loss_per_t.mean()  # scalar

    return a_fd, a_total, residuals


# ---------------------------------------------------------------------------
#  Performance computation
# ---------------------------------------------------------------------------
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

    MPE_pos = APE_pos[-moving_window:].mean()

    return MPE_pos


def performance_indices_computation(
    traj_followed_phys,
    traj_phys,
    tt_vect,
    n_sim=None,
    end_win=None,
    moving_window=8,
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
    (MPE_pos) = performance_indices_vectors(
        traj_followed_phys_cmp,
        traj_phys_cmp[:end_win],
        1,
        end_win,
        moving_window=moving_window,
    )

    return MPE_pos


# ---------------------------------------------------------------------------
#  Online replay buffer for on-orbit learning
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
    part2 = arr[: (end % n), :]
    return np.vstack([part1, part2])


class ReplayBuffer:
    """
    Stores recent executed windows to supervise the network.
    Each item is a tuple (x0: np.ndarray(6,), U_seq: np.ndarray(L,3), target_deltas: np.ndarray(L,6)).
    target_deltas[j] = x_{j+1} - x_{j} for the window starting at x0.
    """

    def __init__(self, capacity: int = 1024):
        self.capacity = int(capacity)  # maximum capacity of the buffer
        # FIFO (First in First Out) buffer
        # when is full, eliminates the oldest one
        self.data: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=self.capacity
        )

    def __len__(self):
        return len(self.data)

    def add(self, x0_np, U_np, Yd_np):
        # x0_np: (6,)
        # U_np:  (L,3)
        # Yd_np: (L,6)
        x0_t = torch.from_numpy(x0_np).to(dtype=torch.float32).view(6)
        U_t = torch.from_numpy(U_np).to(dtype=torch.float32).view(-1, 3)
        Yd_t = torch.from_numpy(Yd_np).to(dtype=torch.float32).view(-1, 6)
        self.data.append((x0_t, U_t, Yd_t))

    def sample(self, batch_size: int, device=None):
        batch = random.sample(self.data, k=min(batch_size, len(self.data)))

        # Converting directly to torch tensor
        x0 = torch.stack([b[0] for b in batch], dim=0)  # (B,6)
        U = torch.stack([b[1] for b in batch], dim=0)  # (B,L,3)
        Yd = torch.stack([b[2] for b in batch], dim=0)  # (B,L,6)

        if device is not None:
            x0 = x0.to(device, non_blocking=True)
            U = U.to(device, non_blocking=True)
            Yd = Yd.to(device, non_blocking=True)
        return x0, U, Yd

    def iter_batches(
        self, batch_size: int, max_batches: int | None = None, shuffle: bool = True
    ):
        """Generate non-overlapping batches taken from the current buffer."""

        # Case no element in the buffer
        if len(self.data) == 0:
            return

        # snapshot of the buffer
        data = list(self.data)  # Shuffling the buffer
        if shuffle:
            # Shuffling the buffer
            random.shuffle(data)

        batches_done = 0
        batches = []

        for i in range(0, len(data), batch_size):
            if max_batches is not None and batches_done >= max_batches:
                break
            block = data[i : i + batch_size]

            x0 = np.stack([c[0] for c in block], axis=0)  # (B,6)
            U = np.stack([c[1] for c in block], axis=0)  # (B,L,3)
            Yd = np.stack([c[2] for c in block], axis=0)  # (B,L,6)
            batches.append((x0, U, Yd))
            batches_done += 1

        return batches


# ---------------------------------------------------------------------------
#  Connection functions
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
            raise ConnectionError("connessione chiusa dal client")
        data += chunk
    return data


def recv_array(sock, dim_expected):
    header = recv_all(sock, 8)
    (n_bytes,) = struct.unpack("!Q", header)
    payload = recv_all(sock, n_bytes)
    arr = np.frombuffer(payload, dtype=DTYPE)
    if arr.size != dim_expected:
        raise ValueError(f"attesi {dim_expected} elementi, ricevuti {arr.size}")
    return arr


def send_array(sock, arr):
    arr = np.asarray(arr, dtype=DTYPE)
    payload = arr.tobytes()
    n_bytes = len(payload)
    sock.sendall(struct.pack("!Q", n_bytes))
    sock.sendall(payload)


# ---------------------------------------------------------------------------
#  Parameters classes
# ---------------------------------------------------------------------------
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
#  Neural network
# ---------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self, model_path: str, phys: PhysicsParams, units: Units, device=None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.phys = phys
        self.units = units

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
        reference_model = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
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
        U = torch.as_tensor(U_vec, dtype=torch.float32, device=self.device).view(
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

    # ------------ ADDITION ONLINE LEARNING METHODS ------------------------------
    def freeze_all(self):
        assert self.model is not None, "Call load_model() first"
        for p in self.model.parameters():
            p.requires_grad = False

    def set_trainable_head(self):
        """
        Freeze everything and unfreeze only the lightweight head.
        Sblocchiamo i parametri solo dell'ultimo layer della rete
        """
        self.freeze_all()
        # Always unfreeze final projection MLP
        for p in self.model.linear_out.parameters():
            p.requires_grad = True

    # ----------------------------------------------------------------
    # TO TRAIN ALL THE NET AND NOT ONLY THE LAST LAYER UNCOMMENT THE FOLLOWING CODE:

    # def set_trainable_head(self):
    #     """
    #     Freeze everything and unfreeze only the lightweight head.
    #     Sblocchiamo i parametri solo dell'ultimo layer della rete
    #     """
    #     #self.freeze_all()
    #     self.train_all()
    #     # Always unfreeze final projection MLP
    #     # for p in self.model.linear_out.parameters():
    #     #     p.requires_grad = True
    #
    # # def set_trainable_head(self):
    # #     """
    # #     Freeze everything and unfreeze only the lightweight head.
    # #     Sblocchiamo i parametri solo dell'ultimo layer della rete
    # #     """
    # #     self.freeze_all()
    # #
    # #     # 2) unfreeze i moduli desiderati
    # #     for p in self.model.embed_input.parameters():
    # #         p.requires_grad = True
    # #
    # #     for p in self.model.linear_out.parameters():
    # #         p.requires_grad = True
    #
    # def train_all(self):
    #     assert self.model is not None, "Call load_model() first"
    #     # Ora blocchiamo il gradiente ovunque
    #     for p in self.model.parameters():
    #         p.requires_grad = True

    # -------------------------------------------------------------------------

    def make_optimizer(self, lr: float = 1e-4, weight_decay: float = 0.0):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError(
                "No trainable parameters set. Call set_trainable_head()."
            )
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))
        self.criterion = torch.nn.SmoothL1Loss()
        return self.optimizer

    def make_optimizer_2(
        self,
        lr: float = 1e-4,
        lr_omega=1e-4,
        lr_mu=1e-4,
        lr_tilt=1e-4,
        lr_azimuth=1e-4,
        weight_decay: float = 0.0,
    ):

        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError(
                "No trainable parameters set. Call set_trainable_head()."
            )

        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Physical optimizer -----------------------------------------------------------------
        self.optimizer_physics = torch.optim.AdamW(
            [
                {"params": [self.phys.omega_raw], "lr": lr_omega, "weight_decay": 0.0},
                {"params": [self.phys.mu_raw],    "lr": lr_mu,    "weight_decay": 0.0},
                {"params": [self.phys.tilt_raw],  "lr": lr_tilt,  "weight_decay": 0.0},
                {"params": [self.phys.azim_raw],  "lr": lr_azimuth,"weight_decay": 0.0},
            ]
        )
        # ------------------------------------------------------------------------------------


        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))
        self.criterion = torch.nn.SmoothL1Loss()
        return self.optimizer, self.optimizer_physics

    def train_step(
        self, x0_b, U_b, target_deltas_b, fast_but_rep=True, grad_clip: float = 1.0
    ):
        """One single optimization step on a single mini-batch.
        The inputs are directly tensors"""
        self.model.train()

        if not fast_but_rep:
            dev = self.device
            x0_t = torch.as_tensor(x0_b, dtype=torch.float32, device=dev)
            U_t = torch.as_tensor(U_b, dtype=torch.float32, device=dev)
            y_t = torch.as_tensor(target_deltas_b, dtype=torch.float32, device=dev)
        else:
            x0_t = x0_b
            U_t = U_b
            y_t = target_deltas_b

        self.optimizer.zero_grad(set_to_none=True)

        # autocast
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            pred_deltas = self.model(x0_t, U_t)  # (B,L,6)
            loss = self.criterion(pred_deltas, y_t)

            pred_traj = torch.cumsum(
                torch.cat([x0_t.unsqueeze(1), pred_deltas], dim=1), dim=1
            )[:, 1:, :]

            pred_pos_phys = pred_traj[..., 0:3].float() * self.units.LU
            pred_vel_phys = pred_traj[..., 3:6].float() * self.units.VU
            dV_phys = U_t * self.units.CU
            pred_traj_phys = torch.cat([pred_pos_phys, pred_vel_phys], dim=-1)

            dt = 300

            _, _, residuals_phys = physics_residuals_with_impulses_general_check(
                pred_traj_phys, dV_phys, self.phys, dt, device=self.device
            )
            residuals_phys = residuals_phys.detach().cpu().numpy()

            residuals_phys_mean = residuals_phys.mean()
            print("\n\n Residuals: residuals_phys: ", residuals_phys_mean)

            residuals_phys_vect.append(residuals_phys_mean)
            print("\n\n Residuals size: ", np.shape(residuals_phys_vect))

            # -------------------------------------------------------------------------------------

        self.scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], grad_clip
            )

        self.scaler.step(self.optimizer)

        self.scaler.update()

        self.model.eval()
        return float(loss.detach().cpu().numpy())

    def train_step_2(self, x0_b, U_b, target_deltas_b, fast_but_rep=True, grad_clip: float = 1.0):
        """One single optimization step on a single mini-batch.
        The inputs are directly tensors """

        self.model.train()

        if not fast_but_rep:
            dev = self.device
            x0_t = torch.as_tensor(x0_b, dtype=torch.float32, device=dev)
            U_t = torch.as_tensor(U_b, dtype=torch.float32, device=dev)
            y_t = torch.as_tensor(target_deltas_b, dtype=torch.float32, device=dev)
        else:
            x0_t = x0_b
            U_t = U_b
            y_t = target_deltas_b

        self.optimizer.zero_grad(set_to_none=True)

        # autocast
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            pred_deltas = self.model(x0_t, U_t)  # (B,L,6)
            loss = self.criterion(pred_deltas, y_t)

            pred_traj = torch.cumsum(
                torch.cat([x0_t.unsqueeze(1), pred_deltas], dim=1), dim=1
            )[:, 1:, :]

            pred_pos_phys = pred_traj[..., 0:3].float() * self.units.LU
            pred_vel_phys = pred_traj[..., 3:6].float() * self.units.VU
            dV_phys = U_t * self.units.CU
            pred_traj_phys = torch.cat([pred_pos_phys, pred_vel_phys], dim=-1)

            dt = 300

            print("Train step \n")
            print(f"Omega: {self.phys.omega_raw} | mu: {self.phys.mu_raw} | tilt: {self.phys.tilt_angle_deg} | azimuth: {self.phys.azimuth_angle_deg} \n")

            _, _, residuals_phys = physics_residuals_with_impulses_general_check(
                pred_traj_phys, dV_phys, self.phys, dt, device=self.device
            )

            print(" Residuals shape: ", residuals_phys.shape)

            residuals_phys_mean = residuals_phys.mean()
            print("\n\n Residuals: residuals_phys: ", residuals_phys_mean)

            residuals_phys_vect.append(residuals_phys_mean.item())
            print("\n\n Residuals size: ", np.shape(residuals_phys_vect))

            ########### DEBUGGING #################################
            physics_loss = residuals_phys_mean

            # Taking trainable parameters
            params = [p for p in self.model.parameters() if p.requires_grad]

            print("\n[GRAD CHECK]")
            print("  loss.requires_grad         :", bool(loss.requires_grad))
            print("  physics_loss.requires_grad :", bool(physics_loss.requires_grad))
            print("  #trainable params          :", len(params))

            # Function for the gradient norm
            def _grad_norm(grads):
                s = 0.0
                for g in grads:
                    if g is not None:
                        s += g.detach().pow(2).sum().item()
                return s**0.5

            if not physics_loss.requires_grad or len(params) == 0:
                print("  -> Physics loss NOT connected to autograd graph (or no trainable params).")
            else:
                g_data = torch.autograd.grad(
                    loss, params, retain_graph=True, allow_unused=True
                )
                g_phys = torch.autograd.grad(
                    physics_loss, params, retain_graph=True, allow_unused=True
                )

                nd = _grad_norm(g_data)
                np_ = _grad_norm(g_phys)
                ratio = np_ / (nd + 1e-12)

                print(f"  ||grad data|| = {nd:.3e}")
                print(f"  ||grad phys|| = {np_:.3e}")
                print(f"  ratio phys/data = {ratio:.3e}")
            #######################################################

            #-------------------------------------------------------------------------------------

        print(f"\nData loss: {loss} | Physics loss: {residuals_phys_mean}")
        alpha = 1
        beta = 3
        print(f" \nalpha * Data loss: {((alpha * loss))} | beta * Physics loss: {beta * residuals_phys_mean}")
        loss_total = ((alpha * loss)) + beta * residuals_phys_mean
        print(f"\nTotal loss: {loss_total}")

        self.scaler.scale(loss_total).backward()


        if grad_clip is not None and grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], grad_clip
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.model.eval()
        return float(loss_total.detach().cpu().numpy())

    def train_step_physics(
        self,
        x0_b: torch.Tensor,
        U_b: torch.Tensor,
        target_deltas_b: torch.Tensor,
        dt: float = 300.0,
        grad_clip: float = 1.0,
    ):
        """
        Update SOLO parametri fisici (self.phys) usando SOLO dati dal buffer:
          - x0_b: (B,6)
          - U_b:  (B,L,3)
          - target_deltas_b: (B,L,6)  (delta step, dal buffer)
        NON usa predizioni della rete.
        """

        assert hasattr(self, "optimizer_physics"), (
            "Call make_optimizer() first (it must create optimizer_physics)."
        )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.phys.train()
        for p in self.phys.parameters():
            p.requires_grad_(True)

        x0_t = x0_b
        U_t = U_b
        Yd_t = target_deltas_b

        self.optimizer_physics.zero_grad(set_to_none=True)

        # --- reconstruction of the trajectory from the buffer (adimensional) ---
        # x1..xL = x0 + cumsum(delta)
        traj_adim = x0_t.unsqueeze(1) + torch.cumsum(Yd_t, dim=1)  # (B,L,6)

        # --- conversion in physical units ---
        pos_phys = traj_adim[..., 0:3].float() * self.units.LU
        vel_phys = traj_adim[..., 3:6].float() * self.units.VU
        traj_phys = torch.cat([pos_phys, vel_phys], dim=-1)  # (B,L,6)

        dV_phys = U_t * self.units.CU  # (B,L,3)

        # --- phisics residual ---
        _, _, residuals = physics_residuals_with_impulses_general_check(
            traj_phys, dV_phys, self.phys, dt, device=self.device
        )

        loss_phys = residuals.mean()
        loss_phys.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.phys.parameters(), grad_clip)

        self.optimizer_physics.step()

        # metriche utili
        metrics = {
            "loss_phys": float(loss_phys.detach().cpu()),
            "omega": float(self.phys.omega_raw.detach().cpu()),
            "mu": float(self.phys.mu.detach().cpu()),
            "tilt_deg": float(self.phys.tilt_angle_deg.detach().cpu()),
            "az_deg": float(self.phys.azimuth_angle_deg.detach().cpu()),
        }
        return metrics

    def save(self, path: str):
        assert self.model is not None, "No model to save"
        ckpt = {"model": self.model.state_dict()}
        torch.save(ckpt, path)


class Optimization:
    def __init__(
        self,
        weight: Weights,
        units: Units,
        x_ref: np.ndarray,
        net: NeuralNetwork,
        # gamma_du: float = 0.0,  # --> not used currently
        INFERENCE_TIME: bool,
    ):
        self.weight = weight
        self.units = units
        self.x_ref = x_ref
        self.net = net
        self.INFERENCE_TIME = INFERENCE_TIME
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

        # Tensors
        x0_t = torch.from_numpy(x0).to(self.dev).view(1, 6)
        U_t = torch.as_tensor(U_vec, dtype=self.dtype, device=self.dev)
        U_t.requires_grad_(need_grad)

        xref_t = self.xref_t

        # Prediction
        if not need_grad:
            with torch.inference_mode():
                if self.INFERENCE_TIME == True:
                    if self.dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    pred = self.net.predict_for_grad(x0_t, U_t)  # (1,L,6) x1...xL
                    if self.dev.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    infer_times_ms.append(1e3 * (t1 - t0))
                else:
                    pred = self.net.predict_for_grad(x0_t, U_t)  # (1,L,6) x1...xL
        else:
            pred = self.net.predict_for_grad(x0_t, U_t)  # (1,L,6) x1...xL
        x_err = xref_t - pred

        # Being the matrices diagonal we can write the matrices-vector in this way
        # Tracking
        Jx_rows = (x_err * x_err * self.q_vec).sum(dim=2, keepdim=True)  # (1,L,1)
        Jx = (self.wpos * Jx_rows).sum()

        # Control effort
        U_seq = U_t.view(1, self.L, 3)
        Ju_rows = (U_seq * U_seq * self.r_vec).sum(dim=2, keepdim=True)
        Ju = (self.wctl * Ju_rows).sum()

        # Control smoothness
        # U_pad = torch.cat([U_seq[:, 0:1, :], U_seq], dim=1)
        # dU = U_pad[:, 1:, :] - U_pad[:, :-1, :]
        # Jdu = self.gamma_du * ((dU @ R) * dU).sum()

        # Terminal
        x_fin = x_err[:, -1, :].view(6)
        Jfin = (x_fin * x_fin * self.z_vec).sum()
        # J = Jx + Ju + Jdu + Jfin
        J = Jx + Ju + Jfin

        if need_grad:
            J.backward()
            grad = U_t.grad.detach().cpu().numpy().astype(np.float64)
        else:
            grad = np.zeros_like(U_vec, dtype=np.float64)
        return float(J.item()), grad  # Tuple[float, np.ndarray]


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
            if want_grad and not self._cache_has_grad:
                loss, grad = self._compute(u, need_grad=True)
                self._cache_loss = float(loss)
                self._cache_grad = np.asarray(grad, dtype=np.float64).copy()
                self._cache_has_grad = True
            return

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

# ----------------------------------------
# Utilities
# ----------------------------------------
def load_reference(traj_mat_path: str, tt_mat_path: str, units: Units):
    m_traj = loadmat(traj_mat_path)
    m_tt = loadmat(tt_mat_path)
    trajectory_phys = np.array(m_traj.get("xx_final"), dtype=float)
    tt_vect = np.array(m_tt.get("tt")).reshape(-1)
    # adimensionalise
    traj_adim = units.traj_to_adim(trajectory_phys)
    return trajectory_phys, traj_adim, tt_vect