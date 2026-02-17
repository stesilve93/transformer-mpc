from dataclasses import dataclass
import numpy as np
import torch
import cyipopt
from pathlib import Path
from scipy.io import loadmat, savemat
import socket
import struct
import time

from net_model.pinnsformer import PINNsformer, get_positional_encoding

# ---------------------------------------------------------------------------
#  Connection to the environment (Server)
# ---------------------------------------------------------------------------
# HOST = "127.0.0.1"
# PORT = 5000
# DTYPE = np.float64
# N_STATE = 6
# #PRINT_LEVEL = 3

# ---------------------------------------------------------------------------
#  Parameters
# ---------------------------------------------------------------------------
SIMULATION_TIME = 40
NS = 16
SAVE_K_ITERATION = SIMULATION_TIME - NS     # specify the iteration at which the times list is saved
INFERENCE_TIME = False
SOLVE_OPT_TIME = False

PRINT_LEVEL = 0

# ---------------------------------------------------------------------------
#  Variables to save time values
# ---------------------------------------------------------------------------
infer_times_ms = []
solve_times_ms = []

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", DEVICE)

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
        self.L = 15     # specifico per lo scenario che stiamo considerando

        # Weights
        self.wpos = torch.as_tensor(self.weight.w_pos_vel, dtype=self.dtype, device=self.dev).view(
            1, self.L, 1)
        self.wctl = torch.as_tensor(self.weight.w_control, dtype=self.dtype, device=self.dev).view(
            1, self.L, 1)

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
            [self.weight.Q_pos] * 3 + [self.weight.Q_vel] * 3, device=self.dev, dtype=self.dtype
        ).view(1, 1, 6)
        self.z_vec = torch.tensor([self.weight.Z_pos]*3 + [self.weight.Z_vel]*3, device=self.dev, dtype=self.dtype)
        self.r_vec = torch.tensor([self.weight.R_value]*3, device=self.dev, dtype=self.dtype).view(1,1,3)

    def set_x_ref(self):
        self.xref_t = torch.as_tensor(self.x_ref, dtype=self.dtype, device=self.dev).view(1, self.L, 6)



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
        #x0_t = torch.from_numpy(x0).to(self.dev).view(1,6)
        x0_t = torch.as_tensor(x0, dtype=self.dtype, device=self.dev).view(1, 6)
        U_t = torch.as_tensor(U_vec, dtype=self.dtype, device=self.dev)
        U_t.requires_grad_(need_grad)

        xref_t = self.xref_t

        # Prediction
        if not need_grad:
            with torch.inference_mode():
                if INFERENCE_TIME == True:
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
        Ju_rows = (U_seq*U_seq*self.r_vec).sum(dim=2, keepdim=True)
        Ju = (self.wctl * Ju_rows).sum()

        # Control smoothness
        # U_pad = torch.cat([U_seq[:, 0:1, :], U_seq], dim=1)
        # dU = U_pad[:, 1:, :] - U_pad[:, :-1, :]
        # Jdu = self.gamma_du * ((dU @ R) * dU).sum()

        # Terminal
        x_fin = x_err[:, -1, :].view(6)
        Jfin    = (x_fin*x_fin*self.z_vec).sum()
        # J = Jx + Ju + Jdu + Jfin
        J = Jx + Ju + Jfin

        if need_grad:
            J.backward()
            grad = U_t.grad.detach().cpu().numpy().astype(np.float64)
        else:
            grad = np.zeros_like(U_vec, dtype=np.float64)
        return float(J.item()), grad  # Tuple[float, np.ndarray]
        #return float(J.detach().cpu().numpy()), grad  # Tuple[float, np.ndarray]


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
        self._cache_u = None          # np.ndarray
        self._cache_loss = None       # float
        self._cache_grad = None       # np.ndarray
        self._cache_has_grad = False  # bool

    def _u_equal(self, u):
        # IPOPT spesso passa lo stesso array identico; array_equal è ok e veloce
        return self._cache_u is not None and np.array_equal(u, self._cache_u)

    def _compute(self, u, need_grad: bool):
        # Qui calcoli (loss, grad) per quel u
        if self.scenario == "Neural_Network":
            return self.opt.loss_and_grad(self.x0, u, need_grad=need_grad)
        else:
            assert "Error in the selection of the scenario variable!"


    def _ensure_cache(self, u, want_grad: bool):
        """
        Garantisce che in cache ci sia almeno la loss, e se want_grad=True anche il grad.
        """
        if self._u_equal(u):
            # stesso u: se voglio grad e non ce l'ho, lo calcolo una volta
            if want_grad and not self._cache_has_grad:
                loss, grad = self._compute(u, need_grad=True)
                self._cache_loss = float(loss)
                self._cache_grad = np.asarray(grad, dtype=np.float64).copy()
                self._cache_has_grad = True
            return

        # u diverso: reset cache e calcola ciò che serve
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

class Algorithm:
    def __init__(self, units, N_s, N_control, weights, net, bounds, opt, traj_phys, traj_adim, tt_vect, scenario):
        self.units=units
        self.N_s=N_s
        self.N_control=N_control
        self.weights=weights
        self.net=net
        self.bounds=bounds
        self.opt=opt
        self.traj_phys=traj_phys
        self.traj_adim=traj_adim
        self.tt_vect=tt_vect
        self.scenario=scenario

    def initialisation(self):
        max_iter = 2000    # TODO: CHECK THIS
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
        x_k= self.traj_adim[0].copy()

        prob = IpoptProblem(
            x_k,         # x_k
            U_guess,     # U0_vec
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

        #U0_vec = self.U_guess.reshape(-1)
        U0_vec = U_guess.reshape(-1)

        # ---------------------------------------
        # Algorithm
        # ---------------------------------------
        self.prob.x0 = x_k
        self.prob._cache_u = None  # reset cache
        self.prob._cache_loss = None
        self.prob._cache_grad = None
        self.prob._cache_has_grad = False

        #print("x0: ", self.prob.x0)

        if SOLVE_OPT_TIME == True:
            if opt.dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            U_opt, info = self.nlp.solve(U0_vec)

            if opt.dev.type == "cuda":
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




def main():

    scenario = "Neural_Network"

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
    traj_mat = ROOT / "reference_trajectories" / "trajectory_reference.mat"
    tt_mat = ROOT / "reference_trajectories" / "time_reference.mat"

    traj_phys, traj_adim, tt_vect = load_reference(
        traj_mat, tt_mat, units
    )

    # -----------------------------------------------------------------------
    # Options solver
    # -----------------------------------------------------------------------
    max_iter = 2000    # TODO: CHECK THIS
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

    # -----------------------------------------------------------------------
    # Neural network
    # -----------------------------------------------------------------------
    model_path = ROOT / "net_model" / "best_test_model_NET_no_rot_no_relu.pt"
    net = NeuralNetwork(model_path, device=DEVICE)
    net.load_model()

    # -----------------------------------------------------------------------
    # Bounds
    # -----------------------------------------------------------------------
    u_max = 0.6 / units.CU  # adim, per componente
    lower = -u_max * np.ones(3 * N_control)
    upper = +u_max * np.ones(3 * N_control)
    bounds = Bounds(lower=lower, upper=upper)

    # -----------------------------------------------------------------------
    # Optimization
    # -----------------------------------------------------------------------
    opt = Optimization(weight=weights, units=units, x_ref=traj_adim, net=net)

    L = N_control
    U_guess = np.zeros((L, 3), dtype=float)


    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #     s.bind((HOST, PORT))
    #     s.listen(1)
    #     print(f"[JETSON] Listening on {HOST}:{PORT}")
    #
    #     # ---------------------------------------
    #     # Ricezione dei dati dall'environment
    #     # --------------------------------------
    #     conn, addr = s.accept()


    # ---------------------------------------
    # Definition of the IPOPT problem
    # --------------------------------------
    U_guess = np.zeros((L, 3), dtype=float)
    x_k= traj_adim[0].copy()

    prob = IpoptProblem(
        x_k,         # x_k
        U_guess,     # U0_vec
        bounds,
        opt,
        scenario=scenario,
        Ns=N_s,
    )

    nlp = cyipopt.Problem(
        n=(15*3),      # U0_vec.size
        m=prob.m,
        problem_obj=prob,
        lb=bounds.lower,
        ub=bounds.upper,
    )
    # IPOPT options
    nlp.add_option("tol", 1e-6)
    nlp.add_option(
        "max_iter", int(options_solv.get("max_iter", 2000))
    )
    nlp.add_option("hessian_approximation", "limited-memory")
    nlp.add_option(
        "print_level", int(options_solv.get("print_level", 5))
    )

    # with conn:
    #     print(f"[JETSON] Connected with {addr}")
    #     while True:
    #         # try:
    #         # 1) ricevo lo stato dal "satellite"
    #         k = recv_int(conn)
    #         x_k = recv_array(conn, N_STATE)
    #         print("[JETSON] k =", k)
    #         print("[JETSON] Received state x =", x_k)


    # TODO: QUA DEVI AGGIUNGERE L'ENTRATA DELLO STATO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # se cambi tipologia di float devi cambiare anche la riga qua sotto
    x_k = x_k.astype(np.float32, copy=True)
    prob.x0 = x_k

    x_ref_window = traj_adim[k : (k + N_s - 1), :]  # (L,6)
    opt.x_ref = x_ref_window
    opt.set_x_ref()

    U0_vec = U_guess.reshape(-1)

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

    # # ---------------------------------------
    # # End algorithm
    # # ---------------------------------------
    # print("[JETSON] Control u =", U_opt)
    #
    # # 3) mando il controllo indietro
    # send_array(conn, U_seq)
    #
    # if SAVE_K_ITERATION == k:
    #     if INFERENCE_TIME == True:
    #         savemat(
    #             "evaluation_results/times/inference_time.mat",
    #             {"inference_time": infer_times_ms},
    #         )
    #         print(f"Inference Times: {infer_times_ms}")
    #
    #     if SOLVE_OPT_TIME == True:
    #         savemat(
    #             "evaluation_results/times/solve_opt_time.mat",
    #             {"solve_opt_time": solve_times_ms},
    #         )
    #         print(f"Inference Times: {solve_times_ms}")
    # except Exception as e:
    #     print("[JETSON] Error / Connection closed:", e)
    #     break


# if __name__ == "__main__":
#     main()