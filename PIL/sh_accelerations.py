import torch
import numpy as np
import math

import torch.nn as nn

##### PARAMETERS ###############################
rho_true = 2670  # kg/m^3
G = 6.67430e-11  # m^3 kg^-1 s^-2
T_eros = 5.27 * 3600  # s
Omega_true = 2 * np.pi / T_eros

# LR: LESS REFINED
rho_LR = (rho_true * 0.3) + rho_true  # error 30%
Omega_LR = (Omega_true * 0.01) + Omega_true  # error 1%

MU_LR = 559361.6014093049

A_E = 16_000

N_global = 2

Cbar = np.zeros((N_global + 1, N_global + 1), dtype=float)
Sbar = np.zeros((N_global + 1, N_global + 1), dtype=float)

# Stokes Coefficients:
Cbar[0, 0] = 1.000000000000e00
Sbar[0, 0] = 0.0
Cbar[1, 0] = 1.175785831520e-03
Sbar[1, 0] = 0.0
Cbar[1, 1] = -3.484427594460e-04
Sbar[1, 1] = 8.766452698130e-05
Cbar[2, 0] = -5.285148878740e-02
Sbar[2, 0] = 0.0
Cbar[2, 1] = 1.021293512930e-04
Sbar[2, 1] = 8.314827416250e-02
Cbar[2, 2] = 1.171641181310e-05
Sbar[2, 2] = -2.819769459150e-02


def fullynorm_to_unnorm(Cbar: np.ndarray, Sbar: np.ndarray):
    """
    Converts fully-normalized coefficients into non-normalized coefficients.
    """
    assert Cbar.shape == Sbar.shape
    Nmax = Cbar.shape[0] - 1
    C = np.zeros_like(Cbar, dtype=float)
    S = np.zeros_like(Sbar, dtype=float)

    for l in range(Nmax + 1):
        for m in range(l + 1):
            if m == 0:
                Nlm = math.sqrt(
                    (2 * l + 1) * math.factorial(l - m) / math.factorial(l + m)
                )
            else:
                Nlm = math.sqrt(
                    2.0 * (2 * l + 1) * math.factorial(l - m) / math.factorial(l + m)
                )
            C[l, m] = Nlm * Cbar[l, m]
            S[l, m] = Nlm * Sbar[l, m]
    return C, S


# ----- Converting into non-normalized coefficients ------------

C_non_norm, S_non_norm = fullynorm_to_unnorm(Cbar, Sbar)

# ---------------------------------------------------------------

##############################################
##############################################


def assoc_legendre_all_m_unnorm_batch(x: torch.Tensor, N: int):
    """
    x: (B,) in [-1,1]
    Returns P: (B, N+1, N+1) with P[:, l, m] defined only for m <= l (0 elsewhere).
    """
    x = x.view(-1)
    B = x.shape[0]
    device, dtype = x.device, x.dtype

    # plm[l][m] -> tensor (B,)
    plm = [[None for _ in range(N + 1)] for _ in range(N + 1)]
    plm[0][0] = torch.ones(B, dtype=dtype, device=device)
    if N >= 1:
        plm[1][0] = x

    # diagonal m=m and subdiagonal
    fact = torch.ones(B, dtype=dtype, device=device)  # (2m-1)!!
    sign = torch.ones(B, dtype=dtype, device=device)  # (-1)^m

    for m in range(1, N + 1):
        fact = fact * (2 * m - 1)
        sign = -sign
        Pmm = sign * fact * (1 - x * x).pow(0.5 * m)
        plm[m][m] = Pmm
        if m + 1 <= N:
            plm[m + 1][m] = (2 * m + 1) * x * Pmm
        for l in range(m + 2, N + 1):
            plm[l][m] = (
                (2 * l - 1) * x * plm[l - 1][m] - (l + m - 1) * plm[l - 2][m]
            ) / (l - m)

    # column m=0 (special recurrence)
    if N >= 1:
        plm[1][0] = x
    for l in range(2, N + 1):
        plm[l][0] = ((2 * l - 1) * x * plm[l - 1][0] - (l - 1) * plm[l - 2][0]) / l

    P = torch.zeros(B, N + 1, N + 1, dtype=dtype, device=device)
    for l in range(N + 1):
        for m in range(l + 1):
            P[:, l, m] = plm[l][m]
    return P


def sh_gravity_accel_cart_phi_torch_batch(
    pos: torch.Tensor,
    mu: torch.Tensor,
    a_e: torch.Tensor,
    C: torch.Tensor,
    S: torch.Tensor,
    N: int,
):
    """
      pos : (B,3) [m] in the body frame
      mu  : ()     scalar torch
      a_e : ()     scalar torch (reference radius)
      C,S : (N+1,N+1) non-normalized coefficients (torch)
      N   : maximum degree spherical harmonics

    Returns:
      a_xyz : (B,3) [m/s^2]

    Note: THE ACCELERATION RETURNED HERE IS IN PHYSICAL UNITS
    """
    assert pos.ndim == 2 and pos.shape[1] == 3, "pos must have shape (B,3)"
    device = pos.device
    dtype = pos.dtype
    B = pos.shape[0]

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    r = torch.sqrt(x * x + y * y + z * z)
    r = torch.clamp(r, min=a_e + 1e-3)

    lam = torch.atan2(y, x)
    rho_xy = torch.hypot(x, y)
    phi = torch.atan2(z, rho_xy)
    sphi = torch.sin(phi)
    cphi = torch.cos(phi)

    q = a_e / r  # shape (B,)

    # cos(mλ), sin(mλ) per m=0..N  -> shape (B, N+1)
    mvals = torch.arange(N + 1, dtype=dtype, device=device)  # (N+1,)
    lam_mat = lam.unsqueeze(1) * mvals.unsqueeze(0)  # (B,N+1)

    cosml = torch.cos(lam_mat)
    sinml = torch.sin(lam_mat)

    # P_l^m(x_leg) con x_leg = sin φ  -> (B, N+1, N+1)
    x_leg = sphi
    x_leg = torch.clamp(x_leg, -1.0 + 1e-7, 1.0 - 1e-7)
    P = assoc_legendre_all_m_unnorm_batch(x_leg, N)

    # Sums (B,)
    ar_sum = torch.zeros(B, dtype=dtype, device=device)
    aphi_sum = torch.zeros(B, dtype=dtype, device=device)
    alam_sum = torch.zeros(B, dtype=dtype, device=device)

    for l in range(0, N + 1):
        q_l = q**l  # (B,)
        for m in range(0, l + 1):
            P_lm = P[:, l, m]  # (B,)
            T = C[l, m] * cosml[:, m] + S[l, m] * sinml[:, m]  # (B,)

            # radial
            ar_sum = ar_sum + (l + 1) * q_l * P_lm * T

            if l >= 1:
                if m <= l - 1:
                    P_lm_prev = P[:, l - 1, m]
                else:
                    P_lm_prev = torch.zeros(B, dtype=dtype, device=device)

                eps = torch.tensor(1e-6, dtype=dtype, device=device)
                den_cphi = torch.where(
                    cphi >= 0,
                    torch.maximum(cphi, eps),  # if cphi>=0, use max(cphi, +eps)
                    torch.minimum(cphi, -eps),  # if cphi<0,  use min(cphi, -eps)
                )

                num = l * x_leg * P_lm - (l + m) * P_lm_prev  # numerator

                dPdphi = num / den_cphi

                aphi_sum = aphi_sum + q_l * dPdphi * T

                if m >= 1:
                    alam_sum = alam_sum + q_l * m * P_lm * (
                        -C[l, m] * sinml[:, m] + S[l, m] * cosml[:, m]
                    )

    eps = torch.tensor(1e-15, dtype=dtype, device=device)
    a_r = -mu / (r * r) * ar_sum
    a_phi = mu / (r * r) * aphi_sum
    a_lam = mu / (r * r) * (alam_sum / torch.maximum(eps, cphi))

    cos_lam = torch.cos(lam)
    sin_lam = torch.sin(lam)

    u_r = torch.stack([cphi * cos_lam, cphi * sin_lam, sphi], dim=1)  # (B,3)
    u_phi = torch.stack([-sphi * cos_lam, -sphi * sin_lam, cphi], dim=1)  # (B,3)
    u_lam = torch.stack([-sin_lam, cos_lam, torch.zeros_like(lam)], dim=1)  # (B,3)

    a_xyz = (
        a_r.unsqueeze(1) * u_r + a_phi.unsqueeze(1) * u_phi + a_lam.unsqueeze(1) * u_lam
    )  # (B,3)
    return a_xyz

class PhysicsParams(nn.Module):
    """
    Learnable:
      - omega
      - mu
      - tilt_angle_deg in [0, 90]
      - azimuth_angle_deg in [0, 360]

    Fixed (buffers):
      - a_e
      - C_base, S_base (spherical harmonic coefficients up to degree N, non-learnable)

    Conventions:
      - tilt: angle from +Z (0 = aligned with +Z), degrees
      - azimuth: angle in XY plane from +X toward +Y, degrees
    """

    def __init__(
        self,
        N: int = 2,
        C: torch.Tensor = C_non_norm,
        S: torch.Tensor = S_non_norm,
        omega: float = 0.0,
        mu: float = 1.0,
        a_e: float = 16000.0,
        tilt_angle_deg: float = 2.0,
        azimuth_angle_deg: float = 15.0,
        check: bool = True,
    ):
        super().__init__()
        self.N = int(N)

        # ---- Fixed C/S (buffers) ----
        if C is None or S is None:
            raise ValueError("You must pass initial C and S tensors with shape (N+1, N+1).")

        C = torch.as_tensor(C, dtype=torch.float32)
        S = torch.as_tensor(S, dtype=torch.float32)

        if check:
            if C.shape != (N + 1, N + 1) or S.shape != (N + 1, N + 1):
                raise ValueError(
                    f"Expected C/S shapes {(N+1, N+1)}, got {tuple(C.shape)} / {tuple(S.shape)}"
                )

        C_base = C.detach().clone()
        S_base = S.detach().clone()
        C_base[0, 0] = 1.0  # fixed by definition

        self.register_buffer("C_base", C_base)
        self.register_buffer("S_base", S_base)

        # ---- Fixed a_e (buffer) ----
        self.register_buffer("a_e", torch.tensor(float(a_e), dtype=torch.float32))

        # ---- Learnable scalars ----
        self.omega_raw = nn.Parameter(torch.tensor(float(omega), dtype=torch.float32))
        self.mu_raw    = nn.Parameter(torch.tensor(float(mu), dtype=torch.float32))

        # ---- Learnable angles (stored raw, mapped to ranges) ----
        self.tilt_raw = nn.Parameter(torch.tensor(
            float(self._inv_sigmoid(tilt_angle_deg / 90.0)), dtype=torch.float32
        ))
        self.azim_raw = nn.Parameter(torch.tensor(
            float(self._inv_sigmoid(azimuth_angle_deg / 360.0)), dtype=torch.float32
        ))

    # --------------------
    # Utils
    # --------------------
    @staticmethod
    def _inv_sigmoid(x, eps=1e-6):
        """
        Inverse sigmoid, safe for x in (0,1).
        Used to initialize raw parameters from constrained physical values.
        """
        x = torch.tensor(float(x))
        x = torch.clamp(x, eps, 1.0 - eps)
        return torch.log(x / (1.0 - x))

    # --------------------
    # Learnable parameters
    # --------------------
    @property
    def omega(self):
        return self.omega_vector()

    @property
    def mu(self):
        return self.mu_raw

    @property
    def tilt_angle_deg(self):
        """Tilt in degrees, constrained to [0, 90]."""
        return 90.0 * torch.sigmoid(self.tilt_raw)

    @property
    def azimuth_angle_deg(self):
        """Azimuth in degrees, constrained to [0, 360]."""
        return 360.0 * torch.sigmoid(self.azim_raw)

    # --------------------
    # Fixed C/S interface
    # --------------------
    def get_CS(self):
        """Return fixed (non-learnable) coefficients."""
        return self.C_base, self.S_base

    # --------------------
    # Spin axis helpers
    # --------------------
    def spin_axis_unit(self):
        """Unit spin axis vector using tilt/azimuth (degrees)."""
        tilt_rad = torch.deg2rad(self.tilt_angle_deg)
        az_rad   = torch.deg2rad(self.azimuth_angle_deg)

        sx = torch.sin(tilt_rad) * torch.cos(az_rad)
        sy = torch.sin(tilt_rad) * torch.sin(az_rad)
        sz = torch.cos(tilt_rad)
        return torch.stack([sx, sy, sz], dim=0)

    def omega_vector(self):
        """Angular velocity vector Ω = omega * u_hat."""
        return self.omega_raw * self.spin_axis_unit()

