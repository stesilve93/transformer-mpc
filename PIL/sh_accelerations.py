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
    # mu: torch.Tensor,
    # a_e: torch.Tensor,
    # C: torch.Tensor,
    # S: torch.Tensor,
    # N: int,
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
    r = torch.clamp(r, min=A_E + 1e-3)

    lam = torch.atan2(y, x)
    rho_xy = torch.hypot(x, y)
    phi = torch.atan2(z, rho_xy)
    sphi = torch.sin(phi)
    cphi = torch.cos(phi)

    q = A_E / r  # shape (B,)

    # cos(mλ), sin(mλ) per m=0..N  -> shape (B, N+1)
    mvals = torch.arange(N_global + 1, dtype=dtype, device=device)  # (N+1,)
    lam_mat = lam.unsqueeze(1) * mvals.unsqueeze(0)  # (B,N+1)

    cosml = torch.cos(lam_mat)
    sinml = torch.sin(lam_mat)

    # P_l^m(x_leg) con x_leg = sin φ  -> (B, N+1, N+1)
    x_leg = sphi
    x_leg = torch.clamp(x_leg, -1.0 + 1e-7, 1.0 - 1e-7)
    P = assoc_legendre_all_m_unnorm_batch(x_leg, N_global)

    # Sums (B,)
    ar_sum = torch.zeros(B, dtype=dtype, device=device)
    aphi_sum = torch.zeros(B, dtype=dtype, device=device)
    alam_sum = torch.zeros(B, dtype=dtype, device=device)

    for l in range(0, N_global + 1):
        q_l = q**l  # (B,)
        for m in range(0, l + 1):
            P_lm = P[:, l, m]  # (B,)
            T = C_non_norm[l, m] * cosml[:, m] + S_non_norm[l, m] * sinml[:, m]  # (B,)

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
                        -C_non_norm[l, m] * sinml[:, m] + S_non_norm[l, m] * cosml[:, m]
                    )

    eps = torch.tensor(1e-15, dtype=dtype, device=device)
    a_r = -MU_LR / (r * r) * ar_sum
    a_phi = MU_LR / (r * r) * aphi_sum
    a_lam = MU_LR / (r * r) * (alam_sum / torch.maximum(eps, cphi))

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
    Learnable physical parameters + spherical coefficients:
      - omega, mu, a_e: learnable
      - C, S: matrices up to degree N; only selected elements are learnable, others remain fixed.

    Args:
      - N (int): maximum expansion degree
      - C (torch.Tensor): initial non-normalized C coefficients, shape (N+1, N+1)
      - S (torch.Tensor): initial non-normalized S coefficients, shape (N+1, N+1)
      - omega (float): initial omega value
      - mu (float): initial mu value
      - a_e (float): initial reference radius (m)
      - learnable_keys (list[tuple]): list of entries to be learnable
      - check (bool): if True, performs consistency checks
    """

    def __init__(
        self,
        N: int = 2,
        C: torch.Tensor = C_non_norm,
        S: torch.Tensor = S_non_norm,
        omega: float = Omega_LR,
        mu: float = MU_LR,
        a_e: float = 16000.0,
        learnable_keys=None,
        check: bool = True,
    ):
        super().__init__()
        self.N = int(N)

        C = torch.as_tensor(C, dtype=torch.float32)
        S = torch.as_tensor(S, dtype=torch.float32)

        if C is None or S is None:
            raise ValueError(
                "You must pass the initial C and S tensors (shape (N+1, N+1))."
            )

        if C.shape != (N + 1, N + 1) or S.shape != (N + 1, N + 1):
            raise ValueError(
                f"Expected C/S shapes: {(N + 1, N + 1)}, got: {C.shape} / {S.shape}"
            )

        C_base = C.detach().clone()
        S_base = S.detach().clone()
        C_base[0, 0] = 1.0

        self.register_buffer("C_base", C_base)
        self.register_buffer("S_base", S_base)

        self.omega_raw = nn.Parameter(torch.tensor(omega))
        self.mu_raw = nn.Parameter(torch.tensor(float(mu)))
        self.a_e_raw = nn.Parameter(torch.tensor(float(a_e)))

        # list of learnable keys for C/S
        if learnable_keys is None:
            learnable_keys = [
                (1, 0, "C"),
                (1, 1, "C"),
                (1, 1, "S"),
                (2, 0, "C"),
                (2, 1, "C"),
                (2, 2, "C"),
                (2, 1, "S"),
                (2, 2, "S"),
            ]
        self.learnable_keys = list(learnable_keys)

        # Some checks are performed:
        if check:
            for l, m, which in self.learnable_keys:
                if not (0 <= l <= self.N and 0 <= m <= l):
                    raise ValueError(f"Learnable key out of range: (l={l}, m={m}).")
                if which not in ("C", "S"):
                    raise ValueError(f'"which" must be "C" or "S", got: {which}')
                if l == 0 and m == 0:
                    raise ValueError(
                        "C[0,0] is fixed by definition; do not set it as learnable."
                    )

        # Initialize learnable C/S parameters with the provided initial values
        params = []
        for l, m, which in self.learnable_keys:
            # Take initial values from the provided C and S matrices and make them trainable parameters
            init_val = (self.C_base[l, m] if which == "C" else self.S_base[l, m]).item()
            params.append(nn.Parameter(torch.tensor(float(init_val))))
        self.learnable_params = nn.ParameterList(params)

    @property
    def omega(self):
        return self.omega_raw

    @property
    def mu(self):
        return self.mu_raw

    @property
    def a_e(self):
        return self.a_e_raw

    def get_CS(self):
        """
        Returns the final C, S:
          - starts from C_base/S_base (fixed)
          - overwrites ONLY the indices in learnable_keys with the corresponding learnable parameters
        """
        C = self.C_base.clone()
        S = self.S_base.clone()

        for p, (l, m, which) in zip(self.learnable_params, self.learnable_keys):
            if which == "C":
                C[l, m] = p
            else:
                S[l, m] = p
        return C, S

    def forward(self, pos_batch):
        """
        Computes accelerations using spherical expansion
        """
        C, S = self.get_CS()
        return sh_gravity_accel_cart_phi_torch_batch(
            pos_batch, self.mu, self.a_e, C, S, self.N
        )

