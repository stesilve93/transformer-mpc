import numpy as np

# ==================== CONSTANTS ====================

rho_true    = 2670.0           # [kg/m^3]
T_eros      = 5.27 * 3600.0    # [s]
Omega_true  = 2 * np.pi / T_eros

# Nominal Model
# used as reference, on this are introduced variations
rho_reference      = rho_true * 1.10
omega_reference    = Omega_true * 1.01

path_scenarios = "scenarios_parameters.npz"

# ==================== Utilities functions ====================

def sample_single_factor_extended():
    """
    Single-factor scenarios + additional ones for tilt with fixed azimuth values.

    Returns a LIST of dictionaries, each containing:
      - rho_scale
      - omega_scale
      - tilt_deg
      - az_deg
      - spin_axis (3,)
    """
    scenarios = []

    def make(rho_s=1.0, omega_s=1.0, tilt_deg=0.0, az_deg=0.0):
        spin_axis = np.array([
            np.sin(np.deg2rad(tilt_deg)) * np.cos(np.deg2rad(az_deg)),
            np.sin(np.deg2rad(tilt_deg)) * np.sin(np.deg2rad(az_deg)),
            np.cos(np.deg2rad(tilt_deg))
        ])

        return dict(
            rho_scale=float(rho_s),
            omega_scale=float(omega_s),
            tilt_deg=float(tilt_deg),
            az_deg=float(az_deg),
            spin_axis=spin_axis.astype(float),
        )

    # ---- Single-factor on density +/- 30% ----
    for s in [0.70, 0.80, 0.90, 1.10, 1.20, 1.30]:
        scenarios.append(make(rho_s=s))

    # ---- Single-factor on period of rotation +/- 5% ----
    for s in [0.9875, 0.97, 0.95, 1.0125, 1.03, 1.05]:
        scenarios.append(make(omega_s=s))

    # ---- Extra tilt: tilt up to 6 deg, azimuth=[0.0, 90.0, 180.0, 270.0, 360.0] ----
    for tilt in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        for az in [0.0, 90.0, 180.0, 270.0, 360.0]:
            scenarios.append(make(tilt_deg=tilt, az_deg=az))

    # print(f"[DEBUG] scenari single-factor: {len(scenarios)}")
    return scenarios


# LATIN HYPERCUBE SAMPLING:
def sample_lhs_4D(N: int, rng=None):
    """
    LHS on 4 continuous parameters:
      ρ_scale ∈ [0.8, 1.2],
      Omega_scale ∈ [...],
      tilt_deg ∈ [...] deg,

    """
    if rng is None:
        rng = np.random.default_rng()

    def stratified_samples(lo, hi, N):
        edges = np.linspace(lo, hi, N + 1)
        vals = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(vals)
        return vals

    # LHS: we split the considered range into N sub-intervals and take one point in each of these sub-intervals.
    # The generated values are then shuffled.

    rho_vals  = stratified_samples(0.70,  1.30,  N)
    omega_vals = stratified_samples(0.95, 1.05, N)
    tilt_vals = stratified_samples(0.0,  6.0, N)
    az_vals = stratified_samples(0.0,  360, N)

    scenarios = []
    for i in range(N):
        spin_axis = np.array([
            np.sin(np.deg2rad(tilt_vals[i])) * np.cos(np.deg2rad(az_vals[i])),
            np.sin(np.deg2rad(tilt_vals[i])) * np.sin(np.deg2rad(az_vals[i])),
            np.cos(np.deg2rad(tilt_vals[i])),
        ])

        scenarios.append(dict(
            rho_scale=float(rho_vals[i]),
            omega_scale=float(omega_vals[i]),
            tilt_deg=float(tilt_vals[i]),
            az_deg=float(az_vals[i]),
            spin_axis=spin_axis.astype(float),   # from tilt and az angles
        ))

    return scenarios


# ==================== Dictionary construction ====================

def build_scenario_dict(n_lhs: int = 1800, add_absolute_values: bool = True, rng=None):
    """
    Builds a DICTIONARY of scenarios:
      key     = scenario_id (e.g., "SF_001", "LHS_037", ...)
      value   = dictionary containing the scenario parameters

    If add_absolute_values=True, it also adds:
      - rho_abs       (scenario density)
      - omega_abs     (scenario angular velocity)
      - omega_scale
    using the global rho_reference and Omega_reference as references.
    """
    if rng is None:
        rng = np.random.default_rng()

    single_factor_scenarios = sample_single_factor_extended()
    lhs_scenarios           = sample_lhs_4D(n_lhs, rng=rng)

    scenarios_dict = {}

    # --- Single-factor ---
    for k, sc in enumerate(single_factor_scenarios, start=1):
        sid = f"SF_{k:03d}"
        sc_with_id = dict(sc)
        sc_with_id["scenario_id"] = sid

        if add_absolute_values:
            rho_scale = sc_with_id["rho_scale"]
            omega_scale   = sc_with_id["omega_scale"]
            sc_with_id["rho_abs"]   = rho_reference * rho_scale
            sc_with_id["omega_abs"] = omega_reference * omega_scale

        #scenarios_dict[sid] = sc_with_id
        scenarios_dict[k] = sc_with_id

    number_SF_scenarios = k

    # --- LHS ---
    for k, sc in enumerate(lhs_scenarios, start=1):
        sid = f"LHS_{k:03d}"
        sc_with_id = dict(sc)
        sc_with_id["scenario_id"] = sid

        if add_absolute_values:
            rho_scale = sc_with_id["rho_scale"]
            omega_scale   = sc_with_id["omega_scale"]
            sc_with_id["rho_abs"]   = rho_reference * rho_scale
            sc_with_id["omega_abs"] = omega_reference * omega_scale

        scenarios_dict[(number_SF_scenarios + k)] = sc_with_id

    return scenarios_dict


if __name__ == "__main__":
    scenarios = build_scenario_dict(n_lhs=50)  # per prova metto 10 LHS

    print(f"Number of total scenarios: {len(scenarios)}\n")

    # Dictionary parameters (both the absolute version and the scale factor)
    # rho_scale
    # Omega_scale
    # tilt_deg
    # az_deg
    # spin_axis

    np.savez(
        path_scenarios, scenarios=scenarios
    )

    for i, (sid, sc) in enumerate(scenarios.items()):
        print(f"--- {sid} ---")
        for k, v in sc.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {v.tolist()}")
            else:
                print(f"{k}: {v}")
        print()
        if i == 30:
            break

    data = np.load(path_scenarios, allow_pickle=True)
    scenarios = data["scenarios"]
    print(scenarios)

