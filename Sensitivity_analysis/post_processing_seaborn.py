import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ================== SETTINGS ==================
METRIC_KEY = "mean_MPE_pos"   # es: "p95_APE_pos", "max_APE_pos", ecc.

# stile seaborn
sns.set_theme(
    context="paper",
    style="whitegrid",
    font_scale=1.2
)

# ================== LOAD DATA ==================
parameters_by_scenario = np.load(
    "Parameters_by_scenario.npy", allow_pickle=True
).item()

scenarios_npz = np.load("scenarios_parameters.npz", allow_pickle=True)
scenarios_dict = scenarios_npz["scenarios"].item()

# ================== GATHERING DATA ==================
rho_data   = []
omega_data = []
tilt_data  = []

for key, scenario in scenarios_dict.items():
    scenario_id = scenario["scenario_id"]

    # solo single-factor
    if not scenario_id.startswith("SF"):
        continue

    rho_scale   = scenario["rho_scale"]
    omega_scale = scenario["omega_scale"]
    tilt_deg    = scenario["tilt_deg"]
    az_deg      = scenario["az_deg"]

    metric_value = parameters_by_scenario[key][METRIC_KEY]

    # 1) rho sweep
    if tilt_deg == 0.0 and az_deg == 0.0 and abs(omega_scale - 1.0) < 1e-8:
        rho_data.append(
            {"rho_scale": rho_scale, "metric": metric_value}
        )

    # 2) omega sweep
    elif tilt_deg == 0.0 and az_deg == 0.0 and abs(rho_scale - 1.0) < 1e-8:
        omega_data.append(
            {"omega_scale": omega_scale, "metric": metric_value}
        )

    # 3) tilt sweep
    elif abs(rho_scale - 1.0) < 1e-8 and abs(omega_scale - 1.0) < 1e-8:
        tilt_data.append(
            {"tilt_deg": tilt_deg, "metric": metric_value}
        )

# ================== DATAFRAME ==================
df_rho   = pd.DataFrame(rho_data).sort_values("rho_scale")
df_omega = pd.DataFrame(omega_data).sort_values("omega_scale")
df_tilt  = pd.DataFrame(tilt_data)

df_tilt_mean = (
    df_tilt.groupby("tilt_deg", as_index=False)
           .mean()
           .rename(columns={"metric": "mean_metric"})
)

df_tilt_max = (
    df_tilt.groupby("tilt_deg", as_index=False)
           .max()
           .rename(columns={"metric": "max_metric"})
)

# ================== PLOTS ==================

# --- 1) Accuracy vs rho ---
plt.figure(figsize=(6,4))
sns.lineplot(
    data=df_rho,
    x="rho_scale",
    y="metric",
    marker="o"
)
plt.xlabel(r"$\rho_\mathrm{scale}$")
# plt.ylabel(f"{METRIC_KEY} [m]")
plt.ylabel("MPE [m]")
plt.title("Tracking accuracy vs density variation")
plt.tight_layout()

# --- 2) Accuracy vs omega ---
plt.figure(figsize=(6,4))
sns.lineplot(
    data=df_omega,
    x="omega_scale",
    y="metric",
    marker="o"
)
plt.xlabel(r"$\omega_\mathrm{scale}$")
#plt.ylabel(f"{METRIC_KEY} [m]")
plt.ylabel("MPE [m]")
plt.title("Tracking accuracy vs angular velocity variation")
plt.tight_layout()

# --- 3) Accuracy vs tilt (mean & worst azimuth) ---
plt.figure(figsize=(6,4))
sns.lineplot(
    data=df_tilt_mean,
    x="tilt_deg",
    y="mean_metric",
    marker="o",
    label="Mean over azimuth"
)
sns.lineplot(
    data=df_tilt_max,
    x="tilt_deg",
    y="max_metric",
    marker="s",
    linestyle="--",
    label="Worst-case over azimuth"
)

plt.xlabel("Tilt [deg]")
# plt.ylabel(f"{METRIC_KEY} [m]")
plt.ylabel("MPE [m]")
plt.title("Tracking accuracy vs spin-axis inclination")
plt.legend()
plt.tight_layout()

plt.show()
