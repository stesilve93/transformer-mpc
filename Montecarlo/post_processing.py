import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# =========================
# Config
# =========================
PARAMS_FILE = "Parameters_by_scenario.npy"  # relative or absolute path
OUT_DIR = "post_processing_plots"
SAVE_FIGS = True
SHOW_FIGS = False

rho_true = 2670.0
G = 6.67430e-11
T_eros = 5.27 * 3600.0
Omega_true = 2 * math.pi / T_eros

rho_reference = rho_true * 1.10
omega_reference = Omega_true * 1.01

# Quale "APE" vuoi plottare?
# Select the APE you want to plot
# Options available in the dict: "max_APE_pos", "p95_APE_pos", "p90_APE_pos", "p80_APE_pos" (and *_vel)
APE_POS_KEY = "max_APE_pos"
APE_VEL_KEY = "max_APE_vel"

MPE_POS_KEY = "mean_MPE_pos"
MPE_VEL_KEY = "mean_MPE_vel"

# =========================
# Load
# =========================
if not os.path.exists(PARAMS_FILE):
    raise FileNotFoundError(f"Cannot find '{PARAMS_FILE}'. CWD is: {os.getcwd()}")

params_obj = np.load(PARAMS_FILE, allow_pickle=True).item()

# params_obj expected: dict indexed by scenario k -> dict of values
if not isinstance(params_obj, dict) or len(params_obj) == 0:
    raise ValueError("Loaded Parameters_by_scenario is empty or not a dict.")

# =========================
# Convert to DataFrame
# =========================
rows = []
for k, d in params_obj.items():
    if d is None:
        continue
    row = {"scenario_id": k}
    row.update(d)
    rows.append(row)

df = pd.DataFrame(rows)

for col in df.columns:
    if col != "scenario_id":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# Print summary
# =========================
n_sims = len(df)
print(f"\nNumber of simulations in Parameters_by_scenario: {n_sims}\n")

missing_cols = [c for c in [MPE_POS_KEY, MPE_VEL_KEY, APE_POS_KEY, APE_VEL_KEY] if c not in df.columns]
if missing_cols:
    raise KeyError(f"Missing expected keys in the dataset: {missing_cols}")

# ==========================================================
# Monte Carlo statistics: position and velocity
# ==========================================================

def compute_stats(series, label):
    series = series.dropna()
    stats = {
        "mean": series.mean(),
        "std": series.std(),
        "median": series.median(),
        "p95": series.quantile(0.95),
        "max": series.max()
    }

    print(f"\nMonte Carlo statistics — {label}")
    for k, v in stats.items():
        print(f"  {k:>7}: {v:.6g}")

    return stats


stats = {}

# --- Position statistics ---
stats["MPE_pos"] = compute_stats(df[MPE_POS_KEY], "MPE Position")
stats["APE_pos"] = compute_stats(df[APE_POS_KEY], "APE Position")

# --- Velocity statistics ---
stats["MPE_vel"] = compute_stats(df[MPE_VEL_KEY], "MPE Velocity")
stats["APE_vel"] = compute_stats(df[APE_VEL_KEY], "APE Velocity")


# =========================
# Scenario with highest MPE (position)
# =========================
idx_max_mpe = df[MPE_POS_KEY].idxmax()
worst_mpe_row = df.loc[idx_max_mpe]

rho_worst = worst_mpe_row.get('rho', np.nan)
omega_worst = worst_mpe_row.get('omega', np.nan)
rho_perc = rho_worst / rho_reference
omega_perc = omega_worst / omega_reference

print("Scenario with highest MPE (position)")
print(f"  scenario_id : {int(worst_mpe_row['scenario_id'])}")
print(f"  rho         : {worst_mpe_row.get('rho', np.nan):.6g}")
print(f"  rho perc    : {rho_perc:.6g}")
print(f"  omega       : {worst_mpe_row.get('omega', np.nan):.6g}")
print(f"  omega perc  : {omega_perc:.6g}")
print(f"  tilt_deg    : {worst_mpe_row.get('tilt_deg', np.nan):.6g}")
print(f"  az_deg      : {worst_mpe_row.get('az_deg', np.nan):.6g}")
print(f"  {MPE_POS_KEY}: {worst_mpe_row[MPE_POS_KEY]:.6g}")
print(f"  {MPE_VEL_KEY}: {worst_mpe_row[MPE_VEL_KEY]:.6g}")
print(f"  {APE_POS_KEY}: {worst_mpe_row[APE_POS_KEY]:.6g}")
print(f"  {APE_VEL_KEY}: {worst_mpe_row[APE_VEL_KEY]:.6g}\n")

# =========================
# Plots (Seaborn)
# =========================
sns.set_theme(style="whitegrid")

os.makedirs(OUT_DIR, exist_ok=True)

def save_or_show(fig, filename):
    if SAVE_FIGS:
        out_path = os.path.join(OUT_DIR, filename)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")
    if SHOW_FIGS:
        plt.show()
    plt.close(fig)

# ---- 1) MPE histograms (pos + vel) ----
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df[MPE_POS_KEY].dropna(), bins=30, kde=False, ax=axes1[0])
#axes1[0].set_title("Histogram of MPE (Position)")
axes1[0].set_xlabel("Mean Position Error (MPE)")
axes1[0].set_ylabel("Count")

sns.histplot(df[MPE_VEL_KEY].dropna(), bins=30, kde=False, ax=axes1[1])
#axes1[1].set_title("Histogram of MPE (Velocity)")
axes1[1].set_xlabel("Mean Velocity Error")
axes1[1].set_ylabel("Count")

#fig1.suptitle(f"MPE Distributions ({n_sims} simulations)", y=1.02)
fig1.tight_layout()
save_or_show(fig1, "hist_MPE_pos_vel.png")

# ---- 2) APE histograms (pos + vel) ----
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df[APE_POS_KEY].dropna(), bins=30, kde=False, ax=axes2[0])
#axes2[0].set_title(f"Histogram of APE (Position) [{APE_POS_KEY}]")
axes2[0].set_xlabel("Absolute Position Error (APE)")
axes2[0].set_ylabel("Count")

sns.histplot(df[APE_VEL_KEY].dropna(), bins=30, kde=False, ax=axes2[1])
#axes2[1].set_title(f"Histogram of APE (Velocity) [{APE_VEL_KEY}]")
axes2[1].set_xlabel("Absolute Velocity Error")
axes2[1].set_ylabel("Count")

#fig2.suptitle(f"APE Distributions ({n_sims} simulations)", y=1.02)
fig2.tight_layout()
save_or_show(fig2, "hist_APE_pos_vel.png")

print("\nPost-processing completed.\n")
