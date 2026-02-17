import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pinnsformer_minimal import PINNsformer, get_positional_encoding

# Workaround to solve a conflict (to be corrected)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========================= USER SETTINGS ================================
TIME_INSTANTS = 16
BATCH_SIZE = 256

MODEL_PATH = "Models/best_test_model_NET_no_rot_no_relu.pt"
TRAJ_TEST_PATH = "train_test_traj/trajectories_test.npy"   # (16, 6, N)
CTRL_TEST_PATH = "train_test_traj/control_test.npy"        # (16, 3, N)

OUT_DIR = "loss_plots"

# Scaling (same as training)
LU = 36_000.0
VU = 18.0
CU = 0.6

# Network architecture (same as training)
D_MODEL = 512
N_LAYERS = 2
N_HEADS = 8
D_HIDDEN = 4 * D_MODEL
D_EMB_IN = 9
D_FINAL = 6
DROPOUT = 0.0

# 3D view control
# If INTERACTIVE_3D=True, figures will open and you can rotate with the mouse.
INTERACTIVE_3D = True

# If INTERACTIVE_3D=False, these angles define the saved view.
ELEV = 25.0   # elevation angle (degrees)
AZIM = 45.0   # azimuth angle (degrees)
ROLL = None   # optional; works only on newer matplotlib (otherwise ignored)

# 3D axis aesthetics
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 9
TICK_PAD = 6          # distance between ticks and axis

# Plot style
SEABORN_CONTEXT = "talk"  # "paper", "notebook", "talk", "poster"
SEABORN_STYLE = "whitegrid"
# ============================================================================


def plot_orbit_3d(
    idx: int,
    pred_traj: torch.Tensor,
    true_traj: torch.Tensor,
    folder: str,
    title: str,
    LU: float,
    elev: float | None,
    azim: float | None,
    roll: float | None,
    interactive: bool,
):
    """
    Save a 3D orbit plot in meters.
    - interactive=True: opens the window so you can rotate with mouse; close it to proceed.
    - interactive=False: uses provided elev/azim/roll to set a reproducible viewpoint.
    """
    true_xyz = (true_traj[:, :3] * LU).detach().cpu().numpy()
    pred_xyz = (pred_traj[:, :3] * LU).detach().cpu().numpy()

    # Seaborn theme for consistent aesthetics
    sns.set_theme(context=SEABORN_CONTEXT, style=SEABORN_STYLE)

    fig = plt.figure(figsize=(7.8, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2],
            label="True", linewidth=2)
    ax.plot(
        pred_xyz[:, 0],
        pred_xyz[:, 1],
        pred_xyz[:, 2],
        label="Predicted",
        linestyle="--",
        linewidth=2,
    )

    ax.set_xlabel("x [m]", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("y [m]", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_zlabel("z [m]", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)

    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE, pad=TICK_PAD)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE, pad=TICK_PAD)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_FONTSIZE, pad=TICK_PAD)

    ax.set_box_aspect([1, 1, 1])  # assi con scala uniforme

    ax.set_title(f"{title} – Orbit {idx}")
    ax.legend()
    ax.grid(True)

    # Set viewpoint (only if not interactive)
    if not interactive:
        if elev is not None or azim is not None:
            ax.view_init(
                elev=elev if elev is not None else ax.elev,
                azim=azim if azim is not None else ax.azim
            )
        if roll is not None:
            # roll supported only in newer matplotlib versions
            try:
                ax.view_init(elev=ax.elev, azim=ax.azim, roll=roll)
            except TypeError:
                pass

    os.makedirs(folder, exist_ok=True)
    plt.tight_layout()

    if interactive:
        # Rotate with mouse, then close the window to save the final view
        plt.show()

    out_path = os.path.join(folder, f"orbit_{idx}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sns.set_theme(context=SEABORN_CONTEXT, style=SEABORN_STYLE)

    # =================== 1) Load test dataset ===================
    if not os.path.exists(TRAJ_TEST_PATH):
        raise FileNotFoundError(f"Missing trajectories file: {TRAJ_TEST_PATH}")
    if not os.path.exists(CTRL_TEST_PATH):
        raise FileNotFoundError(f"Missing controls file: {CTRL_TEST_PATH}")

    trajectories_test = np.load(TRAJ_TEST_PATH)  # (16, 6, N)
    controls_test = np.load(CTRL_TEST_PATH)      # (16, 3, N)

    # Trim controls (remove last instant): (15,3,N)
    controls_test = controls_test[:-1, :, :]

    # Scaling
    trajectories_test[:, 0:3, :] /= LU
    trajectories_test[:, 3:6, :] /= VU
    controls_test[:, :, :] /= CU

    # Tensorize
    x0_test = torch.tensor(trajectories_test[0, :, :].T, dtype=torch.float32)  # (N,6)
    true_traj_test = torch.tensor(
        trajectories_test.transpose(2, 0, 1), dtype=torch.float32
    )  # (N,16,6)

    ctr_test_seq = torch.tensor(
        controls_test.transpose(2, 0, 1), dtype=torch.float32
    )  # (N,15,3)

    dataset = TensorDataset(x0_test, ctr_test_seq, true_traj_test)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # =================== 2) Model & weights ===================
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    Pos_src = get_positional_encoding(TIME_INSTANTS - 1, D_MODEL).to(device)

    model = PINNsformer(
        d_model=D_MODEL,
        d_hidden=D_HIDDEN,
        d_emb_input=D_EMB_IN,
        d_final=D_FINAL,
        N=N_LAYERS,
        heads=N_HEADS,
        dropout=DROPOUT,
        Pos_src=Pos_src,
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    # expecting {"model": state_dict}
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Model loaded from '{MODEL_PATH}'")

    # =================== 3) Inference & reconstruction ===================
    predictions = []
    with torch.no_grad():
        for x0_b, ctr_b, _ in loader:
            x0_b = x0_b.to(device)     # (B,6)
            ctr_b = ctr_b.to(device)   # (B,15,3)

            delta_pred = model(x0_b, ctr_b)  # (B,15,6)

            # Reconstruction
            full_traj = torch.cumsum(
                torch.cat([x0_b.unsqueeze(1), delta_pred], dim=1), dim=1
            )  # (B,16,6)
            predictions.append(full_traj.cpu())

    pred = torch.cat(predictions, dim=0).to(device)  # (N,16,6)
    true_traj_test = true_traj_test.to(device)

    # =================== 4) RMSE metrics (physical units) ===================
    pos_err = (pred[:, :, :3] - true_traj_test[:, :, :3]) * LU
    vel_err = (pred[:, :, 3:] - true_traj_test[:, :, 3:]) * VU

    rmse_pos = torch.sqrt((pos_err ** 2).sum(dim=2).mean(dim=1)).cpu().numpy()
    rmse_vel = torch.sqrt((vel_err ** 2).sum(dim=2).mean(dim=1)).cpu().numpy()

    overall_rmse_pos = float(rmse_pos.mean())
    overall_rmse_vel = float(rmse_vel.mean())

    print("\nTEST RESULTS")
    print(f"Mean Position RMSE: {overall_rmse_pos:9.3f}  m")
    print(f"Mean Velocity RMSE: {overall_rmse_vel:9.6f}  m/s")

    # =================== 5) Best / worst ===================
    sorted_idx = np.argsort(rmse_pos)
    best_10_idx = sorted_idx[:10]
    worst_10_idx = sorted_idx[::-1][:10]

    def report_subset(name: str, indices: np.ndarray):
        print(f"\n{name}")
        print("Idx  |   Pos. RMSE [m] |   Vel. RMSE [m/s]")
        for idx in indices:
            i = int(idx)
            print(f"{i:4d} | {rmse_pos[i]:14.3f} | {rmse_vel[i]:16.6f}")

    report_subset("Best orbits (position)", best_10_idx)
    report_subset("Worst orbits (position)", worst_10_idx)

    # =================== 6) Seaborn histograms ======================
    os.makedirs(OUT_DIR, exist_ok=True)

    # Position RMSE histogram
    plt.figure(figsize=(9, 5))
    sns.histplot(rmse_pos, bins=30, stat="count", kde=False)
    plt.xlabel("Position RMSE [m]")
    plt.ylabel("Count")
    plt.title("Distribution of Position RMSE over Test Orbits")
    plt.tight_layout()
    pos_hist_path = os.path.join(OUT_DIR, "test_rmse_position_seaborn.png")
    plt.savefig(pos_hist_path, dpi=200)
    plt.close()

    # Velocity RMSE histogram
    plt.figure(figsize=(9, 5))
    sns.histplot(rmse_vel, bins=30, stat="count", kde=False)
    plt.xlabel("Velocity RMSE [m/s]")
    plt.ylabel("Count")
    plt.title("Distribution of Velocity RMSE over Test Orbits")
    plt.tight_layout()
    vel_hist_path = os.path.join(OUT_DIR, "test_rmse_velocity_seaborn.png")
    plt.savefig(vel_hist_path, dpi=200)
    plt.close()

    print(f"\nSaved histogram (position): {pos_hist_path}")
    print(f"Saved histogram (velocity): {vel_hist_path}")

    # =================== 7) Save 3 best + 3 worst orbits (3D) ===================
    print("\nSaving orbits (3 best + 3 worst)...")
    best_dir = os.path.join(OUT_DIR, "best_orbits")
    worst_dir = os.path.join(OUT_DIR, "worst_orbits")

    for i in range(3):
        idx_best = int(best_10_idx[i])
        idx_worst = int(worst_10_idx[i])

        plot_orbit_3d(
            idx_best, pred[idx_best], true_traj_test[idx_best],
            folder=best_dir, title=f"Best #{i+1}", LU=LU,
            elev=ELEV, azim=AZIM, roll=ROLL,
            interactive=INTERACTIVE_3D
        )
        plot_orbit_3d(
            idx_worst, pred[idx_worst], true_traj_test[idx_worst],
            folder=worst_dir, title=f"Worst #{i+1}", LU=LU,
            elev=ELEV, azim=AZIM, roll=ROLL,
            interactive=INTERACTIVE_3D
        )

    print(f"Saved 3D plots to: {best_dir} and {worst_dir}")
    if INTERACTIVE_3D:
        print("INTERACTIVE_3D=True: rotate each figure with the mouse, then close it to save.")


if __name__ == "__main__":
    main()
