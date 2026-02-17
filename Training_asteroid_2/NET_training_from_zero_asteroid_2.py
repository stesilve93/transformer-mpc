import os, random
from torch.optim import AdamW
from pinnsformer_minimal import PINNsformer, get_positional_encoding
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)

from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules_training import *

from pathlib import Path


# NOTE: Implemented also the physical loss in the training process.
# To disable the physical learning --> gamma_curr = 0

# -------------- DIRECTORIES -------------------------------
ROOT = Path(__file__).resolve().parents[1]
CURR_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURR_DIR / "Models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = MODELS_DIR / "best_test_model_no_rot_no_relu_NET_asteroid_2.pt"
# -----------------------------------------------------------


def main():
    # ------------ SETTINGS -------------------
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device in uso:", device)

    # ----------- DATA LOADING ------------------
    trajectories_train = np.load(
        str(CURR_DIR / "train_test_traj" / "trajectories_train.npy")
    )
    trajectories_test = np.load(
        str(CURR_DIR / "train_test_traj" / "trajectories_test.npy")
    )
    controls_train = np.load(
        str(CURR_DIR / "train_test_traj" / "control_train.npy")
    )
    controls_test = np.load(
        str(CURR_DIR / "train_test_traj" / "control_test.npy")
    )

    controls_train = controls_train[:-1, :, :]
    controls_test = controls_test[:-1, :, :]

    # ---------- ADIMENSIONAL UNITS -----------------
    LU = 36_000
    VU = 18.0
    CU = 0.6

    # ------ ADIMENSIONALISATION TRAJECTORIES AND CONTROLS ------
    trajectories_train[:, 0:3, :] /= LU
    trajectories_train[:, 3:6, :] /= VU
    trajectories_test[:, 0:3, :] /= LU
    trajectories_test[:, 3:6, :] /= VU

    controls_train[:, :, :] /= CU
    controls_test[:, :, :] /= CU

    time_instants = 16  # number of time instants

    # --------- CONTROLS PREPARATION ---------
    # (B=N, L=16, 3)
    # (16,3,N) → (N,16,3) → (N,105)
    ctr_train_np = controls_train.transpose(2, 0, 1)
    ctr_test_np = controls_test.transpose(2, 0, 1)
    ctr_train_fl = ctr_train_np
    ctr_test_fl = ctr_test_np

    controls_train_tensor = torch.tensor(ctr_train_fl, dtype=torch.float32)  # (N,16,3)
    controls_test_tensor = torch.tensor(ctr_test_fl, dtype=torch.float32)  # (N,16,3)

    # --------- INITIAL STATE ---------
    x0_train = torch.tensor(
        trajectories_train[0, :, :].T, dtype=torch.float32
    )  # (N_train,6)
    x0_test = torch.tensor(
        trajectories_test[0, :, :].T, dtype=torch.float32
    )  # (N_test,6)

    # --------- TRUE DELTA ---------
    tt_train = torch.tensor(
        trajectories_train.transpose(2, 0, 1), dtype=torch.float32
    )  # (N,16,6)
    tt_test = torch.tensor(
        trajectories_test.transpose(2, 0, 1), dtype=torch.float32
    )  # (N,16,6)
    true_delta_train = tt_train[:, 1:, :] - tt_train[:, :-1, :]  # (N,15,6)
    true_delta_test = tt_test[:, 1:, :] - tt_test[:, :-1, :]  # (N,15,6)

    # ---------- DATASET & DATALOADER ------------
    batch_size = 1024

    train_dataset = TensorDataset(x0_train, controls_train_tensor, true_delta_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_dataset = TensorDataset(x0_test, controls_test_tensor, true_delta_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ------- POSITIONAL ENCODING ---------------------
    d_model = 512  # embedding dimension (before encoder/decoder section)

    Pos_encoding_matrix = get_positional_encoding((time_instants - 1), d_model)
    Pos_encoding_matrix = Pos_encoding_matrix.to(device)

    # -------------- MODEL -----------------------
    model = PINNsformer(
        d_model=d_model,
        d_hidden=4 * d_model,
        d_emb_input=9,  # dimension of each input token (initial state + control)
        d_final=6,
        N=2,
        heads=8,
        dropout=0,
        Pos_src=Pos_encoding_matrix,
    ).to(device)

    # ------------ PARAMETERS ---------------
    N_epochs = 6
    test_every = 5
    dt_sec = 60

    # ----------- INITIALISATION PHYSICAL TERMS -----------------
    phys = PhysicsParams().to(
        device
    )  # (N=2, omega_init=0.001, mu_init=1.0, a_e=16000.0)

    # ------------ OPTIMIZER AND SCHEDULER ----------------------
    optim_model = AdamW(model.parameters(), lr=3e-5)
    optim_phys = AdamW(phys.parameters(), lr=3e-8)

    sched_model = SequentialLR(
        optim_model,
        schedulers=[
            LinearLR(optim_model, start_factor=1, end_factor=1, total_iters=25),
            CosineAnnealingLR(optim_model, T_max=N_epochs - 25, eta_min=1e-8),
        ],
        milestones=[25],
    )

    sched_phys = SequentialLR(
        optim_phys,
        schedulers=[
            LinearLR(optim_phys, start_factor=1e-4, end_factor=1, total_iters=500),
            CosineAnnealingLR(optim_phys, T_max=N_epochs - 1000, eta_min=1e-9),
        ],
        milestones=[500],
    )

    # ------------- TEMPORAL WEIGHTS -----------------
    w = torch.linspace(1.0, 0.1, time_instants - 1, device=device)
    w = (w / w.sum()).view(1, -1, 1)  # shape (1, 34, 1)

    # ----------- INITIALISATION TRAINING PARAMETERS ------------
    loss_track = []
    test_loss_track = []
    best_test_loss = None

    alpha = 0.3  # local error
    beta = 0.4  # global error
    gamma_curr = 0.3

    scaler = GradScaler()

    ##########################################################################################
    # TRAINING
    ##########################################################################################
    for epoch in tqdm(range(N_epochs), desc="Epochs"):
        model.train()
        batch_losses = []
        batch_delta_losses = []
        batch_traj_losses = []
        batch_phys_losses = []

        for x0_b, ctr_b, delta_b in train_loader:
            x0_b = x0_b.to(device)
            ctr_b = ctr_b.to(device)
            delta_b = delta_b.to(device)

            with autocast():
                # x0_b --> (B, 6)
                # ctrl_b --> (B, 15, 3)
                out = model(x0_b, ctr_b)

                delta_pos_loss = vector_loss_weights(
                    out[:, :, :3], delta_b[:, :, :3], w
                )
                delta_vel_loss = vector_loss_weights(
                    out[:, :, 3:], delta_b[:, :, 3:], w
                )
                delta_loss = 0.6 * delta_pos_loss + 0.4 * delta_vel_loss

                # Trajectory reconstruction
                pred_traj = torch.cumsum(
                    torch.cat([x0_b.unsqueeze(1), out], dim=1), dim=1
                )[:, 1:, :]
                true_traj = torch.cumsum(
                    torch.cat([x0_b.unsqueeze(1), delta_b], dim=1), dim=1
                )[:, 1:, :]

                # Loss computation
                traj_pos_loss = vector_loss(pred_traj[:, :, :3], true_traj[:, :, :3])
                traj_vel_loss = vector_loss(pred_traj[:, :, 3:], true_traj[:, :, 3:])
                traj_loss = 0.6 * traj_pos_loss + 0.4 * traj_vel_loss

            # Physical component loss
            pred_pos_phys = pred_traj[..., 0:3].float() * LU
            pred_vel_phys = pred_traj[..., 3:6].float() * VU
            dV_phys = ctr_b * CU
            pred_traj_phys = torch.cat([pred_pos_phys, pred_vel_phys], dim=-1)

            phys_loss = physics_residuals_with_impulses(
                pred_traj_phys, dV_phys, phys, dt_sec, device
            )

            # Total loss
            loss = alpha * delta_loss + beta * traj_loss + gamma_curr * phys_loss

            optim_model.zero_grad(set_to_none=True)
            optim_phys.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            scaler.unscale_(optim_model)
            scaler.unscale_(optim_phys)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(phys.parameters()), 1.0
            )

            scaler.step(optim_model)
            scaler.step(optim_phys)
            scaler.update()

            batch_losses.append(loss.item())
            batch_delta_losses.append(delta_loss.item())
            batch_traj_losses.append(traj_loss.item())
            batch_phys_losses.append(phys_loss.item())

        # Average values per epoch (for printing)
        avg_total_loss = np.mean(batch_losses)
        avg_delta_loss = np.mean(batch_delta_losses)
        avg_traj_loss = np.mean(batch_traj_losses)
        avg_phys_loss = np.mean(batch_phys_losses)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Delta Loss: {avg_delta_loss:.6e} | Trajectory Loss: {avg_traj_loss:.6e} | Physical Loss: {avg_phys_loss:.6e} |"
        )

        avg_train = float(np.mean(batch_losses))
        loss_track.append(avg_train)

        # --------------- Scheduler step ---------------
        sched_model.step()
        sched_phys.step()

        lr_m = optim_model.param_groups[0]["lr"]
        lr_p = optim_phys.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d} | LR(model) {lr_m:.2e} | LR(phys) {lr_p:.2e} | Train Loss {avg_train * 1e4:.6f}"
        )
        #####################################################################################
        # VALIDATION
        #####################################################################################
        if (epoch + 1) % test_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad(), autocast():
                for x0_b, ctr_b, delta_b in test_loader:
                    x0_b = x0_b.to(device)
                    ctr_b = ctr_b.to(device)
                    delta_b = delta_b.to(device)

                    out = model(x0_b, ctr_b)

                    delta_pos_loss = vector_loss_weights(
                        out[:, :, :3], delta_b[:, :, :3], w
                    )
                    delta_vel_loss = vector_loss_weights(
                        out[:, :, 3:], delta_b[:, :, 3:], w
                    )
                    delta_loss = 0.6 * delta_pos_loss + 0.4 * delta_vel_loss

                    # Trajectory reconstruction
                    pred_traj = torch.cumsum(
                        torch.cat([x0_b.unsqueeze(1), out], dim=1), dim=1
                    )[:, 1:, :]
                    true_traj = torch.cumsum(
                        torch.cat([x0_b.unsqueeze(1), delta_b], dim=1), dim=1
                    )[:, 1:, :]

                    # Loss computation
                    traj_pos_loss = vector_loss(
                        pred_traj[:, :, :3], true_traj[:, :, :3]
                    )
                    traj_vel_loss = vector_loss(
                        pred_traj[:, :, 3:], true_traj[:, :, 3:]
                    )
                    traj_loss = 0.6 * traj_pos_loss + 0.4 * traj_vel_loss

                    # Physical loss computation
                    pred_pos_phys = pred_traj[..., 0:3].float() * LU
                    pred_vel_phys = pred_traj[..., 3:6].float() * VU
                    dV_phys = ctr_b * CU
                    pred_traj_phys = torch.cat([pred_pos_phys, pred_vel_phys], dim=-1)
                    phys_loss = physics_residuals_with_impulses(
                        pred_traj_phys, dV_phys, phys, dt_sec, device
                    )

                    val_loss = (
                        (alpha * delta_loss)
                        + beta * (traj_loss)
                        + gamma_curr * phys_loss
                    )
                    val_losses.append(val_loss.item())

            avg_test = float(np.mean(val_losses))
            test_loss_track.append(avg_test)
            print(f"[Epoch {epoch + 1:03d}] Test Loss {avg_test * 1e4:.6f}")

            # Saving best model
            if best_test_loss is None or avg_test < best_test_loss:
                best_test_loss = avg_test
                torch.save(
                    {
                        "model": model.state_dict(),
                        "phys": phys.state_dict(),
                    },
                    str(BEST_PATH),
                )
                print(f"Best model updated: {BEST_PATH}")

    ############################################################################################
    # ---------------- Saving the loss vectors -----------------------
    np.save("loss_plots/train_loss_track.npy", np.array(loss_track))
    np.save("loss_plots/test_loss_track.npy", np.array(test_loss_track))

    # ---------------- Loss plots ------------------------------------
    os.makedirs("loss_plots", exist_ok=True)

    plt.figure()
    # plt.plot(loss_track[5:], label="Training Loss")
    plt.plot(loss_track, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plots/train_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(test_loss_track, label="Testing Loss")
    plt.title("Testing Loss")
    plt.xlabel(f"Epochs (per {test_every})")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plots/testing_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    # plt.plot(loss_track[5:], label="Train")
    plt.plot(loss_track, label="Train")
    plt.plot(
        np.arange(0, test_every * len(test_loss_track), test_every),
        test_loss_track,
        label="Test",
    )
    plt.title("Training & Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plots/train_test_loss.png", dpi=200)
    plt.close()

    print("Training completed! Plots saved in 'loss_plots/'.")


if __name__ == "__main__":
    main()
