from pathlib import Path
from torch import nn
import torch

from PINN.pinn import PINN, total_loss, DEVICE
from PINN.predictor import Predictor
from GEOMETRY.collocate import assemble_collocation_points
from GEOMETRY.probes import centerline_points
from TRAIN.monitor import (
    monitor_crosssectional_massflow,
    monitor_centerline_velocity_rho_p,
)
from DIRS.dirs import MODEL_PATH

torch.set_default_dtype(torch.float32)

# --------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------

EPOCHS = 20000
LEARNING_RATE = 2.0e-3


def train_pinn(model: nn.Module, epochs: int = EPOCHS):
    """
    Train the Physics-Informed Neural Network.
    """

    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    history = {
        "loss": [],
        "pde": [],
        "massflow": [],
        "sonic_throat": [],
        # "axisymmetry": [],
        "wall": [],
        "inlet": [],
        "outlet": [],
    }
    best_loss = float("inf")

    # For monitoring centerline velocity:
    predictor = Predictor(model=model, device=DEVICE)
    x_centerline = centerline_points(n_points=7)
    fo_loss = open("results/loss_history.txt", "w")

    ratio_files = {}
    for throat_ratio in (0.2, 0.3, 0.4):
        ratio_label = f"tr{int(throat_ratio * 100):02d}"
        ratio_files[throat_ratio] = {
            "velocity": open(f"results/centerline_velocity_{ratio_label}.txt", "w"),
            "massflow": open(f"results/centerline_massflow_{ratio_label}.txt", "w"),
            "rho": open(f"results/centerline_rho_{ratio_label}.txt", "w"),
            "p": open(f"results/centerline_p_{ratio_label}.txt", "w"),
        }
        for key in ("velocity", "massflow", "rho", "p"):
            for xi in x_centerline:
                ratio_files[throat_ratio][key].write(f"{xi[0].item():.6f} ")
            ratio_files[throat_ratio][key].write("\n")

    for key in history:
        fo_loss.write(f"{key} ")
    fo_loss.write("\n")

    for epoch in range(epochs):

        # --------------------------------------------------------------
        # Resample collocation points
        # --------------------------------------------------------------

        x_interior, x_wall, x_inlet, x_outlet, sampled_throat_ratios = (
            assemble_collocation_points()
        )
        optimizer.zero_grad()

        (
            loss,
            loss_pde,
            loss_massflow,
            loss_sonic_throat,
            # loss_axisymmetry,
            loss_wall,
            loss_inlet,
            loss_outlet,
        ) = total_loss(
            model,
            x_interior,
            x_wall,
            x_inlet,
            x_outlet,
            sampled_throat_ratios,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["massflow"].append(loss_massflow.item())
        history["sonic_throat"].append(loss_sonic_throat.item())
        # history["axisymmetry"].append(loss_axisymmetry.item())
        history["wall"].append(loss_wall.item())
        history["inlet"].append(loss_inlet.item())
        history["outlet"].append(loss_outlet.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        if epoch % 1 == 0 or epoch == epochs - 1:
            lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch:6d} | "
                f"Loss {loss.item():10.4e} | "
                f"PDE {loss_pde.item():10.4e} | "
                f"MassFlow {loss_massflow.item():10.4e} | "
                f"SonicThroat {loss_sonic_throat.item():10.4e} | "
                # f"Axisymmetry {loss_axisymmetry.item():10.4e} | "
                f"Wall {loss_wall.item():10.4e} | "
                f"Inlet {loss_inlet.item():10.4e} | "
                f"Outlet {loss_outlet.item():10.4e} | "
                f"LR {lr:.2e}"
            )

            # if epoch % 1 == 0 or epoch == epochs - 1:
            for throat_ratio, files in ratio_files.items():
                rho_tr, velocity_magnitude_tr, p_tr = monitor_centerline_velocity_rho_p(
                    predictor, throat_ratio=throat_ratio, x_centerline=x_centerline
                )
                massflow_tr = monitor_crosssectional_massflow(
                    predictor, throat_ratio=throat_ratio, x_centerline=x_centerline
                )

                for vi in velocity_magnitude_tr:
                    files["velocity"].write(f"{vi[0].item():.6f} ")
                files["velocity"].write("\n")
                files["velocity"].flush()

                for value in massflow_tr:
                    files["massflow"].write(f"{value[0].item():.6f} ")
                files["massflow"].write("\n")
                files["massflow"].flush()

                for value in rho_tr:
                    files["rho"].write(f"{value[0].item():.6f} ")
                files["rho"].write("\n")
                files["rho"].flush()

                for value in p_tr:
                    files["p"].write(f"{value[0].item():.6f} ")
                files["p"].write("\n")
                files["p"].flush()

            for key in history:
                fo_loss.write(f"{history[key][-1]:.6e} ")
            fo_loss.write("\n")
            fo_loss.flush()

    fo_loss.close()
    for files in ratio_files.values():
        for handle in files.values():
            handle.close()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    return (model, history)


if __name__ == "__main__":
    model = PINN(hidden_dim=64, n_layers=2)
    train_pinn(model)
