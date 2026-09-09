from pathlib import Path
from torch import nn
import torch

from pinn import PINN, total_loss, DEVICE
from collocate import assemble_collocation_points
from predictor import Predictor
from monitor import monitor_crosssectional_massflow, monitor_centerline_velocity_rho_p
from probes import centerline_points

torch.set_default_dtype(torch.float32)

# --------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------

EPOCHS = 10000
LEARNING_RATE = 1.0e-3
MODEL_PATH = Path("model/model.pth")


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
    fo_velocity = open("results/centerline_velocity.txt", "w")
    fo_massflow = open("results/centerline_massflow.txt", "w")
    fo_rho = open("results/centerline_rho.txt", "w")
    fo_p = open("results/centerline_p.txt", "w")
    fo_loss = open("results/loss_history.txt", "w")

    for xi in x_centerline:
        fo_velocity.write(f"{xi[0].item():.6f} ")
    fo_velocity.write("\n")
    for xi in x_centerline:
        fo_massflow.write(f"{xi[0].item():.6f} ")
    fo_massflow.write("\n")
    for xi in x_centerline:
        fo_rho.write(f"{xi[0].item():.6f} ")
    fo_rho.write("\n")
    for xi in x_centerline:
        fo_p.write(f"{xi[0].item():.6f} ")
    fo_p.write("\n")
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
            rho, velocity_magnitude, p = monitor_centerline_velocity_rho_p(
                predictor, throat_ratio=0.3, x_centerline=x_centerline
            )
            for vi in velocity_magnitude:
                fo_velocity.write(f"{vi[0].item():.6f} ")
            fo_velocity.write("\n")
            fo_velocity.flush()

            massflow = monitor_crosssectional_massflow(
                predictor, throat_ratio=0.3, x_centerline=x_centerline
            )
            for value in massflow:
                fo_massflow.write(f"{value[0].item():.6f} ")
            fo_massflow.write("\n")
            fo_massflow.flush()

            for value in rho:
                fo_rho.write(f"{value[0].item():.6f} ")
            fo_rho.write("\n")
            fo_rho.flush()

            for value in p:
                fo_p.write(f"{value[0].item():.6f} ")
            fo_p.write("\n")
            fo_p.flush()

            for key in history:
                fo_loss.write(f"{history[key][-1]:.6e} ")
            fo_loss.write("\n")
            fo_loss.flush()

    fo_velocity.close()
    fo_massflow.close()
    fo_rho.close()
    fo_p.close()
    fo_loss.close()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    return (model, history)


if __name__ == "__main__":
    model = PINN(hidden_dim=64, n_layers=2)
    train_pinn(model)
