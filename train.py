from pathlib import Path
from torch import nn
import torch

from pinn import PINN, total_loss, DEVICE
from collocate import assemble_collocation_points

torch.set_default_dtype(torch.float32)

# --------------------------------------------------------------------------
# Training parameters
# --------------------------------------------------------------------------

EPOCHS = 2000
LEARNING_RATE = 1e-3
MODEL_PATH = Path("model/model.pth")


def train_pinn(model: nn.Module, epochs: int = EPOCHS):
    """
    Train the Physics-Informed Neural Network.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    history = {"loss": [], "pde": [], "wall": [], "inlet": [], "outlet": []}
    best_loss = float("inf")

    for epoch in range(epochs):

        # --------------------------------------------------------------
        # Resample collocation points
        # --------------------------------------------------------------

        x_interior, x_wall, x_inlet, x_outlet, sampled_throat_ratios = (
            assemble_collocation_points()
        )
        optimizer.zero_grad()

        loss, loss_pde, loss_wall, loss_inlet, loss_outlet = total_loss(
            model, x_interior, x_wall, x_inlet, x_outlet
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())
        history["pde"].append(loss_pde.item())
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
                f"Wall {loss_wall.item():10.4e} | "
                f"Inlet {loss_inlet.item():10.4e} | "
                f"Outlet {loss_outlet.item():10.4e} | "
                f"LR {lr:.2e}"
            )
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    return (model, history)


if __name__ == "__main__":
    model = PINN(hidden_dim=64, n_layers=4)
    train_pinn(model)
