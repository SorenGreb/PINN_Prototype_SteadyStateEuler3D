import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pinn import PINN, DEVICE
from probes import nozzle_surface_points, longitudinal_slice
from train import MODEL_PATH
from predictor import Predictor

torch.set_default_dtype(torch.float32)


def plot_longitudinal_velocity(
    predictor: Predictor,
    output_path: Path,
    theta_location: float = 0.0,
    n_x: int = 120,
    n_radius: int = 220,
    cmap: str = "viridis",
    throat_ratio: float = 0.30,
):
    points, xx, r = longitudinal_slice(
        throat_ratio, theta_location, n_x=n_x, n_radius=n_radius
    )

    _, u, v, w, _ = predictor.predict(points, throat_ratio=throat_ratio)
    velocity = predictor.velocity_magnitude(u, v, w).view(n_x, n_radius).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(
        xx.cpu().numpy(), r.cpu().numpy(), velocity, shading="auto", cmap=cmap
    )
    ax.set_title(
        f"Velocity magnitude in longitudinal slice at θ={theta_location} rad, "
        f"throat ratio={throat_ratio:.2f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("radial distance")
    fig.colorbar(mesh, ax=ax, label="Velocity magnitude")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.show()


def plot_nozzle_surface_velocity(
    predictor: Predictor,
    output_path: Path,
    n_x: int = 120,
    n_theta: int = 180,
    cmap: str = "hot",
    throat_ratio: float = 0.30,
):
    points, xx, yy, zz = nozzle_surface_points(
        throat_ratio=throat_ratio, n_x=n_x, n_theta=n_theta
    )

    _, u, v, w, _ = predictor.predict(points, throat_ratio=throat_ratio)
    velocity = predictor.velocity_magnitude(u, v, w)
    velocity = velocity.view(xx.shape[0], xx.shape[1]).cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(velocity.min(), velocity.max())
    facecolors = plt.get_cmap(cmap)(norm(velocity))

    mesh = ax.plot_surface(
        xx.cpu().numpy(),
        yy.cpu().numpy(),
        zz.cpu().numpy(),
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.set_title(
        f"Velocity magnitude on the nozzle surface (throat ratio={throat_ratio:.2f})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_facecolor("steelblue")
    ax.xaxis._axinfo["grid"]["color"] = "lightgray"
    ax.yaxis._axinfo["grid"]["color"] = "lightgray"
    ax.zaxis._axinfo["grid"]["color"] = "lightgray"
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label="Velocity magnitude",
        pad=0.08,
    )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.show()


def plot_centerline_quantity(quantity: str):
    fo = open(f"results/centerline_{quantity}.txt", "r")
    lines = fo.readlines()
    count = 0
    y = np.zeros((len(lines) - 1, len(lines[0].strip().split())))
    for line in lines:
        if count == 0:
            x = [float(val) for val in line.strip().split()]
        else:
            y[count - 1, :] = [float(val) for val in line.strip().split()]
        count += 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(x.__len__()):
        ax.plot(y[:, i], label=f"{quantity} at point {x[i]:.2f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(quantity)
    ax.set_title(f"centerline {quantity}")
    ax.legend()
    plt.grid()
    plt.savefig(f"results/centerline_{quantity}.png", dpi=200)
    plt.show()
    fo.close()


def plot_loss_history():
    fo = open(f"results/loss_history.txt", "r")
    lines = fo.readlines()
    count = 0
    y = np.zeros((len(lines) - 1, len(lines[0].strip().split())))
    for line in lines:
        if count == 0:
            x = [val for val in line.strip().split()]
        else:
            y[count - 1, :] = [float(val) for val in line.strip().split()]
        count += 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(x.__len__()):
        ax.plot(y[:, i], label=f"{x[i]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"loss history")
    ax.set_yscale("log")
    ax.legend()
    plt.grid()
    plt.savefig(f"results/loss_history.png", dpi=200)
    plt.show()
    fo.close()


def main():
    model = PINN(hidden_dim=64, n_layers=2).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    predictor = Predictor(model=model, device=DEVICE)

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    throat_ratio = 0.20

    output_path = output_dir / "delaval_pinn_slice.png"
    # plot_longitudinal_velocity(predictor, output_path, throat_ratio=throat_ratio)

    output_path = output_dir / f"delaval_pinn_surf_{throat_ratio:.2f}.png"
    plot_nozzle_surface_velocity(predictor, output_path, throat_ratio=throat_ratio)

    plot_centerline_quantity(quantity="velocity")
    plot_centerline_quantity(quantity="massflow")
    plot_centerline_quantity(quantity="rho")
    plot_centerline_quantity(quantity="p")
    plot_loss_history()


if __name__ == "__main__":
    main()
