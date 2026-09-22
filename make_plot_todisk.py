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
    cmap: str = "gist_stern",
    throat_ratio: float = 0.30,
):
    points, xx, r = longitudinal_slice(
        throat_ratio, theta_location, n_x=n_x, n_radius=n_radius
    )

    _, u, v, w, _ = predictor.predict(points, throat_ratio=throat_ratio)
    velocity = predictor.velocity_magnitude(u, v, w).view(n_x, n_radius).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor("gray")
    # vmin = float(np.nanmin(velocity))
    # vmax = float(np.nanmax(velocity))
    vmin = 0.0
    vmax = 2.0
    mesh = ax.pcolormesh(
        xx.cpu().numpy(),
        r.cpu().numpy(),
        velocity,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        f"Velocity magnitude in longitudinal slice at θ={theta_location} rad, "
        f"throat ratio={throat_ratio:.2f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("radial distance")
    # ax.set_xlim(float(xx.min().cpu().numpy()), float(xx.max().cpu().numpy()))
    # ax.set_ylim(float(r.min().cpu().numpy()), float(r.max().cpu().numpy()))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 0.55)
    fig.colorbar(mesh, ax=ax, label="Velocity magnitude")
    plt.grid()

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.show()


def plot_nozzle_surface_velocity(
    predictor: Predictor,
    output_path: Path,
    n_x: int = 120,
    n_theta: int = 180,
    cmap: str = "rainbow",
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
        edgecolor="black",
        linewidth=0.2,
        antialiased=True,
        shade=True,
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


def plot_centerline_quantity(quantity: str, throat_ratios=(0.2, 0.3, 0.4)):
    line_styles = [
        "-",
        "--",
        ":",
        "-.",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (3, 5, 1, 5)),
    ]
    line_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))

    for ratio_index, throat_ratio in enumerate(throat_ratios):
        file_name = f"results/centerline_{quantity}_tr{int(throat_ratio * 100):02d}.txt"
        try:
            with open(file_name, "r") as fo:
                lines = fo.readlines()
        except FileNotFoundError:
            continue

        if len(lines) < 2:
            continue

        x = [float(val) for val in lines[0].strip().split()]
        y = np.zeros((len(lines) - 1, len(x)))
        for row_index, line in enumerate(lines[1:]):
            if line.strip():
                y[row_index, :] = [float(val) for val in line.strip().split()]

        for point_index in range(len(x)):
            ax.plot(
                y[:, point_index],
                color=line_colors[point_index % len(line_colors)],
                linestyle=line_styles[ratio_index % len(line_styles)],
                linewidth=1.5,
                alpha=1.0,
                label=f"{quantity} tr={throat_ratio:.2f} @ x={x[point_index]:.2f}",
            )

    ax.set_xscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(quantity)
    ax.set_title(
        f"centerline {quantity} for throat ratios {', '.join(f'{r:.2f}' for r in throat_ratios)}"
    )
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results/centerline_{quantity}_all_ratios.pdf", dpi=200)
    plt.show()


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
    ax.set_xscale("log")
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

    output_path = output_dir / f"delaval_pinn_slice_{throat_ratio:.2f}.pdf"
    plot_longitudinal_velocity(predictor, output_path, throat_ratio=throat_ratio)

    """
    output_path = output_dir / f"delaval_pinn_surf_{throat_ratio:.2f}.png"
    plot_nozzle_surface_velocity(predictor, output_path, throat_ratio=throat_ratio)

    plot_centerline_quantity(quantity="velocity")
    plot_centerline_quantity(quantity="massflow")
    plot_centerline_quantity(quantity="rho")
    plot_centerline_quantity(quantity="p")
    plot_loss_history()
    """


if __name__ == "__main__":
    main()
