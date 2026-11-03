import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from PINN.pinn import PINN, DEVICE


from PINN.predictor import Predictor
from GEOMETRY.probes import nozzle_surface_points, longitudinal_slice
from GEOMETRY.geometry import L_CONV, L_TOTAL, MACH_MAX, MACH_MIN
from TRAIN.train import MODEL_PATH
from DIRS.dirs import RESULTS_DIR
from MISC.calc_errors import calculate_errors

torch.set_default_dtype(torch.float32)

QUANTITY_LABELS = {
    "velocity": "Velocity magnitude",
    "massflow": "Mass flow",
    "rho": "Density",
    "p": "Pressure",
}
LINE_STYLES = [
    "-",
    "--",
    ":",
    "-.",
    (0, (3, 1, 1, 1)),
    (0, (5, 1)),
    (0, (3, 5, 1, 5)),
]
MACH_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINE_COLORS = [
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


def plot_longitudinal_velocity(
    predictor: Predictor,
    output_path: Path,
    theta_location: float = 0.0,
    n_x: int = 120,
    n_radius: int = 220,
    cmap: str = "gist_stern",
    throat_ratio: float = 0.30,
    *,
    mach_number: float,
):
    points, xx, r = longitudinal_slice(
        throat_ratio,
        theta_location,
        n_x=n_x,
        n_radius=n_radius,
        mach_number=mach_number,
    )

    _, u, v, w, _ = predictor.predict(
        points, throat_ratio=throat_ratio, mach_number=mach_number
    )
    velocity = predictor.velocity_magnitude(u, v, w).view(n_x, n_radius).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor("gray")
    # vmin = float(np.nanmin(velocity))
    # vmax = float(np.nanmax(velocity))
    vmin = 0.0
    vmax = 2.0
    # pcolormesh expects the X and Y coordinates to be monotonic in the mesh axes.
    # Here the longitudinal grid is naturally laid out with x varying along columns
    # and radial distance varying along rows, so transpose the arrays to match.
    mesh = ax.pcolormesh(
        xx.T.cpu().numpy(),
        r.T.cpu().numpy(),
        velocity.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        f"Velocity magnitude in longitudinal slice at θ={theta_location} rad, "
        f"throat ratio={throat_ratio:.2f}, exit Mach={mach_number:.2f}"
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


def plot_nozzle_surface_velocity(
    predictor: Predictor,
    output_path: Path,
    n_x: int = 120,
    n_theta: int = 180,
    cmap: str = "rainbow",
    throat_ratio: float = 0.30,
    *,
    mach_number: float,
):
    points, xx, yy, zz = nozzle_surface_points(
        throat_ratio=throat_ratio,
        n_x=n_x,
        n_theta=n_theta,
        mach_number=mach_number,
    )

    _, u, v, w, _ = predictor.predict(
        points, throat_ratio=throat_ratio, mach_number=mach_number
    )
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
        f"Velocity magnitude on the nozzle surface (throat ratio={throat_ratio:.2f}, "
        f"exit Mach={mach_number:.2f})"
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


def plot_centerline_quantity(
    quantity: str, throat_ratios=(0.2, 0.4), mach_numbers=(2.0, 4.0)
):
    fig, ax = plt.subplots(figsize=(10, 5))

    for ratio_index, throat_ratio in enumerate(throat_ratios):
        for mach_index, mach_number in enumerate(mach_numbers):
            file_name = (
                f"{RESULTS_DIR}/centerline_{quantity}_"
                f"tr{throat_ratio:.2f}_ma{mach_number:.2f}.txt"
            )
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
                    color=LINE_COLORS[point_index % len(LINE_COLORS)],
                    linestyle=LINE_STYLES[ratio_index % len(LINE_STYLES)],
                    marker=MACH_MARKERS[mach_index % len(MACH_MARKERS)],
                    markevery=max(1, len(y) // 12),
                    markersize=4,
                    linewidth=1.0,
                    alpha=0.8,
                    label=(
                        f"{quantity} tr={throat_ratio:.2f} "
                        f"M={mach_number:.2f} @ x={x[point_index]:.2f}"
                    ),
                )

    ax.set_xscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(quantity)
    ax.set_title(
        f"centerline {quantity} for throat ratios "
        f"{', '.join(f'{r:.2f}' for r in throat_ratios)} and exit Mach numbers "
        f"{', '.join(f'{mach:.2f}' for mach in mach_numbers)}"
    )
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/centerline_{quantity}_all_ratios.pdf", dpi=200)


def plot_loss_history():
    fo = open(f"{RESULTS_DIR}/loss_history.txt", "r")
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
    plt.savefig(f"{RESULTS_DIR}/loss_history.pdf", dpi=200)

    fo.close()


def plot_location_errors(
    errors: list[tuple[float, float, str, float, list[tuple[float, float]]]],
    output_path: Path,
) -> tuple[float, float]:
    """Save pointwise relative errors for all quantities and cases as one PDF."""
    maximum_error_percent = max(
        error * 100
        for _, _, _, _, location_errors in errors
        for _, error in location_errors
    )
    y_axis_max = maximum_error_percent * 1.1 if maximum_error_percent else 1.0
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes_by_quantity = {
        quantity: axes.flat[index] for index, quantity in enumerate(QUANTITY_LABELS)
    }
    throat_ratios = sorted({throat_ratio for throat_ratio, *_ in errors})
    mach_numbers = sorted({mach_number for _, mach_number, *_ in errors})

    for throat_ratio, mach_number, quantity, _, location_errors in errors:
        x_values, relative_errors = zip(*location_errors)
        axes_by_quantity[quantity].plot(
            x_values,
            [error * 100 for error in relative_errors],
            color="black",
            linestyle=LINE_STYLES[throat_ratios.index(throat_ratio) % len(LINE_STYLES)],
            marker=MACH_MARKERS[mach_numbers.index(mach_number) % len(MACH_MARKERS)],
            linewidth=1.5,
            label=f"tr={throat_ratio:.2f}, M={mach_number:.2f}",
        )

    legend_handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    for quantity, axis in axes_by_quantity.items():
        axis.set_title(QUANTITY_LABELS[quantity])
        axis.set_ylabel("Relative error (%)")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0, y_axis_max)
        for location, label, alignment in (
            (0.0, "Inlet", "left"),
            (L_CONV, "Throat", "center"),
            (L_TOTAL, "Outlet", "right"),
        ):
            axis.axvline(location, color="black", linestyle=":", linewidth=0.9)
            axis.text(
                location,
                0.98,
                label,
                transform=axis.get_xaxis_transform(),
                horizontalalignment=alignment,
                verticalalignment="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    for axis in axes[-1, :]:
        axis.set_xlabel("Axial location, x")

    figure.suptitle("Final centerline relative error by location")
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=len(legend_labels),
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.95))
    figure.savefig(output_path, format="pdf")
    plt.close(figure)
    return maximum_error_percent, y_axis_max


def plot_centerline_relative_errors():
    errors = calculate_errors(RESULTS_DIR)
    if not errors:
        raise FileNotFoundError(f"No centerline result files found in {RESULTS_DIR}")

    output_path = RESULTS_DIR / "centerline_relative_errors.pdf"
    _, _ = plot_location_errors(errors, output_path)


def main():
    model = PINN(hidden_dim=64, n_layers=2).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    predictor = Predictor(model=model, device=DEVICE)

    RESULTS_DIR.mkdir(exist_ok=True)

    throat_ratio = 0.40
    mach_number = (MACH_MIN + MACH_MAX) / 2.0

    output_path = RESULTS_DIR / f"delaval_pinn_slice_{throat_ratio:.2f}.pdf"
    plot_longitudinal_velocity(
        predictor, output_path, throat_ratio=throat_ratio, mach_number=mach_number
    )

    output_path = RESULTS_DIR / f"delaval_pinn_surf_{throat_ratio:.2f}.pdf"
    plot_nozzle_surface_velocity(
        predictor, output_path, throat_ratio=throat_ratio, mach_number=mach_number
    )

    plot_centerline_quantity(quantity="velocity")
    plot_centerline_quantity(quantity="massflow")
    plot_centerline_quantity(quantity="rho")
    plot_centerline_quantity(quantity="p")
    plot_loss_history()
    plot_centerline_relative_errors()


if __name__ == "__main__":
    main()
