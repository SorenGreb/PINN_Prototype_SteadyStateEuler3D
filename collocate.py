import math
import torch

from pinn import DEVICE
from geometry import (
    TARGET_EXIT_MACH,
    nozzle_radius,
    throat_radius,
    exit_radius,
)
from geometry import L_CONV, L_TOTAL, R_INLET

torch.set_default_dtype(torch.float32)


THROAT_RATIO_MIN = 0.2
THROAT_RATIO_MAX = 0.4

N_INTERIOR = 5000
N_WALL = 600
N_INLET = 300
N_OUTLET = 300

THROAT_FOCUS_FRACTION = 0.5
THROAT_FOCUS_HALF_WIDTH = 0.5


# ================================================
# HELPERS
# ================================================


def _random_angles(
    n_points: int,
) -> torch.Tensor:
    return 2.0 * math.pi * torch.rand(n_points, 1, device=DEVICE)


def _stack_points(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, y, z], dim=1)


def _sample_circular_points(
    n_points: int,
    radius: torch.Tensor,
    x: torch.Tensor,
    throat_ratio: float,
    interior: bool = False,
) -> torch.Tensor:
    theta = _random_angles(n_points)

    r = radius
    if interior:
        r = torch.sqrt(torch.rand(n_points, 1, device=DEVICE)) * radius

    y = r * torch.cos(theta)
    z = r * torch.sin(theta)
    points = _stack_points(x, y, z)
    ratio = torch.full((n_points, 1), float(throat_ratio), device=DEVICE)
    return torch.cat([points, ratio], dim=1)


# ================================================
# COLLOCATION FUNCTIONS
# ================================================


def collocate_interior(
    n_points: int,
    throat_ratio: float,
) -> torch.Tensor:
    """
    Sample the nozzle volume with extra axial density around the throat.

    Returns
    -------
    Tensor of shape (N,4)
    """

    n_focused = int(n_points * THROAT_FOCUS_FRACTION)
    n_global = n_points - n_focused

    x_global = torch.rand(n_global, 1, device=DEVICE) * L_TOTAL
    throat_x = L_CONV
    x_focused = (
        throat_x
        + (2.0 * torch.rand(n_focused, 1, device=DEVICE) - 1.0)
        * THROAT_FOCUS_HALF_WIDTH
    )
    x = torch.cat([x_global, x_focused], dim=0)

    radius = nozzle_radius(x, throat_ratio)
    return _sample_circular_points(
        n_points, radius, x, throat_ratio=throat_ratio, interior=True
    )


def collocate_wall(
    n_points: int,
    throat_ratio: float,
) -> torch.Tensor:
    """
    Sample points on the nozzle wall.
    """

    x = torch.rand(n_points, 1, device=DEVICE) * L_TOTAL
    radius = nozzle_radius(x, throat_ratio)
    return _sample_circular_points(
        n_points, radius, x, throat_ratio=throat_ratio, interior=False
    )


def collocate_inlet(
    n_points: int,
    throat_ratio: float,
) -> torch.Tensor:
    """
    Uniform inlet disk.
    """

    radius = torch.sqrt(torch.rand(n_points, 1, device=DEVICE)) * R_INLET
    x = torch.zeros_like(radius)
    return _sample_circular_points(
        n_points, radius, x, throat_ratio=throat_ratio, interior=False
    )


def collocate_outlet(
    n_points: int,
    throat_ratio: float,
) -> torch.Tensor:
    """
    Uniform outlet disk.
    """

    r_throat = throat_radius(throat_ratio)
    r_exit = exit_radius(r_throat, target_exit_mach=TARGET_EXIT_MACH)
    radius = torch.sqrt(torch.rand(n_points, 1, device=DEVICE)) * r_exit
    x = torch.full_like(radius, L_TOTAL)

    return _sample_circular_points(
        n_points, radius, x, throat_ratio=throat_ratio, interior=False
    )


def collocate_throat_ratios(
    n_values: int,
    min_ratio: float,
    max_ratio: float,
) -> torch.Tensor:
    """
    Create a discrete set of throat ratios spanning the parameter range.
    """

    return torch.linspace(
        float(min_ratio),
        float(max_ratio),
        steps=n_values,
        device=DEVICE,
        dtype=torch.float32,
    ).view(-1, 1)


# ================================================
# ASSEMBLE
# ================================================


def assemble_collocation_points(throat_ratios: torch.Tensor | None = None):
    """
    Generate collocation point sets for a sampled family of throat ratios.

    Returns
    -------
    x_interior, x_wall, x_inlet, x_outlet, sampled_throat_ratios
    """

    if throat_ratios is None:
        throat_ratios = collocate_throat_ratios(
            n_values=8, min_ratio=THROAT_RATIO_MIN, max_ratio=THROAT_RATIO_MAX
        )

    throat_ratios = torch.as_tensor(
        throat_ratios, device=DEVICE, dtype=torch.float32
    ).view(-1)

    x_interior_list = []
    x_wall_list = []
    x_inlet_list = []
    x_outlet_list = []

    for ratio in throat_ratios:
        x_interior_list.append(
            collocate_interior(N_INTERIOR, throat_ratio=float(ratio))
        )
        x_wall_list.append(collocate_wall(N_WALL, throat_ratio=float(ratio)))
        x_inlet_list.append(collocate_inlet(N_INLET, throat_ratio=float(ratio)))
        x_outlet_list.append(collocate_outlet(N_OUTLET, throat_ratio=float(ratio)))

    return (
        torch.cat(x_interior_list, dim=0),
        torch.cat(x_wall_list, dim=0),
        torch.cat(x_inlet_list, dim=0),
        torch.cat(x_outlet_list, dim=0),
        throat_ratios,
    )
