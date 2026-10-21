import math
import torch

from PINN.pinn import DEVICE
from GEOMETRY.geometry import (
    MACH_MAX,
    MACH_MIN,
    nozzle_radius,
    throat_radius,
    exit_radius,
)
from GEOMETRY.geometry import L_TOTAL, R_INLET

torch.set_default_dtype(torch.float32)


THROAT_RATIO_MIN = 0.2
THROAT_RATIO_MAX = 0.4
N_MACH_VALUES = 8

N_INTERIOR = 5000
N_WALL = 600
N_INLET = 300
N_OUTLET = 300


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
    mach_number: float,
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
    mach = torch.full((n_points, 1), float(mach_number), device=DEVICE)
    return torch.cat([points, ratio, mach], dim=1)


# ================================================
# COLLOCATION FUNCTIONS
# ================================================


def collocate_interior(
    n_points: int,
    throat_ratio: float,
    mach_number: float,
) -> torch.Tensor:
    """
    Uniformly sample the nozzle volume.

    Returns
    -------
    Tensor of shape (N,5): x, y, z, throat ratio, exit Mach.
    """

    x = torch.rand(n_points, 1, device=DEVICE) * L_TOTAL
    radius = nozzle_radius(x, throat_ratio, target_exit_mach=mach_number)
    return _sample_circular_points(
        n_points, radius, x, throat_ratio=throat_ratio,
        mach_number=mach_number, interior=True
    )


def collocate_wall(
    n_points: int,
    throat_ratio: float,
    mach_number: float,
) -> torch.Tensor:
    """
    Sample points on the nozzle wall.
    """

    x = torch.rand(n_points, 1, device=DEVICE) * L_TOTAL
    radius = nozzle_radius(x, throat_ratio, target_exit_mach=mach_number)
    return _sample_circular_points(
        n_points, radius, x, throat_ratio=throat_ratio,
        mach_number=mach_number, interior=False
    )


def collocate_inlet(
    n_points: int,
    throat_ratio: float,
    mach_number: float,
) -> torch.Tensor:
    """
    Uniform inlet disk.
    """

    radius = torch.sqrt(torch.rand(n_points, 1, device=DEVICE)) * R_INLET
    x = torch.zeros_like(radius)
    return _sample_circular_points(
        n_points, radius, x, throat_ratio=throat_ratio,
        mach_number=mach_number, interior=False
    )


def collocate_outlet(
    n_points: int,
    throat_ratio: float,
    mach_number: float,
) -> torch.Tensor:
    """
    Uniform outlet disk.
    """

    r_throat = throat_radius(throat_ratio)
    r_exit = exit_radius(r_throat, target_exit_mach=mach_number)
    radius = torch.sqrt(torch.rand(n_points, 1, device=DEVICE)) * r_exit
    x = torch.full_like(radius, L_TOTAL)

    return _sample_circular_points(
        n_points, radius, x, throat_ratio=throat_ratio,
        mach_number=mach_number, interior=False
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


def collocate_mach_numbers(
    n_values: int,
    min_mach: float = MACH_MIN,
    max_mach: float = MACH_MAX,
) -> torch.Tensor:
    """Create a discrete set of exit Mach numbers spanning the training range."""

    return torch.linspace(
        float(min_mach),
        float(max_mach),
        steps=n_values,
        device=DEVICE,
        dtype=torch.float32,
    ).view(-1, 1)


# ================================================
# ASSEMBLE
# ================================================


def assemble_collocation_points(
    throat_ratios: torch.Tensor | None = None,
    mach_numbers: torch.Tensor | None = None,
):
    """
    Generate collocation point sets for a sampled family of throat ratios.

    Returns
    -------
    x_interior, x_wall, x_inlet, x_outlet, sampled_throat_ratios,
    sampled_mach_numbers
    """

    if throat_ratios is None:
        throat_ratios = collocate_throat_ratios(
            n_values=8, min_ratio=THROAT_RATIO_MIN, max_ratio=THROAT_RATIO_MAX
        )

    throat_ratios = torch.as_tensor(
        throat_ratios, device=DEVICE, dtype=torch.float32
    ).view(-1)

    if mach_numbers is None:
        mach_numbers = collocate_mach_numbers(n_values=N_MACH_VALUES)

    mach_numbers = torch.as_tensor(
        mach_numbers, device=DEVICE, dtype=torch.float32
    ).view(-1)
    if mach_numbers.numel() == 0 or not torch.isfinite(mach_numbers).all():
        raise ValueError("mach_numbers must contain finite values")
    if torch.any((mach_numbers < MACH_MIN) | (mach_numbers > MACH_MAX)):
        raise ValueError(f"mach_numbers must be between {MACH_MIN} and {MACH_MAX}")

    x_interior_list = []
    x_wall_list = []
    x_inlet_list = []
    x_outlet_list = []

    interior_per_mach = math.ceil(N_INTERIOR / mach_numbers.numel())
    wall_per_mach = math.ceil(N_WALL / mach_numbers.numel())
    inlet_per_mach = math.ceil(N_INLET / mach_numbers.numel())
    outlet_per_mach = math.ceil(N_OUTLET / mach_numbers.numel())

    for ratio in throat_ratios:
        for mach in mach_numbers:
            ratio_value = float(ratio)
            mach_value = float(mach)
            x_interior_list.append(
                collocate_interior(interior_per_mach, ratio_value, mach_value)
            )
            x_wall_list.append(collocate_wall(wall_per_mach, ratio_value, mach_value))
            x_inlet_list.append(collocate_inlet(inlet_per_mach, ratio_value, mach_value))
            x_outlet_list.append(
                collocate_outlet(outlet_per_mach, ratio_value, mach_value)
            )

    return (
        torch.cat(x_interior_list, dim=0),
        torch.cat(x_wall_list, dim=0),
        torch.cat(x_inlet_list, dim=0),
        torch.cat(x_outlet_list, dim=0),
        throat_ratios,
        mach_numbers,
    )
