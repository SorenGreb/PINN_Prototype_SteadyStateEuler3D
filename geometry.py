import math
import torch

from flow_quantities import GAMMA

torch.set_default_dtype(torch.float32)

# --------------------------------------------------------------------------
# Geometry parameters
# --------------------------------------------------------------------------

L_CONV = 0.40
L_DIV = 0.80
R_INLET = 0.50
L_TOTAL = L_CONV + L_DIV
TARGET_EXIT_MACH = 2.0


def throat_radius(throat_ratio: float) -> float:
    return throat_ratio * R_INLET


def exit_radius(
    throat_radius: float, target_exit_mach: float = TARGET_EXIT_MACH
) -> float:
    """Return exit radius for a choked isentropic nozzle at target Mach."""

    if target_exit_mach < 1.0:
        raise ValueError("target_exit_mach must be at least 1 for a choked nozzle")

    mach_term = 1.0 + 0.5 * (GAMMA - 1.0) * target_exit_mach**2
    area_ratio = (1.0 / target_exit_mach) * (2.0 / (GAMMA + 1.0) * mach_term) ** (
        (GAMMA + 1.0) / (2.0 * (GAMMA - 1.0))
    )
    return throat_radius * math.sqrt(area_ratio)


def nozzle_radius(
    x: torch.Tensor,
    throat_ratio: torch.Tensor | float,
    target_exit_mach: float = TARGET_EXIT_MACH,
) -> torch.Tensor:

    if not torch.is_tensor(throat_ratio):
        throat_ratio = torch.full_like(x, float(throat_ratio))
    else:
        throat_ratio = throat_ratio.to(device=x.device, dtype=x.dtype)

    r_throat = throat_ratio * R_INLET
    r_exit = exit_radius(r_throat, target_exit_mach=target_exit_mach)

    r_conv = r_throat + 0.5 * (R_INLET - r_throat) * (
        1.0 + torch.cos(math.pi * x / L_CONV)
    )

    xd = x - L_CONV

    r_div = r_throat + 0.5 * (r_exit - r_throat) * (
        1.0 - torch.cos(math.pi * xd / L_DIV)
    )

    return torch.where(x <= L_CONV, r_conv, r_div)


def nozzle_radius_gradient(
    x: torch.Tensor,
    throat_ratio: torch.Tensor | float,
    target_exit_mach: float = TARGET_EXIT_MACH,
) -> torch.Tensor:

    if not torch.is_tensor(throat_ratio):
        throat_ratio = torch.full_like(x, float(throat_ratio))
    else:
        throat_ratio = throat_ratio.to(device=x.device, dtype=x.dtype)

    r_throat = throat_ratio * R_INLET
    r_exit = exit_radius(r_throat, target_exit_mach=target_exit_mach)

    dr_conv = (
        -0.5 * (R_INLET - r_throat) * math.pi / L_CONV * torch.sin(math.pi * x / L_CONV)
    )

    xd = x - L_CONV

    dr_div = (
        0.5 * (r_exit - r_throat) * math.pi / L_DIV * torch.sin(math.pi * xd / L_DIV)
    )

    return torch.where(x <= L_CONV, dr_conv, dr_div)


def wall_normal(
    x_wall: torch.Tensor,
    throat_ratio: float,
    target_exit_mach: float = TARGET_EXIT_MACH,
):
    """
    Compute outward unit normal vector on the nozzle wall.

    Parameters
    ----------
    x_wall : torch.Tensor
        Wall coordinates with shape (N, 3).
    throat_ratio : float
        Throat radius ratio.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Unit normal components (nx, ny, nz).
    """

    x = x_wall[:, 0:1]
    y = x_wall[:, 1:2]
    z = x_wall[:, 2:3]

    R = nozzle_radius(x, throat_ratio, target_exit_mach=target_exit_mach)
    dRdx = nozzle_radius_gradient(x, throat_ratio, target_exit_mach=target_exit_mach)

    nx = -R * dRdx
    ny = y
    nz = z

    norm = torch.sqrt(nx * nx + ny * ny + nz * nz)

    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    return (nx, ny, nz)
