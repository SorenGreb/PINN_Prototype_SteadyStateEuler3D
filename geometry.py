import math
import torch

from flow_quantities import GAMMA

torch.set_default_dtype(torch.float32)

# --------------------------------------------------------------------------
# Geometry parameters
# --------------------------------------------------------------------------

L_CONV = 2.0
L_DIV = 4.0
R_INLET = 0.50
L_TOTAL = L_CONV + L_DIV
TARGET_EXIT_MACH = 3.5


def throat_radius(throat_ratio: float) -> float:
    return throat_ratio * R_INLET


def inlet_speed_for_sonic_throat(throat_radius: float) -> float:
    """Return the subsonic inlet speed compatible with a choked throat.

    The inlet density and pressure are treated as static values from the
    boundary conditions. The returned speed makes the isentropic mass flow at
    the inlet equal to the choked mass flow through the supplied throat area.
    """

    from boundary_conditions import P_INLET, RHO_INLET

    if throat_radius <= 0.0 or throat_radius > R_INLET:
        raise ValueError("throat_radius must be greater than 0 and at most R_INLET")
    if RHO_INLET <= 0.0 or P_INLET <= 0.0:
        raise ValueError("RHO_INLET and P_INLET must be positive")

    area_ratio = (throat_radius / R_INLET) ** 2
    sound_speed = math.sqrt(GAMMA * P_INLET / RHO_INLET)
    critical_mass_flux_factor = (2.0 / (GAMMA + 1.0)) ** (
        (GAMMA + 1.0) / (2.0 * (GAMMA - 1.0))
    )
    mach_factor_exponent = (GAMMA + 1.0) / (2.0 * (GAMMA - 1.0))

    def mass_flow_residual(mach: float) -> float:
        pressure_factor = 1.0 + 0.5 * (GAMMA - 1.0) * mach**2
        choked_mass_flux = (
            critical_mass_flux_factor * pressure_factor**mach_factor_exponent
        )
        return mach - area_ratio * choked_mass_flux

    lower_mach = 0.0
    upper_mach = 1.0
    for _ in range(80):
        mach = 0.5 * (lower_mach + upper_mach)
        if mass_flow_residual(mach) > 0.0:
            upper_mach = mach
        else:
            lower_mach = mach

    return sound_speed * 0.5 * (lower_mach + upper_mach)


def sonic_throat_state(throat_radius: float) -> tuple[float, float, float]:
    """Return sonic throat density, axial speed, and pressure."""

    from boundary_conditions import P_INLET, RHO_INLET

    inlet_sound_speed = math.sqrt(GAMMA * P_INLET / RHO_INLET)
    inlet_speed = inlet_speed_for_sonic_throat(throat_radius)
    inlet_mach = inlet_speed / inlet_sound_speed
    stagnation_factor = 1.0 + 0.5 * (GAMMA - 1.0) * inlet_mach**2
    stagnation_density = RHO_INLET * stagnation_factor ** (1.0 / (GAMMA - 1.0))
    stagnation_pressure = P_INLET * stagnation_factor ** (GAMMA / (GAMMA - 1.0))
    critical_factor = 2.0 / (GAMMA + 1.0)
    throat_density = stagnation_density * critical_factor ** (1.0 / (GAMMA - 1.0))
    throat_pressure = stagnation_pressure * critical_factor ** (GAMMA / (GAMMA - 1.0))
    throat_speed = math.sqrt(GAMMA * throat_pressure / throat_density)
    return throat_density, throat_speed, throat_pressure


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
