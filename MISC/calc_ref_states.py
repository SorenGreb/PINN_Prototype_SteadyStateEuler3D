import math

import torch

from GEOMETRY.geometry import L_CONV, nozzle_radius

gamma = 1.4
rho_in = 1.0
p_in = 0.85
R_in = 0.5


def calc_ref_states(throat_ratio, mach_out, x):
    """Calculate radius and isentropic flow state at longitudinal position x."""
    throat_radius = throat_ratio * R_in
    R_x = nozzle_radius(
        torch.tensor([x], dtype=torch.float64),
        throat_ratio,
        target_exit_mach=mach_out,
    ).item()

    def mach_from_area_ratio(area_ratio, supersonic):
        lower_mach, upper_mach = (1.0, mach_out) if supersonic else (0.0, 1.0)
        for _ in range(100):
            mach = 0.5 * (lower_mach + upper_mach)
            mach_area_ratio = (
                (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * mach**2)
            ) ** ((gamma + 1) / (2 * (gamma - 1))) / mach
            if (mach_area_ratio < area_ratio) == supersonic:
                lower_mach = mach
            else:
                upper_mach = mach
        return 0.5 * (lower_mach + upper_mach)

    mach_in = mach_from_area_ratio((R_in / throat_radius) ** 2, False)
    rho0 = rho_in * (1 + (gamma - 1) / 2 * mach_in**2) ** (1 / (gamma - 1))
    p0 = p_in * (1 + (gamma - 1) / 2 * mach_in**2) ** (gamma / (gamma - 1))
    mach_x = (
        1.0
        if x == L_CONV
        else mach_from_area_ratio(
            (R_x / throat_radius) ** 2,
            supersonic=x > L_CONV,
        )
    )
    rho_x = rho0 * (1 + (gamma - 1) / 2 * mach_x**2) ** (-1 / (gamma - 1))
    p_x = p0 * (1 + (gamma - 1) / 2 * mach_x**2) ** (-gamma / (gamma - 1))
    c_x = (gamma * p_x / rho_x) ** 0.5
    u_x = mach_x * c_x
    massflow = rho_x * u_x * math.pi * R_x**2

    return R_x, rho_x, u_x, p_x, c_x, mach_x, massflow


throat_ratio = 0.4
mach_out = 2.0
x = 6.0
R_x, rho_x, u_x, p_x, c_x, mach_x, massflow = calc_ref_states(throat_ratio, mach_out, x)

print("R_x     :", f"{R_x:.3f}")
print("rho_x   :", f"{rho_x:.3f}")
print("u_x     :", f"{u_x:.3f}")
print("p_x     :", f"{p_x:.3f}")
print("c_x     :", f"{c_x:.3f}")
print("mach_x  :", f"{mach_x:.3f}")
print("massflow:", f"{massflow:.3f}")
