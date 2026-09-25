import torch

torch.set_default_dtype(torch.float32)

# --------------------------------------------------------------------------
# Physical constants and quantities
# --------------------------------------------------------------------------

GAMMA = 1.4


def velocity_magnitude(u, v, w):
    """
    Velocity magnitude.
    """

    return torch.sqrt(u * u + v * v + w * w)


def speed_of_sound(rho, p):
    """
    Local speed of sound.
    """

    return torch.sqrt(GAMMA * p / rho)


def mach_number(rho, u, v, w, p):
    """
    Compute local Mach number.
    """

    V = velocity_magnitude(u, v, w)
    a = speed_of_sound(rho, p)

    return V / a
