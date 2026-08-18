import torch
import math

from pinn import DEVICE
from geometry import nozzle_radius, L_TOTAL

torch.set_default_dtype(torch.float32)


def centerline_points(n_points=500):
    """
    Generate points along the nozzle axis.
    """

    x = torch.linspace(0.0, L_TOTAL, n_points, device=DEVICE).view(-1, 1)
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)

    return torch.cat([x, y, z], dim=1)


def radial_slice(throat_ratio: float, x_location, n_radius=80, n_theta=120):
    """
    Generate one circular slice.
    """

    radius = nozzle_radius(
        torch.tensor([[x_location]], device=DEVICE), throat_ratio=throat_ratio
    ).item()

    r = torch.linspace(0.0, radius, n_radius, device=DEVICE)
    theta = torch.linspace(0.0, 2.0 * math.pi, n_theta, device=DEVICE)
    rr, tt = torch.meshgrid(r, theta, indexing="ij")
    xx = torch.full_like(rr, x_location)
    yy = rr * torch.cos(tt)
    zz = rr * torch.sin(tt)
    points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    return (points, rr, tt)


def longitudinal_slice(throat_ratio: float, theta_location=0.0, n_x=80, n_radius=120):
    """
    Generate a longitudinal slice at a fixed azimuthal angle.
    """

    theta = float(theta_location)
    x = torch.linspace(0.0, L_TOTAL, n_x, device=DEVICE)
    radius = nozzle_radius(x, throat_ratio=throat_ratio).view(-1)

    r = torch.linspace(0.0, 1.0, n_radius, device=DEVICE).unsqueeze(
        0
    ) * radius.unsqueeze(1)
    xx = x.unsqueeze(1).repeat(1, n_radius)
    yy = r * math.cos(theta)
    zz = r * math.sin(theta)

    points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    return (points, xx, r)


def nozzle_surface_points(throat_ratio: float, n_x=120, n_theta=180):
    """
    Sample the full nozzle sidewall surface while excluding the inlet and outlet
    end caps.

    Returns
    -------
    points : (N, 3) tensor
        Cartesian coordinates of the wall points.
    xx : (n_x, n_theta) tensor
        Axial coordinates.
    yy : (n_x, n_theta) tensor
        Radial/circumferential y coordinates.
    zz : (n_x, n_theta) tensor
        Radial/circumferential z coordinates.
    """

    x = torch.linspace(0.0, L_TOTAL, n_x, device=DEVICE)
    if n_x > 2:
        x = x[1:-1]

    theta = torch.linspace(0.0, 2.0 * math.pi, n_theta, device=DEVICE)
    xx, tt = torch.meshgrid(x, theta, indexing="ij")
    radius = nozzle_radius(xx, throat_ratio=throat_ratio)

    yy = radius * torch.cos(tt)
    zz = radius * torch.sin(tt)

    points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    return (points, xx, yy, zz)
