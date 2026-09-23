import torch

from PINN.predictor import Predictor
from GEOMETRY.probes import radial_slice


def monitor_centerline_velocity_rho_p(
    predictor: Predictor,
    throat_ratio: float,
    x_centerline: torch.Tensor,
):
    rho, u, v, w, p = predictor.predict(x_centerline, throat_ratio=throat_ratio)

    velocity_magnitude = predictor.velocity_magnitude(u, v, w)

    return rho, velocity_magnitude, p


def monitor_crosssectional_massflow(
    predictor: Predictor,
    throat_ratio: float,
    x_centerline: torch.Tensor,
    n_radius: int = 30,
    n_theta: int = 45,
):
    mass_flow = []

    for x_location in x_centerline[:, 0]:
        points, radial_coordinates, angular_coordinates = radial_slice(
            throat_ratio=throat_ratio,
            x_location=x_location.item(),
            n_radius=n_radius,
            n_theta=n_theta,
        )
        rho, u, _, _, _ = predictor.predict(points, throat_ratio=throat_ratio)

        rho_u = (rho * u).reshape(radial_coordinates.shape)
        integrand = rho_u * radial_coordinates
        radial_integral = torch.trapezoid(integrand, radial_coordinates[:, 0], dim=0)
        slice_mass_flow = torch.trapezoid(
            radial_integral, angular_coordinates[0, :], dim=0
        )
        mass_flow.append(slice_mass_flow)

    rho_u_area = torch.stack(mass_flow).view(-1, 1)

    return rho_u_area
