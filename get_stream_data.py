import torch
from pathlib import Path
import matplotlib.pyplot as plt

from pinn import DEVICE
from geometry import nozzle_radius, L_TOTAL
from probes import nozzle_surface_points
from predictor import Predictor


from dataclasses import dataclass
import math
import torch

torch.set_default_dtype(torch.float32)


@dataclass
class SurfaceParticle:
    x: float
    theta: float


class SurfaceParticleSystem:
    """
    Particles advected along an axisymmetric nozzle surface.

    Each particle is represented by:
        x     : axial position
        theta : azimuthal position

    The Cartesian position is reconstructed from the nozzle radius:
        y = R(x) cos(theta)
        z = R(x) sin(theta)
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        injection_rate: float,
        throat_ratio: float = 0.30,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.injection_rate = injection_rate
        self.throat_ratio = float(throat_ratio)

        self.particles = []

        # Keeps fractional particles between timesteps.
        self._injection_remainder = 0.0

    def inject(self, dt: float):
        """
        Inject particles continuously at the inlet.
        injection_rate is particles / second.
        """

        self._injection_remainder += self.injection_rate * dt
        n_new = int(self._injection_remainder)
        self._injection_remainder -= n_new

        for _ in range(n_new):
            theta = 2.0 * math.pi * torch.rand(1).item()
            self.particles.append(SurfaceParticle(x=self.x_min, theta=theta))

    def advect(self, velocity_magnitude: torch.Tensor, dt: float):
        """
        Advect particles along the nozzle surface.

        Parameters
        ----------
        velocity_magnitude : Tensor of shape (n_x, n_theta).
        dt : Physical timestep.

        The particle travels ds = velocity * dt along the surface.
        """

        if not self.particles:
            return

        n_x, n_theta = velocity_magnitude.shape

        # Axial grid corresponding to nozzle_surface_points().
        x_grid = torch.linspace(
            self.x_min,
            self.x_max,
            n_x + 2,
            device=velocity_magnitude.device,
        )[1:-1]

        x_grid = x_grid.cpu()
        velocity_cpu = velocity_magnitude.detach().cpu()
        updated_particles = []

        for particle in self.particles:
            x = particle.x
            theta = particle.theta

            # --------------------------------------------------
            # Find axial interpolation position
            # --------------------------------------------------

            if x <= float(x_grid[0]):
                i0 = 0
                i1 = 1
                alpha = 0.0

            elif x >= float(x_grid[-1]):
                i0 = n_x - 2
                i1 = n_x - 1
                alpha = 1.0

            else:
                i1 = int(torch.searchsorted(x_grid, x))
                i0 = i1 - 1

                x0 = float(x_grid[i0])
                x1 = float(x_grid[i1])

                alpha = (x - x0) / (x1 - x0)

            # --------------------------------------------------
            # Find theta interpolation position
            # --------------------------------------------------

            # theta grid is 0 ... 2π.
            theta_position = theta / (2.0 * math.pi) * (n_theta - 1)

            j0 = int(math.floor(theta_position))
            j1 = min(j0 + 1, n_theta - 1)

            beta = theta_position - j0

            # --------------------------------------------------
            # Bilinear interpolation of velocity
            # --------------------------------------------------

            v00 = float(velocity_cpu[i0, j0])
            v01 = float(velocity_cpu[i0, j1])
            v10 = float(velocity_cpu[i1, j0])
            v11 = float(velocity_cpu[i1, j1])

            v0 = v00 * (1.0 - beta) + v01 * beta
            v1 = v10 * (1.0 - beta) + v11 * beta

            local_velocity = v0 * (1.0 - alpha) + v1 * alpha

            # --------------------------------------------------
            # Convert surface distance to axial displacement
            # --------------------------------------------------

            # Numerical derivative of nozzle radius.
            dx_eps = 1e-5

            r_plus = float(
                nozzle_radius(
                    torch.tensor(x + dx_eps, device=DEVICE),
                    throat_ratio=self.throat_ratio,
                )
            )
            r_minus = float(
                nozzle_radius(
                    torch.tensor(x - dx_eps, device=DEVICE),
                    throat_ratio=self.throat_ratio,
                )
            )

            dRdx = (r_plus - r_minus) / (2.0 * dx_eps)

            # Distance travelled along the surface.
            ds = local_velocity * dt

            # Surface metric:
            #
            # ds² = dx² + dR²
            #     = (1 + (dR/dx)²) dx²
            #
            dx = ds / math.sqrt(1.0 + dRdx**2)
            particle.x += dx
            updated_particles.append(particle)

        self.particles = updated_particles

    def remove_outside_domain(self):
        """
        Remove particles that have left the nozzle domain.
        """

        self.particles = [
            particle
            for particle in self.particles
            if self.x_min <= particle.x < self.x_max
        ]

    def positions(self):
        """
        Return Cartesian particle positions.

        Returns
        -------
        torch.Tensor
            Shape (N, 3), containing x, y, z.
        """

        positions = []

        for particle in self.particles:

            x = particle.x
            theta = particle.theta

            radius = float(
                nozzle_radius(
                    torch.tensor(x, device=DEVICE), throat_ratio=self.throat_ratio
                )
            )

            y = radius * math.cos(theta)
            z = radius * math.sin(theta)

            positions.append([x, y, z])

        if not positions:
            return torch.empty((0, 3), dtype=torch.float32)

        return torch.tensor(positions, dtype=torch.float32)


particle_system = SurfaceParticleSystem(
    x_min=0.0, x_max=L_TOTAL, injection_rate=5.0  # particles / second
)


def generate_surface_velocity_data(
    predictor: Predictor, n_x: int = 50, n_theta: int = 60, throat_ratio: float = 0.30
):
    global particle_system

    if particle_system.throat_ratio != throat_ratio:
        particle_system = SurfaceParticleSystem(
            x_min=0.0, x_max=L_TOTAL, injection_rate=5.0, throat_ratio=throat_ratio
        )

    points, xx, yy, zz = nozzle_surface_points(
        throat_ratio=throat_ratio, n_x=n_x, n_theta=n_theta
    )

    _, u, v, w, _ = predictor.predict(points, throat_ratio=throat_ratio)
    velocity = predictor.velocity_magnitude(u, v, w)
    velocity = velocity.view(xx.shape[0], xx.shape[1]).cpu()
    dt = 0.5
    particle_system.inject(dt)
    particle_system.advect(velocity_magnitude=velocity, dt=dt)
    particle_system.remove_outside_domain()

    return {
        "geometry": {"x": xx.tolist(), "y": yy.tolist(), "z": zz.tolist()},
        "velocity": {"magnitude": velocity.tolist()},
        "particles": {"positions": particle_system.positions().tolist()},
    }
