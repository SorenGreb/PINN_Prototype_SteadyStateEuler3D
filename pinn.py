import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
from geometry import nozzle_radius, nozzle_radius_gradient

from derivatives import gradient, divergence
from boundary_conditions import (
    RHO_INLET,
    U_INLET,
    V_INLET,
    W_INLET,
    P_INLET,
    P_OUTLET,
    FIXED_OUTLET_P_BC,
)
from geometry import wall_normal
from flow_quantities import GAMMA

# ==========================================================================
# Physics-Informed Neural Network
# ==========================================================================


class PINN(nn.Module):
    """
    Fully-connected neural network representing the flow field.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()

        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]

        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])

        layers.append(nn.Linear(hidden_dim, 5))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==========================================================================
# Primitive variables
# ==========================================================================


def primitive_variables(model: nn.Module, x: torch.Tensor):
    """
    Evaluate the neural network.

    Returns
    -------
    rho,u,v,w,p
    """

    q = model(x)

    rho = q[:, 0:1]
    u = q[:, 1:2]
    v = q[:, 2:3]
    w = q[:, 3:4]
    p = q[:, 4:5]

    return (rho, u, v, w, p)


# ==========================================================================
# Thermodynamics
# ==========================================================================


def total_energy(rho, u, v, w, p):
    """
    Total energy per unit volume.
    """

    kinetic = 0.5 * rho * (u**2 + v**2 + w**2)
    internal = p / (GAMMA - 1.0)

    return kinetic + internal


# ==========================================================================
# Primitive variable gradients
# ==========================================================================


def primitive_gradients(rho, u, v, w, p, x):
    """
    Compute all first derivatives.
    """

    rho_x, rho_y, rho_z = gradient(rho, x)
    u_x, u_y, u_z = gradient(u, x)
    v_x, v_y, v_z = gradient(v, x)
    w_x, w_y, w_z = gradient(w, x)
    p_x, p_y, p_z = gradient(p, x)

    return (
        rho_x,
        rho_y,
        rho_z,
        u_x,
        u_y,
        u_z,
        v_x,
        v_y,
        v_z,
        w_x,
        w_y,
        w_z,
        p_x,
        p_y,
        p_z,
    )


# ==========================================================================
# Conservative fluxes
# ==========================================================================


def conservative_fluxes(rho, u, v, w, p):
    """
    Compute all Euler flux vectors.
    """

    E = total_energy(rho, u, v, w, p)

    # ----------------------------------------------------------
    # Mass
    # ----------------------------------------------------------

    Fm_x = rho * u
    Fm_y = rho * v
    Fm_z = rho * w

    # ----------------------------------------------------------
    # Momentum x
    # ----------------------------------------------------------

    Fx_x = rho * u * u + p
    Fx_y = rho * u * v
    Fx_z = rho * u * w

    # ----------------------------------------------------------
    # Momentum y
    # ----------------------------------------------------------

    Fy_x = rho * v * u
    Fy_y = rho * v * v + p
    Fy_z = rho * v * w

    # ----------------------------------------------------------
    # Momentum z
    # ----------------------------------------------------------

    Fz_x = rho * w * u
    Fz_y = rho * w * v
    Fz_z = rho * w * w + p

    # ----------------------------------------------------------
    # Energy
    # ----------------------------------------------------------

    Fe_x = (E + p) * u
    Fe_y = (E + p) * v
    Fe_z = (E + p) * w

    return (
        Fm_x,
        Fm_y,
        Fm_z,
        Fx_x,
        Fx_y,
        Fx_z,
        Fy_x,
        Fy_y,
        Fy_z,
        Fz_x,
        Fz_y,
        Fz_z,
        Fe_x,
        Fe_y,
        Fe_z,
    )


# ==========================================================================
# Euler residual
# ==========================================================================


def euler_residual(
    model: nn.Module,
    x: torch.Tensor,
):
    """
    Evaluate the steady compressible Euler equations.

    Returns
    -------
    continuity residual
    x-momentum residual
    y-momentum residual
    z-momentum residual
    energy residual
    """

    x.requires_grad_(True)

    # ----------------------------------------------------------
    # Primitive variables
    # ----------------------------------------------------------

    rho, u, v, w, p = primitive_variables(model, x)

    # ----------------------------------------------------------
    # Conservative fluxes
    # ----------------------------------------------------------

    (
        Fm_x,
        Fm_y,
        Fm_z,
        Fx_x,
        Fx_y,
        Fx_z,
        Fy_x,
        Fy_y,
        Fy_z,
        Fz_x,
        Fz_y,
        Fz_z,
        Fe_x,
        Fe_y,
        Fe_z,
    ) = conservative_fluxes(rho, u, v, w, p)

    # ----------------------------------------------------------
    # Governing equations
    # ----------------------------------------------------------

    continuity = divergence(Fm_x, Fm_y, Fm_z, x)

    momentum_x = divergence(Fx_x, Fx_y, Fx_z, x)
    momentum_y = divergence(Fy_x, Fy_y, Fy_z, x)
    momentum_z = divergence(Fz_x, Fz_y, Fz_z, x)

    energy = divergence(Fe_x, Fe_y, Fe_z, x)

    return (continuity, momentum_x, momentum_y, momentum_z, energy)


# ==========================================================================
# PDE loss
# ==========================================================================


def pde_loss(model, x):
    """
    Mean squared Euler residual.
    """

    continuity, momentum_x, momentum_y, momentum_z, energy = euler_residual(model, x)

    loss = (
        torch.mean(continuity.pow(2))
        + torch.mean(momentum_x.pow(2))
        + torch.mean(momentum_y.pow(2))
        + torch.mean(momentum_z.pow(2))
        + torch.mean(energy.pow(2))
    )

    return loss


# ==========================================================================
# Wall boundary condition
# ==========================================================================


def wall_loss(
    model: nn.Module,
    x_wall: torch.Tensor,
):
    """
    Slip-wall boundary condition: u · n = 0
    """

    _, u, v, w, _ = primitive_variables(model, x_wall)

    x = x_wall[:, 0:1]
    y = x_wall[:, 1:2]
    z = x_wall[:, 2:3]
    throat_ratio = x_wall[:, 3:4]

    R = nozzle_radius(x, throat_ratio)
    dRdx = nozzle_radius_gradient(x, throat_ratio)

    nx = -R * dRdx
    ny = y
    nz = z

    norm = torch.sqrt(nx**2 + ny**2 + nz**2)

    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    normal_velocity = u * nx + v * ny + w * nz

    return torch.mean(normal_velocity.pow(2))


# ==========================================================================
# Inlet boundary condition
# ==========================================================================


def inlet_loss(
    model: nn.Module,
    x_inlet: torch.Tensor,
):
    """
    Uniform inlet boundary condition.
    """

    rho, u, v, w, p = primitive_variables(model, x_inlet)
    rho_loss = torch.mean((rho - RHO_INLET).pow(2))
    u_loss = torch.mean((u - U_INLET).pow(2))
    v_loss = torch.mean((v - V_INLET).pow(2))
    w_loss = torch.mean((w - W_INLET).pow(2))
    p_loss = torch.mean((p - P_INLET).pow(2))

    return rho_loss + u_loss + v_loss + w_loss + p_loss


# ==========================================================================
# Outlet boundary condition
# ==========================================================================


def outlet_loss(model: nn.Module, x_outlet: torch.Tensor):
    """
    Pressure fixed-value outlet boundary condition with zero-gradient for the
    remaining primitive variables.
    """

    x_outlet = x_outlet.clone().requires_grad_(True)
    rho, u, v, w, p = primitive_variables(model, x_outlet)
    rho_x = gradient(rho, x_outlet)[0]
    u_x = gradient(u, x_outlet)[0]
    v_x = gradient(v, x_outlet)[0]
    w_x = gradient(w, x_outlet)[0]

    if FIXED_OUTLET_P_BC:
        p_loss = torch.mean((p - P_OUTLET).pow(2))  # For subsonic outlet
    else:
        p_x = gradient(p, x_outlet)[0]
        p_loss = torch.mean(p_x.pow(2))  # For supersonic outlet

    loss = (
        torch.mean(rho_x.pow(2))
        + torch.mean(u_x.pow(2))
        + torch.mean(v_x.pow(2))
        + torch.mean(w_x.pow(2))
        + p_loss
    )

    return loss


# ==========================================================================
# Loss evaluation
# ==========================================================================


def compute_losses(model, x_interior, x_wall, x_inlet, x_outlet):
    """
    Compute every individual loss term.
    """

    loss_pde = pde_loss(model, x_interior)
    loss_wall = wall_loss(model, x_wall)
    loss_inlet = inlet_loss(model, x_inlet)
    loss_outlet = outlet_loss(model, x_outlet)

    return (loss_pde, loss_wall, loss_inlet, loss_outlet)


# ==========================================================================
# Total objective function
# ==========================================================================


def total_loss(model, x_interior, x_wall, x_inlet, x_outlet):
    """
    Complete PINN objective.
    """

    loss_pde, loss_wall, loss_inlet, loss_outlet = compute_losses(
        model, x_interior, x_wall, x_inlet, x_outlet
    )
    loss = loss_pde * 2 + loss_wall * 2 + loss_inlet + loss_outlet

    return (loss, loss_pde, loss_wall, loss_inlet, loss_outlet)
