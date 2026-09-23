import torch

torch.set_default_dtype(torch.float32)


# ==========================================================================
# Automatic differentiation
# ==========================================================================


def gradient(f: torch.Tensor, x: torch.Tensor):
    """
    Compute df/dx, df/dy, df/dz using automatic differentiation.
    """

    if not x.requires_grad:
        x = x.requires_grad_(True)

    grad = torch.autograd.grad(
        outputs=f,
        inputs=x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]

    return (grad[:, 0:1], grad[:, 1:2], grad[:, 2:3])


# ==========================================================================
# Divergence of a vector field
# ==========================================================================


def divergence(Fx, Fy, Fz, x):
    """
    Compute div(F) for a vector field.
    """

    Fx_x = gradient(Fx, x)[0]
    Fy_y = gradient(Fy, x)[1]
    Fz_z = gradient(Fz, x)[2]

    return Fx_x + Fy_y + Fz_z
