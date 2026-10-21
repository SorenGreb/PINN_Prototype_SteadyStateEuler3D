import numpy as np
import torch

from GEOMETRY.geometry import MACH_MAX, MACH_MIN

torch.set_default_dtype(torch.float32)


class Predictor:
    """
    Wrapper around a trained PyTorch model for predicting flow variables.
    """

    def __init__(self, model, device: torch.device | None = None) -> None:
        """
        Initialize the predictor with a trained model and target device.

        Parameters
        ----------
        model : Trained PyTorch model used for inference.
        device : Device on which to run the model. If None, uses CUDA when
            available and otherwise falls back to CPU.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def _prepare_input(
        self,
        x: np.ndarray | torch.Tensor,
        throat_ratio: float | np.ndarray | torch.Tensor | None = None,
        mach_number: float | np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert point data to model inputs (x, y, z, throat ratio, exit Mach).
        """

        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(device=self.device, dtype=torch.float32)
        else:
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] == 3:
            if throat_ratio is None:
                raise ValueError("throat_ratio is required for (N, 3) coordinates")
            if mach_number is None:
                raise ValueError("mach_number is required for (N, 3) coordinates")
            ratio = self._expand_parameter(throat_ratio, x.shape[0], "throat_ratio")
            x = torch.cat([x, ratio], dim=1)
        elif x.shape[-1] not in (4, 5):
            raise ValueError(
                "Expected input points with shape (N, 3), (N, 4), or (N, 5); "
                f"got shape {tuple(x.shape)}."
            )

        if x.shape[-1] == 4:
            if mach_number is None:
                raise ValueError("mach_number is required for (N, 4) inputs")
            mach = self._expand_parameter(mach_number, x.shape[0], "mach_number")
            x = torch.cat([x, mach], dim=1)

        mach_values = x[:, 4]
        if not torch.isfinite(mach_values).all():
            raise ValueError("mach_number values must be finite")
        if torch.any((mach_values < MACH_MIN) | (mach_values > MACH_MAX)):
            raise ValueError(f"mach_number must be between {MACH_MIN} and {MACH_MAX}")

        return x

    def _expand_parameter(
        self,
        value: float | np.ndarray | torch.Tensor,
        n_points: int,
        name: str,
    ) -> torch.Tensor:
        parameter = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if parameter.dim() == 0:
            parameter = parameter.expand(n_points)
        parameter = parameter.reshape(-1, 1)
        if parameter.shape[0] != n_points:
            raise ValueError(
                f"{name} must be a scalar or match the number of points. "
                f"Got {parameter.shape[0]} values for {n_points} points."
            )
        return parameter

    def _primitive_variables(self, x: torch.Tensor):
        """
        Evaluate the neural network.

        Returns
        -------
        rho,u,v,w,p
        """

        q = self.model(x)

        rho = torch.nn.functional.softplus(q[:, 0:1]) + 1.0e-6
        u = q[:, 1:2]
        v = q[:, 2:3]
        w = q[:, 3:4]
        p = torch.nn.functional.softplus(q[:, 4:5]) + 1.0e-6

        return (rho, u, v, w, p)

    @staticmethod
    def velocity_magnitude(
        u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(u**2 + v**2 + w**2)

    def predict(
        self,
        x: np.ndarray | torch.Tensor,
        throat_ratio: float | np.ndarray | torch.Tensor | None = None,
        mach_number: float | np.ndarray | torch.Tensor | None = None,
    ):
        """
        Evaluate the trained PINN.

        Parameters
        ----------
        x : Array-like of shape (N, 3), (N, 4), or (N, 5). Three-column
            coordinates receive both parameters; four-column inputs receive
            the exit Mach value.
        throat_ratio : Scalar or array-like matching the batch size. Used when
            evaluating a variable throat-ratio nozzle family.
        mach_number : Required for 3- or 4-column input; scalar or array-like in
            [2, 4]. A 5-column input already contains the Mach feature.
        """

        x = self._prepare_input(
            x, throat_ratio=throat_ratio, mach_number=mach_number
        )

        with torch.no_grad():
            return self._primitive_variables(x)
