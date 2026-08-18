import numpy as np
import torch

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
    ) -> torch.Tensor:
        """
        Convert point data to a tensor and append the throat-ratio feature when
        the input is given as (x, y, z) only.
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
                throat_ratio = 0.60

            ratio = torch.as_tensor(
                throat_ratio, dtype=torch.float32, device=self.device
            )
            if ratio.dim() == 0:
                ratio = ratio.repeat(x.shape[0], 1)
            ratio = ratio.reshape(-1, 1)

            if ratio.shape[0] != x.shape[0]:
                raise ValueError(
                    "throat_ratio must be a scalar or match the number of points. "
                    f"Got {ratio.shape[0]} ratios for {x.shape[0]} points."
                )

            x = torch.cat([x, ratio], dim=1)
        elif x.shape[-1] != 4:
            raise ValueError(
                "Expected input points with shape (N, 3) or (N, 4); "
                f"got shape {tuple(x.shape)}."
            )

        return x

    def _primitive_variables(self, x: torch.Tensor):
        """
        Evaluate the neural network.

        Returns
        -------
        rho,u,v,w,p
        """

        q = self.model(x)

        rho = q[:, 0:1]
        u = q[:, 1:2]
        v = q[:, 2:3]
        w = q[:, 3:4]
        p = q[:, 4:5]

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
    ):
        """
        Evaluate the trained PINN.

        Parameters
        ----------
        x : Array-like of shape (N, 3) or (N, 4). If shape (N, 3), the throat
            ratio is appended automatically unless `throat_ratio` is provided.
        throat_ratio : Scalar or array-like matching the batch size. Used when
            evaluating a variable throat-ratio nozzle family.
        """

        x = self._prepare_input(x, throat_ratio=throat_ratio)

        with torch.no_grad():
            return self._primitive_variables(x)
