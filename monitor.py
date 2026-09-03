import numpy as np

from predictor import Predictor
from probes import centerline_points


def monitor_centerline_velocity(predictor: Predictor, throat_ratio: float):
    """
    Monitor the centerline velocity along the nozzle.

    Parameters
    ----------
    predictor : Predictor
        The predictor object used for inference.
    throat_ratio : float, optional
        The throat ratio of the nozzle, by default 0.60.

    Returns
    -------
    np.ndarray
        The centerline velocity along the nozzle.
    """

    # Define the x-coordinates along the nozzle centerline
    x_centerline = centerline_points(n_points=5)

    # Predict flow variables using the predictor
    _, u, v, w, _ = predictor.predict(x_centerline, throat_ratio=throat_ratio)

    # Extract the velocity component (assuming it's the first component)
    centerline_velocity = predictor.velocity_magnitude(u, v, w)

    return x_centerline, centerline_velocity
