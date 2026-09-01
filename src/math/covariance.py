import numpy as np

from src.math.statistics import center_data


def covariance_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculate the covariance matrix of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Covariance matrix with shape
        (n_features, n_features).
    """

    if data.ndim != 2:
        raise ValueError(
            "Input data must be a 2-dimensional array."
        )

    n_samples = data.shape[0]

    if n_samples < 2:
        raise ValueError(
            "At least two samples are required "
            "to calculate covariance."
        )

    centered_data, _ = center_data(data)

    return (
        centered_data.T @ centered_data
    ) / (n_samples - 1)
