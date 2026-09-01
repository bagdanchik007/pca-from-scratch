import numpy as np


def mean(data: np.ndarray) -> np.ndarray:
    """
    Calculate the mean value for each feature.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Mean value of each feature.
    """

    if data.ndim != 2:
        raise ValueError(
            "Input data must be a 2-dimensional array."
        )

    if data.shape[0] == 0:
        raise ValueError(
            "Input data must contain at least one sample."
        )

    return np.mean(data, axis=0)


def center_data(
    data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Center data around the mean.

    PCA requires centered data. The mean of every
    feature is subtracted from the corresponding values.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_samples, n_features).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:

        - centered_data
        - feature_means
    """

    feature_means = mean(data)

    centered_data = data - feature_means

    return centered_data, feature_means