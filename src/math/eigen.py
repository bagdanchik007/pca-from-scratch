import numpy as np


def eigen_decomposition(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a symmetric matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square symmetric matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Eigenvalues and eigenvectors.
    """

    if matrix.ndim != 2:
        raise ValueError(
            "Input matrix must be 2-dimensional."
        )

    rows, cols = matrix.shape

    if rows != cols:
        raise ValueError(
            "Input matrix must be square."
        )

    if not np.allclose(matrix, matrix.T):
        raise ValueError(
            "Input matrix must be symmetric."
        )

    eigenvalues, eigenvectors = np.linalg.eigh(
        matrix
    )

    return eigenvalues, eigenvectors
