import numpy as np
import pytest

from src.math.eigen import eigen_decomposition


def test_eigen_decomposition():
    matrix = np.array([
        [2.0, 0.0],
        [0.0, 3.0],
    ])

    eigenvalues, eigenvectors = (
        eigen_decomposition(matrix)
    )

    np.testing.assert_array_almost_equal(
        eigenvalues,
        np.array([2.0, 3.0])
    )

    expected_vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    np.testing.assert_array_almost_equal(
        np.abs(eigenvectors),
        expected_vectors
    )


def test_eigenvectors_satisfy_equation():
    matrix = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    eigenvalues, eigenvectors = (
        eigen_decomposition(matrix)
    )

    for index in range(len(eigenvalues)):
        eigenvalue = eigenvalues[index]

        eigenvector = eigenvectors[:, index]

        left = matrix @ eigenvector

        right = eigenvalue * eigenvector

        np.testing.assert_array_almost_equal(
            left,
            right
        )


def test_requires_square_matrix():
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    with pytest.raises(ValueError):
        eigen_decomposition(matrix)


def test_requires_symmetric_matrix():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    with pytest.raises(ValueError):
        eigen_decomposition(matrix)


def test_requires_2d_matrix():
    matrix = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        eigen_decomposition(matrix)
