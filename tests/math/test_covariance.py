import numpy as np
import pytest

from src.math.covariance import covariance_matrix


def test_covariance_matrix():
    data = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    result = covariance_matrix(data)

    expected = np.array([
        [4.0, 4.0],
        [4.0, 4.0]
    ])

    np.testing.assert_array_almost_equal(
        result,
        expected
    )


def test_covariance_matrix_is_symmetric():
    data = np.array([
        [1.0, 2.0],
        [2.0, 5.0],
        [3.0, 4.0],
        [4.0, 8.0]
    ])

    result = covariance_matrix(data)

    np.testing.assert_array_almost_equal(
        result,
        result.T
    )


def test_covariance_matrix_matches_numpy():
    data = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 7.0],
        [4.0, 8.0]
    ])

    result = covariance_matrix(data)

    expected = np.cov(
        data,
        rowvar=False
    )

    np.testing.assert_array_almost_equal(
        result,
        expected
    )


def test_covariance_requires_2d_data():
    data = np.array([
        1.0,
        2.0,
        3.0
    ])

    with pytest.raises(ValueError):
        covariance_matrix(data)


def test_covariance_requires_at_least_two_samples():
    data = np.array([
        [1.0, 2.0]
    ])

    with pytest.raises(ValueError):
        covariance_matrix(data)
