import numpy as np
import pytest

from src.math.statistics import mean, center_data


def test_mean():
    data = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    result = mean(data)

    expected = np.array([
        3.0,
        4.0
    ])

    np.testing.assert_array_almost_equal(
        result,
        expected
    )


def test_center_data():
    data = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    centered_data, feature_means = center_data(
        data
    )

    expected_means = np.array([
        3.0,
        4.0
    ])

    expected_centered = np.array([
        [-2.0, -2.0],
        [0.0, 0.0],
        [2.0, 2.0]
    ])

    np.testing.assert_array_almost_equal(
        feature_means,
        expected_means
    )

    np.testing.assert_array_almost_equal(
        centered_data,
        expected_centered
    )


def test_centered_data_has_zero_mean():
    data = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0]
    ])

    centered_data, _ = center_data(data)

    result = np.mean(
        centered_data,
        axis=0
    )

    expected = np.array([
        0.0,
        0.0
    ])

    np.testing.assert_array_almost_equal(
        result,
        expected
    )


def test_mean_requires_2d_array():
    data = np.array([
        1.0,
        2.0,
        3.0
    ])

    with pytest.raises(ValueError):
        mean(data)


def test_mean_requires_non_empty_data():
    data = np.empty((0, 2))

    with pytest.raises(ValueError):
        mean(data)