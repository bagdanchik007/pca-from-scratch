import numpy as np
import pytest

from src.pca import PCA


def test_pca_reduces_dimensions():
    data = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
    ])

    pca = PCA(n_components=2)

    transformed = pca.fit_transform(data)

    assert transformed.shape == (4, 2)


def test_pca_stores_components():
    data = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ])

    pca = PCA(n_components=1)

    pca.fit(data)

    assert pca.components_ is not None
    assert pca.components_.shape == (2, 1)


def test_pca_stores_mean():
    data = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    pca = PCA(n_components=1)

    pca.fit(data)

    np.testing.assert_array_almost_equal(
        pca.mean_,
        np.array([3.0, 4.0])
    )


def test_pca_explained_variance_is_sorted():
    data = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ])

    pca = PCA(n_components=2)

    pca.fit(data)

    assert (
        pca.explained_variance_[0]
        >= pca.explained_variance_[1]
    )


def test_pca_transform_requires_fit():
    data = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    pca = PCA(n_components=1)

    with pytest.raises(RuntimeError):
        pca.transform(data)


def test_pca_rejects_invalid_components():
    with pytest.raises(ValueError):
        PCA(n_components=0)


def test_pca_rejects_too_many_components():
    data = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    pca = PCA(n_components=3)

    with pytest.raises(ValueError):
        pca.fit(data)