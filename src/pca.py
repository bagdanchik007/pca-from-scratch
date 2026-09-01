import numpy as np

from src.math.statistics import center_data
from src.math.covariance import covariance_matrix
from src.math.eigen import eigen_decomposition


class PCA:
    """
    Principal Component Analysis implementation.
    """

    def __init__(self, n_components: int):
        if n_components < 1:
            raise ValueError(
                "n_components must be at least 1."
            )

        self.n_components = n_components

        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "PCA":
        """
        Fit PCA to the input data.
        """

        if data.ndim != 2:
            raise ValueError(
                "Input data must be 2-dimensional."
            )

        n_features = data.shape[1]

        if self.n_components > n_features:
            raise ValueError(
                "n_components cannot exceed "
                "the number of features."
            )

        centered_data, mean = center_data(data)

        covariance = covariance_matrix(data)

        eigenvalues, eigenvectors = (
            eigen_decomposition(covariance)
        )

        sorted_indices = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = (
            eigenvalues[sorted_indices]
        )

        eigenvectors = (
            eigenvectors[:, sorted_indices]
        )

        self.components_ = (
            eigenvectors[:, :self.n_components]
        )

        self.mean_ = mean

        self.explained_variance_ = (
            eigenvalues[:self.n_components]
        )

        total_variance = np.sum(eigenvalues)

        self.explained_variance_ratio_ = (
            self.explained_variance_
            / total_variance
        )

        return self

    def transform(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Transform data into the principal
        component space.
        """

        if self.components_ is None:
            raise RuntimeError(
                "PCA must be fitted before "
                "calling transform()."
            )

        centered_data = data - self.mean_

        return (
            centered_data
            @ self.components_
        )

    def fit_transform(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Fit PCA and transform the data.
        """

        self.fit(data)

        return self.transform(data)