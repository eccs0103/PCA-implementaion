import numpy as np

from typing import Optional
from numpy import float64
from numpy.typing import NDArray
from typeguard import typechecked


class PCA:
	"""
	Principal Component Analysis (PCA) implementation using NumPy.

	This class provides functionality to reduce the dimensionality
	of data while preserving as much variance as possible.
	"""

	@typechecked
	def __init__(self, desired_principal_components: int) -> None:
		"""
		Initializes the PCA model with the desired number of principal components.

		Args:
			desired_principal_components (int): Number of principal components to retain.
		"""
		self.desired_principal_components: int = desired_principal_components
		self.extracted_eigenvectors: Optional[NDArray[float64]] = None
		self.feature_mean: Optional[NDArray[float64]] = None

	@typechecked
	def fit(self, feature_table: NDArray[float64]) -> None:
		"""
		Computes the principal components from the given feature table.

		Args:
			feature_table (NDArray[float64]): 2D array of shape (n_samples, n_features)
				containing the training data.
		"""
		self.feature_mean = np.mean(feature_table, axis=0)
		feature_table = feature_table - self.feature_mean

		covariance_matrix: NDArray[float64] = np.cov(feature_table.T)

		eigenvalues: NDArray[float64]
		eigenvectors: NDArray[float64]
		(eigenvalues, eigenvectors) = np.linalg.eig(covariance_matrix)
		eigenvectors = eigenvectors.T

		indices: NDArray[np.intp] = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[indices]
		eigenvectors = eigenvectors[indices]
		self.extracted_eigenvectors = eigenvectors[0:self.desired_principal_components]

	@typechecked
	def transform(self, feature_table: NDArray[float64]) -> NDArray[float64]:
		"""
		Applies the fitted PCA to transform the input feature table to principal components.

		Args:
			feature_table (NDArray[float64]): 2D array of shape (n_samples, n_features)
				containing the data to transform.

		Returns:
			NDArray[float64]: Transformed data in the principal component space.
		"""
		if self.feature_mean is None or self.extracted_eigenvectors is None: raise ValueError("PCA model is not fitted yet.")
		feature_table = feature_table - self.feature_mean
		return np.dot(feature_table, self.extracted_eigenvectors.T)

	@typechecked
	def inverse_transform(self, principal_components: NDArray[float64]) -> NDArray[float64]:
		"""
		Reconstructs the original feature data from the principal components.

		Args:
			principal_components (NDArray[float64]): 2D array of shape (n_samples, n_components)
				containing the data in the principal component space.

		Returns:
			NDArray[float64]: Approximated original feature data reconstructed from the components.
		"""
		if self.feature_mean is None or self.extracted_eigenvectors is None: raise ValueError("PCA model is not fitted yet.")
		return np.dot(principal_components, self.extracted_eigenvectors) + self.feature_mean
