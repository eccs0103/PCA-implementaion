import numpy as np

from typing import Optional
from numpy import float64
from numpy.typing import NDArray


class PCA:

	def __init__(self, desired_principal_components: int) -> None:
		self.desired_principal_components: int = desired_principal_components
		self.extracted_eigenvectors: Optional[NDArray[float64]] = None
		self.feature_mean: Optional[NDArray[float64]] = None

	def fit(self, feature_table: NDArray[float64]) -> None:
		self.feature_mean = np.mean(feature_table, axis=0)
		feature_table = feature_table - self.feature_mean
		covariance_matrix: NDArray[float64] = np.cov(feature_table.T)
		eigenvalues: NDArray[float64]
		eigenvectors: NDArray[float64]
		eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
		eigenvectors = eigenvectors.T
		indices: NDArray[np.intp] = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[indices]
		eigenvectors = eigenvectors[indices]
		self.extracted_eigenvectors = eigenvectors[0:self.desired_principal_components]

	def transform(self, feature_table: NDArray[float64]) -> NDArray[float64]:
		if self.feature_mean is None or self.extracted_eigenvectors is None: raise ValueError("PCA model is not fitted yet.")
		feature_table = feature_table - self.feature_mean
		return np.dot(feature_table, self.extracted_eigenvectors.T)
