import matplotlib.pyplot as plt

from numpy import intp, float64
from numpy.typing import NDArray
from sklearn import datasets
from pca import PCA

wine_data = datasets.load_wine()

input_features_matrix: NDArray[float64] = wine_data.data
target_labels: NDArray[intp] = wine_data.target

pca_object: PCA = PCA(2)
pca_object.fit(input_features_matrix)
principal_components: NDArray[float64] = pca_object.transform(input_features_matrix)

PC1: NDArray[float64] = principal_components[:, 0]
PC2: NDArray[float64] = principal_components[:, 1]

plt.scatter(PC1, PC2, c=target_labels, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
