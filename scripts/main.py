import numpy as np
import matplotlib.pyplot as plt

from numpy import intp, float64
from numpy.typing import NDArray
from matplotlib.axes import Axes
from sklearn import datasets
from pca import PCA

#region Data initialization
X_features: NDArray[float64]
y_types: NDArray[intp]
(X_features, y_types) = datasets.load_iris(return_X_y=True)

colors: list[str] = ["blue", "darkorange", "green"]
types: list[str] = ['setosa', 'versicolor', 'virginica']
color_labels: NDArray = np.array([colors[label] for label in y_types])
type_labels: NDArray = np.array([types[label] for label in y_types])
#endregion

#region PCA initialization
pca_object: PCA = PCA(2)
pca_object.fit(X_features)

X_reduced_features: NDArray[float64] = pca_object.transform(X_features)

X_inversed_features: NDArray[float64] = pca_object.inverse_transform(X_reduced_features)

mse_per_feature: NDArray[float64] = np.mean((X_features - X_inversed_features)**2, axis=0)
#endregion

#region PCA features comparision
axes_before: Axes
axes_after: Axes
(_, (axes_before, axes_after)) = plt.subplots(1, 2, figsize=(12, 5))

# axes_before.legend(loc="best", title="Classes")
axes_before.scatter(X_features[:, 0], X_features[:, 1], color=color_labels, alpha=0.8, label=type_labels)
axes_before.set_title('Before PCA')
axes_before.set_xlabel('Future 1')
axes_before.set_ylabel('Future 2')

# axes_after.legend(loc="best", title="Classes")
axes_after.scatter(X_reduced_features[:, 0], X_reduced_features[:, 1], color=color_labels, alpha=0.8, label=type_labels)
axes_after.set_title('After PCA')
axes_after.set_xlabel('Reduced future 1')
axes_after.set_ylabel('Reduced future 2')

plt.tight_layout()
plt.show()
#endregion

#region PCA details comparision
axes_heatmap: Axes
axes_MSE: Axes
(_, (axes_heatmap, axes_MSE)) = plt.subplots(1, 2, figsize=(12, 5))

difference = np.abs(X_features - X_inversed_features)
axes_heatmap.imshow(difference, aspect='auto', cmap='hot')
axes_heatmap.set_title('Difference heatmap')
axes_heatmap.set_xlabel('Features')
axes_heatmap.set_ylabel('Samples')

axes_MSE.bar(np.arange(1, 5), mse_per_feature)
axes_MSE.set_title('MSE by features')
axes_MSE.set_xlabel('Feature')
axes_MSE.set_ylabel('MSE')

plt.tight_layout()
plt.show()
#endregion
