import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, random_state=42)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

cluster_center = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_center[:, 0], cluster_center[:, 1], marker='x', s=300, c='red')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('K means Clustring')
plt.show()