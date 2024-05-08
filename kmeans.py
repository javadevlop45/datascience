import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generating random data
n_samples = 300
n_clusters = 3
random_state = 20
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state)

# Visualizing the generated data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Generated Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Performing K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
kmeans.fit(X)

# Getting the cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualizing the clustering result
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('K-Means Clustering Result')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
