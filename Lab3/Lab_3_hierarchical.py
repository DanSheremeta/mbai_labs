import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

data = pd.read_csv('out.csv')

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1])
plt.show()

data = list(zip(data_2d[:, 0], data_2d[:, 1]))

hierarchical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels)

cluster_centers = []
for cluster_label in set(labels):
    cluster_points = data_2d[labels == cluster_label]
    cluster_center = np.mean(cluster_points, axis=0)
    cluster_centers.append(cluster_center)

cluster_centers = np.array(cluster_centers)

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='.', s=100)

plt.show()
