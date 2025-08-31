from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
iris = load_iris()
X = iris.data 
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
print("K-means Labels:", labels.tolist())
print("\nK-means Centroids:\n", pd.DataFrame(centroids, columns=iris.feature_names))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=60, c='red', marker='X', label='Centroids')  
plt.title("K-Means Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.grid(True)
plt.show()
