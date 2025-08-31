from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load data
iris = load_iris()
X = iris.data  

# Compute linkage matrices
Z_single = linkage(X, method='single')
Z_complete = linkage(X, method='complete')

# Plot dendrograms
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
dendrogram(Z_single, truncate_mode="level", p=5)
plt.title("Dendrogram - Single Linkage")
plt.xlabel("Sample Index")
plt.ylabel("Distance")

plt.subplot(1, 2, 2)
dendrogram(Z_complete, truncate_mode="level", p=5)
plt.title("Dendrogram - Complete Linkage")
plt.xlabel("Sample Index")
plt.ylabel("Distance")

plt.tight_layout()
plt.show()
