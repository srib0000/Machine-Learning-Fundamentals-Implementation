import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Loading the Diabetes dataset from CSV file
data = np.loadtxt('diabetes.csv', delimiter=',')
n, p = np.shape(data)

# Splitting the dataset into training and testing sets
num_train = int(0.75 * n)
num_test = int(0.25 * n)

sample_train = data[0:num_train, 0:-1]
sample_test = data[n - num_test:, 0:-1]
label_train = data[0:num_train, -1]
label_test = data[n - num_test:, -1]

# Standardizing the data using StandardScaler
scaler = StandardScaler()
sample_train = scaler.fit_transform(sample_train)

# Implementing K-means clustering
def kmeans(X, k, max_iters=100):
    n, d = X.shape
    
    # Initializing centroids randomly
    centroids = X[np.random.choice(n, k, replace=False)]
    
    for _ in range(max_iters):
        
        # Assigning each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Updating centroids based on the mean of data points in each cluster
        centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
    
    return labels

# Implementing PCA
def pca(X, num_components=2):
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:num_components]
    top_eigenvectors = eigenvectors[:, top_indices]
    principal_components = np.dot(X, top_eigenvectors)
    return principal_components, top_eigenvectors

# Applying PCA to reduce feature dimension to 2
num_components = 2
X_pca, eigenvectors = pca(sample_train, num_components)

# Applying K-means to cluster the Diabetes dataset into k groups
k_values = [3]

for k in k_values:
    
    # Getting cluster labels
    labels = kmeans(X_pca, k)
    
# Plotting the projected data and mark the clusters using different colors
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title(f'K-means Clustering (k={k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
