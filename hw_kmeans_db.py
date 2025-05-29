import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Implementation of PCA
def pca(X, num_components):
    
    # Centering the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Principal Components
    principal_components = Vt[:num_components, :]

    # Projecting the data onto the principal components
    X_pca = np.dot(X_centered, principal_components.T)

    return X_pca

# Specifying the number of components for PCA
num_components = 2

# Applying PCA to the standardized data
sample_train_pca = pca(sample_train, num_components)

# Implementation of K-means clustering algorithm
def kmeans(X, k, max_iters=100):
    
    # Initializing centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        
        # Assigning each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Updating centroids based on the mean of the assigned points
        centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
    
    return labels, centroids

# Davies-Bouldin Index calculation
def davies_bouldin_index(X, labels, centroids):
    k = len(np.unique(labels))
    cluster_distances = np.zeros(k)
    
    for i in range(k):
        
        cluster_points = X[labels == i]
        intra_cluster_distance = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))
        
        inter_cluster_distances = []
        
        for j in range(k):
            
            if i != j:
                other_cluster_points = X[labels == j]
                inter_cluster_distances.append(
                    (intra_cluster_distance + np.mean(np.linalg.norm(cluster_points - centroids[j], axis=1))) /
                    np.linalg.norm(centroids[i] - centroids[j])
                )
        
        cluster_distances[i] = max(inter_cluster_distances)
    
    return np.mean(cluster_distances)

# Evaluating K-means clustering for different values of K
k_values = [2, 3, 4, 5, 6]
db_indices = []

for k in k_values:
    
    # Applying PCA and K-means clustering
    sample_train_pca = pca(sample_train, num_components)
    labels, centroids = kmeans(sample_train_pca, k)
    
    # Evaluating clustering using Davies-Bouldin Index
    db_index = davies_bouldin_index(sample_train_pca, labels, centroids)
    db_indices.append(db_index)

# Plotting the Davies-Bouldin Index vs. K
plt.plot(k_values, db_indices, marker='o')
plt.title('Davies-Bouldin Index vs. K with PCA')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Davies-Bouldin Index')
plt.grid(True)
plt.show()
