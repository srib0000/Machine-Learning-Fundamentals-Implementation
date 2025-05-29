import numpy as np
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

# Implementing PCA
def pca(X, num_components=2):
    cov_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:num_components]
    top_eigenvectors = eigenvectors[:, top_indices]
    principal_components = np.dot(X, top_eigenvectors)
    return principal_components

# Implementing K-means clustering
def kmeans(X, k, max_iters=1000):
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

# Implementing Rand Index calculation
def rand_score(true_labels, pred_labels):
    n = len(true_labels)
    
    # Calculating the number of concordant and discordant pairs
    concordant_pairs = discordant_pairs = 0
    
    for i in range(n - 1):
        
        for j in range(i + 1, n):
            
            same_true_cluster = true_labels[i] == true_labels[j]
            same_pred_cluster = pred_labels[i] == pred_labels[j]
            
            if same_true_cluster and same_pred_cluster:
                concordant_pairs += 1
            elif not same_true_cluster and not same_pred_cluster:
                discordant_pairs += 1
    
    # Calculating the Rand Index
    rand_index = (concordant_pairs + discordant_pairs) / (n * (n - 1) / 2)
    
    return rand_index

# Evaluating clustering for different values of K
k_values = [2, 3, 4, 5, 6]  
rand_scores = []

for k in k_values:
    
    # Applying PCA
    X_pca = pca(sample_train, num_components=2)
    
    # Applying K-means
    labels = kmeans(X_pca, k)
    
    # Evaluating clustering using Rand Index
    rand_score_value = rand_score(label_train, labels)
    rand_scores.append(rand_score_value)

# Plotting Rand Index versus K
plt.plot(k_values, rand_scores, marker='o')
plt.title('Rand Index versus K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Rand Index')
plt.show()
