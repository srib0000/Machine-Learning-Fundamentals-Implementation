import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the Diabetes dataset from the CSV file
data = np.loadtxt('diabetes.csv', delimiter=',')
[n, p] = np.shape(data)

# Split the dataset into training and testing sets
num_train = int(0.75 * n)
num_test = int(0.25 * n)

sample_train = data[0:num_train, 0:-1]
sample_test = data[n - num_test:, 0:-1]
label_train = data[0:num_train, -1]
label_test = data[n - num_test:, -1]

# Standardize the data (mean=0 and variance=1)
scaler = StandardScaler()
sample_train_scaled = scaler.fit_transform(sample_train)
sample_test_scaled = scaler.transform(sample_test)

def pca(X, k):
    # Calculate the covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)
    
    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvectors based on eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]
    
    # Project the data onto the k-dimensional subspace
    projected_data = X.dot(top_k_eigenvectors)
    
    return projected_data


# Specify the number of components (k) for PCA
k = 2

# Apply PCA to the training data
projected_train_data = pca(sample_train_scaled, k)

# Plot the projected instances
plt.scatter(projected_train_data[:, 0], projected_train_data[:, 1])
plt.title('PCA: 2-dimensional feature space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()