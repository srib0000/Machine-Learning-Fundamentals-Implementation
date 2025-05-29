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

def pca_reconstruction(X, k):
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
    
    # Reconstruct the data from the projected set
    reconstructed_data = projected_data.dot(top_k_eigenvectors.T)
    
    # Calculate the reconstruction error
    reconstruction_error = np.mean(np.square(X - reconstructed_data))
    
    return reconstructed_data, reconstruction_error

# Specify the values of k
k_values = [2, 5, 10, 15, 20]

# For each value of k, perform PCA and calculate the reconstruction error
reconstruction_errors = []
for k in k_values:
    _, error = pca_reconstruction(sample_train_scaled, k)
    reconstruction_errors.append(error)

# Plot the reconstruction errors versus k
plt.plot(k_values, reconstruction_errors, marker='o')
plt.title('Reconstruction Error vs. k')
plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Reconstruction Error')
plt.show()
