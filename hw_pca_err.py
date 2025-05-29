import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the Diabetes dataset from CSV file
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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    n, m = X.shape
    X = np.c_[np.ones((n, 1)), X]  
    
    theta = np.zeros((m + 1, 1))
    
    for _ in range(num_iterations):
        
        # Calculate the hypothesis
        z = np.dot(X, theta)
        h = sigmoid(z)
        
        # Calculate the gradient and update the parameters
        gradient = np.dot(X.T, (h - y)) / n
        theta -= learning_rate * gradient
    
    return theta

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
    
    return projected_data, top_k_eigenvectors

# Specify the values of k
k_values = [2, 5, 10, 15, 20]

testing_errors = []
for k in k_values:
    
    # Project training and testing data using PCA
    projected_train_data, _ = pca(sample_train_scaled, k)
    projected_test_data, _ = pca(sample_test_scaled, k)  

    # Train logistic regression model on the projected training set
    theta = logistic_regression(projected_train_data, label_train[:, np.newaxis])

    # Predict using the logistic regression model
    predictions = (sigmoid(np.dot(np.c_[np.ones((projected_test_data.shape[0], 1)), projected_test_data], theta)) >= 0.5).astype(int)

    # Calculate testing accuracy
    accuracy = np.mean(predictions.flatten() == label_test)
    testing_error = 1 - accuracy
    testing_errors.append(testing_error)

# Plot the testing errors versus k
plt.plot(k_values, testing_errors, marker='o')
plt.title('Testing Error vs. k')
plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Testing Error')
plt.show()
