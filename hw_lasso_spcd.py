import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('crime.csv')

# Define a custom function for train-test split
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    n = X.shape[0]
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(n)
    test_set_size = int(n * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

# Define the feature matrix (X) and target vector (y) using df.iloc
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column

# Split the data into training and testing sets using custom_train_test_split
X_train, X_test, y_train, y_test = custom_train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Function to compute soft threshold
def soft_threshold(rho, lambda_val):
    if rho < -lambda_val:
        return rho + lambda_val
    elif rho > lambda_val:
        return rho - lambda_val
    else:
        return 0

# Define the Lasso coordinate descent function
def lasso_coordinate_descent(X, y, lambda_, max_iterations=100):
    n, p = X.shape
    beta = np.zeros(p)
    
    sparsity_list = []
    
    for iteration in range(max_iterations):
        for j in range(p):
            # Calculate the residual
            r = y - np.dot(X, beta) + beta[j] * X[:, j]
            
            # Update beta[j] using soft thresholding
            beta[j] = np.dot(X[:, j], r) / np.sum(X[:, j]**2)
            beta[j] = np.sign(beta[j]) * max(0, abs(beta[j]) - lambda_)
            
        # Calculate model sparsity
        sparsity = np.sum(beta == 0) / p
        sparsity_list.append(sparsity)
    
    return beta, sparsity_list

# Define your chosen lambda
lambda_ = 0.001

# Run Lasso coordinate descent on the training data
beta, sparsity_list = lasso_coordinate_descent(X_train, y_train, lambda_)

# Evaluate the model on the testing data
predictions = np.dot(X_test, beta)

# Calculate model sparsity versus the number of CD updates
x_values = np.arange(1, len(sparsity_list) + 1)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x_values, sparsity_list)
plt.xlabel("Number of CD Updates")
plt.ylabel("Model Sparsity")
plt.title(f"Lasso Model Sparsity (λ={lambda_})")
plt.grid(True)
plt.show()

# Report the final sparsity
final_sparsity = sparsity_list[-1]
print(f"Final Model Sparsity: {final_sparsity}")
