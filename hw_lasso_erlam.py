import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
file_path = 'crime.csv'
df = pd.read_csv(file_path)

# Separate the data into features (X) and the target variable (y)
X = df.iloc[:, :-1].values  # Exclude the last column as the target variable
y = df.iloc[:, -1].values   # The last column is the target variable

# Custom Standard Scaler Implementation
def custom_standard_scaler(X_train, X_test):
    # Calculate mean and standard deviation from the training set
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Standardize the training set
    X_train_scaled = (X_train - mean) / std
    
    # Standardize the testing set using the same mean and std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled

# Custom Train-Test Split Implementation
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle the data
    data = np.arange(len(X))
    np.random.shuffle(data)
    X = X[data]
    y = y[data]
    
    # Calculate the split index
    split_index = int((1 - test_size) * len(X))
    
    # Split the data into training and testing sets
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test

# Split the data into a training set (S) and a testing set (T)
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of lambda values to test
lambda_values = np.logspace(-3, 3, 100)  # You can adjust the range as needed

# Function to compute soft threshold
def soft_threshold(rho, lambda_val):
    if rho < -lambda_val:
        return rho + lambda_val
    elif rho > lambda_val:
        return rho - lambda_val
    else:
        return 0

# Function to perform coordinate descent for Lasso
def coordinate_descent_lasso(X, y, lambda_val, tolerance=0.0005, max_iterations=100):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    old_beta = np.copy(beta)
    update_counts = []  # To store the number of CD updates
    
    for i in range(max_iterations):
        
        for j in range(n_features):
            
            # Calculate the residual without the j-th feature
            r = y - X.dot(beta) + X[:, j] * beta[j]
            
            # Compute the coordinate-wise update
            new_beta_j = soft_threshold(X[:, j].dot(r), lambda_val) / (X[:, j]**2).sum()
            
            if new_beta_j != beta[j]:
                update_counts.append(1)
            else:
                update_counts.append(0)
            
            # Update the j-th coefficient
            beta[j] = new_beta_j
        
        # Check for convergence
        if np.linalg.norm(beta - old_beta) < tolerance:
            break
        
        old_beta = np.copy(beta)
    
    return beta, update_counts

# Initialize lists to store testing errors
testing_errors = []

# Define the five specific lambda values you want to evaluate
selected_lambda_values = [0.5, 3, 25, 50, 100]

# Train Lasso models with different lambda values
for lambda_val in selected_lambda_values:
    beta, updates = coordinate_descent_lasso(X_train, y_train, lambda_val)
    
    # Calculate testing error (MSE) on the testing set (T)
    test_predictions = X_test.dot(beta)
    test_error = np.mean((test_predictions - y_test) ** 2)
    
    # Print testing error for each lambda value
    print(f"Lambda: {lambda_val:.3f}, Testing Error: {test_error:.3f}")
    
    
    testing_errors.append(test_error)

# Plot testing error versus selected lambda values
plt.figure(figsize=(10, 5))
plt.plot(selected_lambda_values, testing_errors, marker='o')
plt.xlabel("Lambda")
plt.ylabel("Testing Error")
plt.title("Testing Error vs. Lambda")
plt.grid(True)
plt.show()



