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
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
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
def coordinate_descent_lasso(X, y, lambda_val, max_iterations=100):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    
    mse_list = []
    cd_updates = []
    
    for iteration in range(max_iterations):
        
        for j in range(n_features):
            
            # Calculate the residual without the j-th feature
            r = y - X.dot(beta) + X[:, j] * beta[j]
            
            # Compute the coordinate-wise update
            new_beta_j = soft_threshold(X[:, j].dot(r), lambda_val) / (X[:, j]**2).sum()
            
            # Update the j-th coefficient
            beta[j] = new_beta_j
        
        # Calculate predictions using the current model
        predictions = X_test.dot(beta)
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((y_test - predictions) ** 2)
        
        # Store MSE and CD updates
        mse_list.append(mse)
        cd_updates.append(iteration)
    
    return mse_list, cd_updates

# Define your chosen lambda
lambda_ = 0.1

# Run Lasso coordinate descent on the training data
mse_list, cd_updates = coordinate_descent_lasso(X_train, y_train, lambda_)

# Plot the testing error versus the number of CD updates
plt.figure(figsize=(8, 6))
plt.plot(cd_updates, mse_list)
plt.xlabel("Number of CD Updates")
plt.ylabel("Testing Error (MSE)")
plt.title(f"Lasso Testing Error (λ={lambda_})")
plt.grid(True)
plt.show()

# Report the final testing error
final_testing_error = mse_list[-1]
print(f"Final Testing Error (MSE): {final_testing_error}")
