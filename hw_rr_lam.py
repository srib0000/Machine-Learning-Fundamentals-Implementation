import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# percentages used for training and testing respectively
num_train = int(0.01* n)
num_test = int(0.25*n)

# split data into training set and testing set
sample_train = data[0:num_train, 0:-1]
sample_test = data[num_train:, 0:-1]
label_train = data[0:num_train, -1]
label_test = data[num_train:, -1]

# Values of lambda (regularization strength)
lambda_values = [0.5, 3, 12, 32, 55]
testing_errors = []

for lambda_val in lambda_values:
    # Implementing Ridge Regression from scratch
    X = sample_train
    y = label_train
    n, p = X.shape
    
    # Add a small regularization term to the diagonal of X'X
    alpha = lambda_val
    XTX = X.T @ X
    ridge_term = alpha * np.identity(p)
    XTX += ridge_term
    
    # Calculate the coefficients using the Ridge Regression formula
    coefficients = np.linalg.solve(XTX, X.T @ y)
    
    # Predict on the testing dataset
    y_pred = sample_test @ coefficients
    
    # Calculate the mean squared error on the testing dataset
    mse = np.mean((label_test - y_pred) ** 2)
    
    # Store the testing error for this lambda
    testing_errors.append(mse)
    
    # Print the testing error for this lambda
    print(f"Lambda = {lambda_val}: Testing Error (MSE) = {mse}")

# Plot the testing error vs. Lambda
plt.figure(figsize=(8, 6))
plt.semilogx(lambda_values, testing_errors, marker='o', linestyle='-')
plt.xlabel('Lambda')
plt.ylabel('Testing Error (MSE)')
plt.title('Testing Error vs. Lambda for Ridge Regression')
plt.grid(True)
plt.show()

