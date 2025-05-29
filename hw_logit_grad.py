import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
file_path = 'diabetes.csv'
df = pd.read_csv(file_path)

# Separate the data into features (X) and the target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Step 1: Implementing Train-Test Split
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]  
    
    return X_train, X_test, y_train, y_test

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Initialize model parameters and set the learning rate
theta = np.zeros(X_train.shape[1])
bias = 0
alpha = 0.01

# Step 3: Gradient Descent

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def least_squares_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def gradient_descent(X, y, theta, bias, alpha, num_iterations):
    m = len(y)
    errors = []

    for iteration in range(num_iterations):
        # Calculate predictions
        z = X.dot(theta) + bias
        y_pred = sigmoid(z)

        # Update parameters
        theta -= (alpha / m) * X.T.dot(y_pred - y)
        bias -= (alpha / m) * np.sum(y_pred - y)

        errors.append(least_squares_error(y, y_pred))

    return theta, bias, errors

# Train the model using gradient descent
num_iterations = 1000
theta, bias, errors = gradient_descent(X_train, y_train, theta, bias, alpha, num_iterations)

# Step 4: Evaluate the Model
z = X_test.dot(theta) + bias
y_pred = sigmoid(z)
testing_error_value = least_squares_error(y_test, y_pred)
print("Testing Error:", testing_error_value)

# Step 5: Plot the Loss vs. Number of Updates
plt.plot(range(num_iterations), errors)
plt.xlabel("Number of Updates")
plt.ylabel("Testing Error")
plt.title("Testing Error vs. Number of Updates")
plt.show()
