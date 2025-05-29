import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
file_path = 'diabetes.csv'
df = pd.read_csv(file_path)

# Separate the data into features (X) and the target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Step 1: Implementing train_test_split
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

# Step 2: Initialize model parameters
theta = np.zeros(X_train.shape[1])

# Step 3: Newton's Method

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def least_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def hessian(X, theta):
    z = X.dot(theta)
    p = sigmoid(z)
    return X.T.dot(np.diag(p * (1 - p))).dot(X)

def newton_method(X, y, theta, num_iterations):
    m = len(y)
    errors = []

    for iteration in range(num_iterations):
        z = X.dot(theta)
        y_pred = sigmoid(z)
        gradient = X.T.dot(y_pred - y)
        hess = hessian(X, theta)
        theta -= np.linalg.inv(hess).dot(gradient)
        
        errors.append(least_squared_error(y, y_pred))

    return theta, errors

# Train the model using Newton's method
num_iterations = 100
theta, errors = newton_method(X_train, y_train, theta, num_iterations)

# Step 4: Evaluate the Model
z = X_test.dot(theta)
y_pred = sigmoid(z)
testing_error_value = least_squared_error(y_test, y_pred)
print("Testing Error :", testing_error_value)

# Step 5: Plot the Loss vs. Number of Updates
plt.plot(range(num_iterations), errors)
plt.xlabel("Number of Updates")
plt.ylabel("Testing Error")
plt.title("Testing Error vs. Number of Updates")
plt.show()
