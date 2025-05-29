import numpy as np
from numpy.random import randn
from sklearn.metrics import mean_squared_error

# Data Preparation
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)
num_train = int(0.01 * n)
num_test = int(0.25 * n)

# Split Data into Training Set and Testing Set
sample_train = data[0:num_train, 0:-1]
sample_test = data[n - num_test:, 0:-1]
label_train = data[0:num_train, -1]
label_test = data[n - num_test:, -1]

# Hyperparameters
lam = 0.1  # Ridge regularization parameter
alpha = 0.01  # New regularization parameter

# Implementation of ridge regression
T = np.matmul(sample_train.T, sample_train) + lam * np.identity(p - 1)
T = np.linalg.inv(T)
T2 = np.matmul(sample_train.T, label_train)
beta = np.matmul(T, T2)

# Implementation of the new learner
theta = randn(p - 1, 1)[0]
T = np.matmul(sample_train.T, sample_train) + lam * np.identity(p - 1)
T = np.linalg.inv(T)
T2 = np.matmul(sample_train.T, label_train) + lam * theta + alpha * np.matmul(sample_train.T, np.matmul(sample_train, beta) - label_train)
beta_new = np.matmul(T, T2)

# Evaluate testing error (MSE)
label_test_pred_ridge = np.matmul(sample_test, beta)
error_test_ridge = mean_squared_error(label_test, label_test_pred_ridge)

label_test_pred_new = np.matmul(sample_test, beta_new)
error_test_new = mean_squared_error(label_test, label_test_pred_new)

# Print the errors
print("Ridge Regression Error:", error_test_ridge)
print("New Regularization Error:", error_test_new)
