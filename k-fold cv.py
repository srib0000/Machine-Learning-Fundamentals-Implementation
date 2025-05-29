import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

# Load the data
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# percentages used for training and testing respectively
num_train = int(0.75 * n)
num_test = n - num_train

# split data into training set and testing set
sample_train = data[0:num_train, 0:-1]
sample_test = data[num_train:, 0:-1]
label_train = data[0:num_train, -1]
label_test = data[num_train:, -1]

# Values of lambda (regularization strength)
lambda_values = [0.1, 0.01, 0.5, 1, 10]
validation_errors_mean = []
validation_errors_std = []

# Perform K-fold cross-validation for each lambda value
k_fold = KFold(n_splits=4, shuffle=True)

for lambda_val in lambda_values:
    # Initialize the Ridge Regression model with the current lambda
    ridge_model = Ridge(alpha=lambda_val)
    
    # Perform cross-validation and calculate validation errors
    validation_scores = cross_val_score(ridge_model, sample_train, label_train, cv=k_fold, scoring='neg_mean_squared_error')
    validation_errors = -validation_scores  # Convert to positive MSE values
    
    # Calculate mean and standard deviation of validation errors
    mean_error = np.mean(validation_errors)
    std_error = np.std(validation_errors)
    
    # Store the mean and standard deviation of validation errors for this lambda
    validation_errors_mean.append(mean_error)
    validation_errors_std.append(std_error)
    
    # Print the mean and standard deviation of validation errors for this lambda
    print(f"Lambda = {lambda_val}: Mean Validation Error (MSE) = {mean_error}, Standard Deviation Validation Error = {std_error}")


