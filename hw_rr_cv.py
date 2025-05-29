import numpy as np

# Define Ridge Regression function
def ridge_regression(X, y, lambda_val):
    n, p = X.shape
    # Augment the feature matrix with a bias term (intercept)
    X_augmented = np.c_[X, np.ones(n)]
    
    # Compute the Ridge Regression coefficients using the closed-form solution
    A = X_augmented.T @ X_augmented + lambda_val * np.identity(p + 1)
    b = X_augmented.T @ y
    coefficients = np.linalg.solve(A, b)
    
    return coefficients[:-1], coefficients[-1]

# Load the data
data = np.loadtxt('crime.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)

# Values of lambda
lambda_values = [0.5, 3, 12, 32, 55]

# Number of folds for cross-validation
k = 4

# Initialize lists to store validation errors for each lambda
validation_errors_mean = []
validation_errors_std = []

# Shuffle the data randomly
np.random.shuffle(data)

# Calculate the number of samples in each fold
fold_size = n // k

for lambda_val in lambda_values:
    # Initialize lists to store validation errors for each fold
    validation_errors = []

    for fold in range(k):
        # Determine the indices for the current fold
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k - 1 else n

        # Split the data into training and validation sets for this fold
        sample_train_fold = np.concatenate((data[:start_idx, 0:-1], data[end_idx:, 0:-1]))
        label_train_fold = np.concatenate((data[:start_idx, -1], data[end_idx:, -1]))
        sample_val_fold = data[start_idx:end_idx, 0:-1]
        label_val_fold = data[start_idx:end_idx, -1]

        # Perform Ridge Regression on the training fold
        coefficients, intercept = ridge_regression(sample_train_fold, label_train_fold, lambda_val)

        # Make predictions on the validation fold
        predictions = sample_val_fold @ coefficients + intercept

        # Calculate the mean squared error for this fold
        fold_error = np.mean((label_val_fold - predictions) ** 2)
        validation_errors.append(fold_error)

    # Calculate mean and standard deviation of validation errors for this lambda
    mean_error = np.mean(validation_errors)
    std_error = np.std(validation_errors)

    # Store the mean and standard deviation of validation errors for this lambda
    validation_errors_mean.append(mean_error)
    validation_errors_std.append(std_error)

    # Print the mean and standard deviation of validation errors for this lambda
    print(f"Lambda = {lambda_val}: Mean Validation Error = {mean_error}, Standard Deviation Validation Error = {std_error}")
