import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data (replace 'crime.csv' with your dataset)
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
lambda_values = [0.01, 3, 20, 50, 100]
testing_errors = []

for lambda_val in lambda_values:
    # Initialize the Ridge Regression model with the current lambda
    ridge_model = Ridge(alpha=lambda_val)
    
    # Train the model on the training dataset
    ridge_model.fit(sample_train, label_train)
    
    # Predict on the testing dataset
    y_pred = ridge_model.predict(sample_test)
    
    # Calculate the mean squared error on the testing dataset
    mse = mean_squared_error(label_test, y_pred)
    
    # Store the testing error for this lambda
    testing_errors.append(mse)
    
    # Print the testing error for this lambda
    print(f"Lambda = {lambda_val}: Testing Error (MSE) = {mse}")

# Plot the testing error vs. Lambda
plt.figure(figsize=(8, 6))
plt.semilogx(lambda_values, testing_errors, marker='o', linestyle='-')
plt.xlabel('Lambda (Regularization Strength)')
plt.ylabel('Testing Error (MSE)')
plt.title('Testing Error vs. Lambda for Ridge Regression')
plt.grid(True)
plt.show()


