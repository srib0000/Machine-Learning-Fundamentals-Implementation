import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load the dataset
dataset = pd.read_csv("crime.csv")
data = np.loadtxt("crime.csv", delimiter=',', skiprows=1)
[n, p] = np.shape(data)
size = int(0.75 * n)
training_set = data[0:size, 0:]

# Define a list of tau values
taus = [0.3, 0.5, 0.7, 0.9, 1]

# Lists to store training and testing errors
training_errors = []
testing_errors = []

for tau in taus: # Iterate over the tau values
    num_train = int(tau * size) # Calculate the number of training samples based on tau
    
    # Split data into training set and testing set
    sample_train = training_set[0:num_train, 0:-1]
    sample_test = data[size:, 0:-1]
    label_train = training_set[0:num_train, -1]
    label_test = data[size:, -1]
    
    # Train the linear regression model
    model = linear_model.LinearRegression()
    model.fit(sample_train, label_train)
    
    # Predict labels for training and testing sets
    label_train_pred = model.predict(sample_train)
    label_test_pred = model.predict(sample_test)
    
    # Calculate the training error
    a = (label_train - label_train_pred) ** 2
    weights = np.eye(len(a))
    final_value1 = np.sum(weights * a)
    training_errors.append(final_value1)
    print("Training Error (tau = {}): {:.2f}".format(tau, final_value1))

    # Calculate the testing error
    b = (label_test - label_test_pred) ** 2
    weights1 = np.eye(len(b))
    final_value2 = np.sum(weights1 * b)
    testing_errors.append(final_value2)
    print("Testing Error (tau = {}): {:.2f}".format(tau, final_value2))


# Plotting the training and testing errors
plt.figure(figsize=(8, 6))
plt.plot(taus, training_errors, marker='o', label='Training Error')
plt.plot(taus, testing_errors, marker='o', label='Testing Error')
plt.xlabel('Tau')
plt.ylabel('Mean Squared Error')
plt.title('Training and Testing Errors vs. Tau')
plt.legend()
plt.grid(True)
plt.show()
    
    





