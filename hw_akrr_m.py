import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
file_path = 'crime.csv'
df = pd.read_csv(file_path)

# Separate the data into features (X) and the target variable (y)
X = df.iloc[:, :-1].values  # Exclude the last column as the target variable
y = df.iloc[:, -1].values   # The last column is the target variable

# train_test_split function
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int((1 - test_size) * len(indices))
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    return X_train, X_test, y_train, y_test

# Split the dataset into a training set (S) and a testing set (T)
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
gamma = 0.1  # RBF kernel parameter
m_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 1500]  # Values of m

# rbf_kernel function
def custom_rbf_kernel(X1, X2, gamma=1.0):
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    return K

# mean_squared_error function
def custom_mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Ridge class
class CustomRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        A = X.T @ (X * sample_weight[:, np.newaxis]) + self.alpha * np.eye(X.shape[1])
        b = X.T @ (y * sample_weight)
        self.weights = np.linalg.solve(A, b)

    def predict(self, X):
        return X @ self.weights

# Define lists to store testing errors for AKRR and KRR
akrr_errors = []
krr_errors = []

# Loop over different values of m
for m in m_values:
    # Compute the RBF kernel for the training data using custom function
    kernel_matrix = custom_rbf_kernel(X_train, X_train, gamma=gamma)
    
    akrr_predictions = []
    krr_predictions = []
    
    # Loop over each test point
    for i in range(len(X_test)):
        # Find the m nearest neighbors in the training set
        distances = np.sum((X_train - X_test[i])**2, axis=1)
        nearest_indices = np.argsort(distances)[:m]
        
        # Compute RBF kernel weights for AKRR
        akrr_weights = kernel_matrix[nearest_indices, i]
        
        # Train AKRR model for this test point
        akrr_model = CustomRidge(alpha=1e-6)  # You can adjust alpha (regularization strength)
        akrr_model.fit(X_train[nearest_indices], y_train[nearest_indices], sample_weight=akrr_weights)
        akrr_predictions.append(akrr_model.predict([X_test[i]])[0])
    
    # Calculate mean squared error for AKRR and KRR using custom function
    akrr_error = custom_mean_squared_error(y_test, akrr_predictions)
    akrr_errors.append(akrr_error)
    
    # Train KRR model for the same value of m (m=n)
    krr_model = CustomRidge(alpha=1e-6)
    krr_model.fit(X_train, y_train)
    krr_predictions = krr_model.predict(X_test)
    krr_error = custom_mean_squared_error(y_test, krr_predictions)
    krr_errors.append(krr_error)

# Print the testing errors for AKRR and KRR for each value of m
for m, akrr_error, krr_error in zip(m_values, akrr_errors, krr_errors):
    print(f"m = {m}: AKRR Testing Error = {akrr_error}, KRR Testing Error = {krr_error}")

# Plot the testing errors versus m for AKRR and KRR
plt.plot(m_values, akrr_errors, label='AKRR')
plt.plot(m_values, krr_errors, label='KRR')
plt.xlabel('m')
plt.ylabel('Testing Error')
plt.legend()
plt.grid(True)
plt.show()
