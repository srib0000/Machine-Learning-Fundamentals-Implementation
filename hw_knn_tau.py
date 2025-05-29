import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
file_path = 'diabetes.csv'
df = pd.read_csv(file_path)

# Separate the data into features (X) and the target variable (y)
X = df.iloc[:, :-1].values  # Exclude the last column as the target variable
y = df.iloc[:, -1].values   # The last column is the target variable

# train-test split function
def train_test_split(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    num_samples = X.shape[0]
    num_test_samples = int(test_size * num_samples)
    
    # Shuffle the data
    shuffled_indices = np.random.permutation(num_samples)
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    # Split the data into training and testing sets
    X_train = X_shuffled[:-num_test_samples]
    y_train = y_shuffled[:-num_test_samples]
    X_test = X_shuffled[-num_test_samples:]
    y_test = y_shuffled[-num_test_samples:]
    
    return X_train, X_test, y_train, y_test

# k-Nearest Neighbors classifier for CkNN
def c_k_neighbors_classifier(X_train, y_train, X_test, k=5, tau=1.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    y_pred = np.zeros(num_test_samples, dtype=int)
    
    # Randomly select a fraction τ of training instances
    selected_indices = np.random.choice(num_train_samples, int(tau * num_train_samples), replace=False)
    X_train_selected = X_train[selected_indices]
    y_train_selected = y_train[selected_indices]
    
    for i in range(num_test_samples):
        distances = np.sqrt(np.sum((X_train_selected - X_test[i])**2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train_selected[nearest_indices]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        y_pred[i] = unique_labels[np.argmax(counts)]
    
    return y_pred

# accuracy score calculation
def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Calculate testing error for different values of τ and k
k_values = [1, 50, 100]  # Pick three values of k
tau_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pick 10 values of τ

testing_errors = []

for k in k_values:
    errors_k = []
    for tau in tau_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        y_pred = c_k_neighbors_classifier(X_train, y_train, X_test, k, tau)
        accuracy = accuracy_score(y_test, y_pred)
        error = 1 - accuracy  # Testing error
        errors_k.append(error)
    testing_errors.append(errors_k)

# Plot the testing errors for different values of k
plt.figure(figsize=(10, 6))
for i, k in enumerate(k_values):
    plt.plot(tau_values, testing_errors[i], label=f'k={k}')

plt.xlabel('τ (Fraction of Training Data Used)')
plt.ylabel('Testing Error')
plt.title('CkNN Classifier Testing Error vs. τ for Different Values of k')
plt.legend()
plt.grid(True)
plt.show()
