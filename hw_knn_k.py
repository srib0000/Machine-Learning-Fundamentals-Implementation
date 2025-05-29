import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
file_path = 'diabetes.csv'
df = pd.read_csv(file_path)

# Separate the data into features (X) and the target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train-test split function with a random seed for reproducibility
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(X)
    num_test_samples = int(test_size * num_samples)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    X_train = X[indices[num_test_samples:]]
    y_train = y[indices[num_test_samples:]]
    X_test = X[indices[:num_test_samples]]
    y_test = y[indices[:num_test_samples]]

    return X_train, X_test, y_train, y_test

# Compressed k-Nearest Neighbors classifier (CkNN)
class CompressedKNeighborsClassifier:
    def __init__(self, n_neighbors=5, compression_ratio=1.0, random_state=None):
        self.n_neighbors = n_neighbors
        self.compression_ratio = compression_ratio
        self.random_state = random_state

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        num_instances = len(X)
        num_selected = int(self.compression_ratio * num_instances)
        indices = np.random.choice(num_instances, num_selected, replace=False)
        self.X_train = X[indices]
        self.y_train = y[indices]

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            y_pred.append(most_common)
        return np.array(y_pred)

# Accuracy score calculation
def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the CkNN classifier for different values of k and τ=1 (no compression)
k_values = [1, 5, 17, 29, 32, 37, 40, 45]
tau = 1.0  # No compression

testing_errors = []

for k in k_values:
    cknn = CompressedKNeighborsClassifier(n_neighbors=k, compression_ratio=tau, random_state=42)
    cknn.fit(X_train, y_train)
    y_pred = cknn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    testing_error = 1 - accuracy
    testing_errors.append(testing_error)

# Plot the testing errors
plt.figure(figsize=(10, 6))
plt.plot(k_values, testing_errors, marker='o', linestyle='-')
plt.title('Testing Error vs. K (τ=1)')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Testing Error')
plt.grid(True)
plt.show()
