import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = np.loadtxt('diabetes.csv', delimiter=',')
[n, p] = np.shape(data)

num_train = int(0.75 * n)
num_test = int(0.25 * n)

sample_train = data[0:num_train, 0:-1]
sample_test = data[n - num_test:, 0:-1]
label_train = data[0:num_train, -1]
label_test = data[n - num_test:, -1]


def bootstrap_sample_custom(X, y, k):
    indices = np.random.choice(len(X), k, replace=True)
    return X[indices], y[indices]


def bagging_custom(X_train, y_train, X_test, y_test, base_model, m_values, k):
    testing_errors = []

    for m in m_values:
        base_models = []

        for _ in range(m):
     
            X_sample, y_sample = bootstrap_sample_custom(X_train, y_train, k)

           
            base_model_instance = base_model()
            base_model_instance.fit(X_sample, y_sample)

           
            base_models.append(base_model_instance)

    
        y_pred_ensemble = np.mean([model.predict(X_test) for model in base_models], axis=0)

       
        y_pred_ensemble_binary = np.round(y_pred_ensemble)

       
        testing_error = 1 - accuracy_score(y_test, y_pred_ensemble_binary)
        testing_errors.append(testing_error)

    return testing_errors

base_model = DecisionTreeClassifier
m_values = [1, 5, 10, 20, 30, 40, 50]
k = int(0.5 * len(sample_train))

testing_errors = bagging_custom(sample_train, label_train, sample_test, label_test, base_model, m_values, k)

plt.figure(figsize=(8, 6))
plt.plot(m_values, testing_errors, marker='o')
plt.title('Testing Error vs. Number of Base Models (Bagging) - Decision Tree')
plt.xlabel('Number of Base Models (m)')
plt.ylabel('Testing Error')
plt.show()
