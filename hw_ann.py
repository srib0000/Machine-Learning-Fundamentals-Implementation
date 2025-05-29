import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) = e^x - x^3
def f(x):
    return np.exp(x) - x**3

# Define the derivative of f(x)
def df(x):
    return np.exp(x) - 3*x**2

# Gradient Descent function with convergence criteria
def gradient_descent(learning_rate, max_updates, convergence_threshold, initial_x=0):
    x = initial_x
    x_values = []
    f_values = []

    for i in range(max_updates):
        x_values.append(x)
        f_values.append(f(x))
        gradient = df(x)
        x_new = x - learning_rate * gradient
        
        # Check for convergence
        if abs(x_new - x) < convergence_threshold:
            break
        
        x = x_new

    return x_values, f_values

# Set hyperparameters
learning_rate = 0.1
max_updates = 100
convergence_threshold = 1e-6

# Perform gradient descent with optional initial_x
x_values, f_values = gradient_descent(learning_rate, max_updates, convergence_threshold, initial_x=0)

# Plot the curve of f(x) versus update numbers
plt.plot(range(len(x_values)), f_values)
plt.xlabel("Number of Updates")
plt.ylabel("f(x)")
plt.title("Gradient Descent Convergence for f(x) = e^x - x^3")
plt.show()
