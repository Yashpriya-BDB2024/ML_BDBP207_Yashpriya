# Implement gradient descent algorithm from scratch using Python.
# Will be imported in exercise-2
import numpy as np

def gradient_descent(X, y, learning_rate, n_iterations):
    num_of_samples, num_of_features = X.shape
    thetas = np.zeros(num_of_features)
    cost_hist = []
    for i in range(n_iterations):
        y_pred = np.dot(X, thetas)
        error = y_pred -y
        wt_grad = 2*np.dot(X.T, error)
        wt -= learning_rate * wt_grad
        cost = 0.5 * np.sum(error ** 2)
        cost_hist.append(cost)
    return thetas, cost_hist
