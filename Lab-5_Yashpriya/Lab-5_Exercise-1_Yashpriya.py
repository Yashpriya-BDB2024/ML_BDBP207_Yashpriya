# Implement Stochastic Gradient Descent algorithm from scratch.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.loc[:, 'age':'Gender'].values
y = data['disease_score'].values.reshape(-1, 1)

# Feature scaling (Z-score normalization)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=999)

# Adding the intercept
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def hypothesis_func(X, theta):
    return np.dot(X, theta)

def compute_gradient(X_index, y_index, theta):
    prediction = hypothesis_func(X_index, theta)
    error = prediction - y_index
    gradient = X_index.T.dot(error)
    return gradient

def compute_cost(X_index, y_index, theta):
    prediction = hypothesis_func(X_index, theta)
    cost = (1 / 2) * (prediction - y_index) ** 2
    return cost

def stochastic_gradient_func(X, y, learning_rate, num_of_iterations):
    theta = np.random.randn(X.shape[1], 1) * 0.01      # small random initialization
    costs = []
    for i in range(num_of_iterations):
        for j in range(X.shape[0]):
            rand_index = np.random.randint(0, X.shape[0])
            X_index = X[rand_index, :].reshape(1, -1)
            y_index = y[rand_index].reshape(1, 1)
            gradient = compute_gradient(X_index, y_index, theta)
            theta = theta - learning_rate * gradient
            cost = np.mean((hypothesis_func(X, theta) - y) ** 2)
            costs.append(cost)
    return theta, costs

def main():
    theta_trained, costs = stochastic_gradient_func(X_train, y_train, learning_rate=0.001, num_of_iterations=1000)
    y_pred = hypothesis_func(X_test, theta_trained)
    mse = np.mean((y_pred - y_test) ** 2)
    print(mse)
    r2 = r2_score(y_test, y_pred)
    print(r2)

    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Ground truth (y)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('y v/s y_pred')
    plt.plot()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(costs)), costs, color='blue', label='Cost Function (Convergence)')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('SGD Convergence: Cost vs Iterations')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
