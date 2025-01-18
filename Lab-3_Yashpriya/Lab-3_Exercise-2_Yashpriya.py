import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read simulated data csv file -
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
#print(data)

# Form X and y -
X = data.loc[:, 'age':'Gender'].values
#print(X)
y = data['disease_score'].values.reshape(-1, 1)
#print(y)

# Feature scaling (Z-score normalization - mean centered around 0 & standard deviation centered around 1)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std
#print(X_scaled)

# Adding the intercept (theta-0)
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
#print(X_scaled)

# Write a function to compute hypothesis (h(x) = theta-0 + theta-1*x1 + theta-2*x2 + .....)
def hypothesis_func(X, theta):
    return np.dot(X, theta)

# Write a function to compute the cost -
def cost_func(X, y, theta):
    hypothes = hypothesis_func(X, theta)
    cost = 1/2 * np.sum((hypothes - y) ** 2)
    return cost

# Write a function to compute the partial derivative w.r.t. theta (gradient) -
def compute_gradient(X, y, theta):
    hypoth = hypothesis_func(X, theta)
    derivat = np.zeros_like(theta)
    for i in range(len(y)):
        for j in range(X.shape[1]):
            derivat[j] += (hypoth[i] - y[i]) * X[i, j]
    return derivat

def main():
    # Write update parameters logic in the main function -
    theta = np.zeros((X_scaled.shape[1], 1))
    learning_rate=0.1
    num_of_iterations=1000
    cost_history = []
    for i in range(num_of_iterations):
        gradient = compute_gradient(X_scaled, y, theta)
        theta -= learning_rate * gradient
        cost = cost_func(X_scaled, y, theta)
        cost_history.append(cost)
        print(f"Iteration-{i}: cost = {cost}, parameters = {theta}")
    print(theta)

    # Plot to see the convergence -
    plt.plot(range(num_of_iterations), cost_history, color='blue')
    plt.title("Convergence plot")
    plt.xlabel("No. of iterations")
    plt.ylabel("cost")
    plt.plot()
    plt.show()

if __name__ == "__main__":
    main()
