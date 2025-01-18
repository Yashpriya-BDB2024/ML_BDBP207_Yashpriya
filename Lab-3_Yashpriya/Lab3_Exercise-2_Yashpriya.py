import numpy as np
import pandas as pd

# Read simulated data csv file -
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
print(data)

# Form X and y -
X = data.loc[:, 'age':'Gender']
print(X)
y = data['disease_score']
print(y)

# Feature scaling (Z-score normalization - mean centered around 0 & standard deviation centered around 1)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std
print(X_scaled)

# Adding the intercept (theta-0)
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
print(X_scaled)

# Write a function to compute hypothesis (h(x) = theta-0 + theta-1*x1 + theta-2*x2 + .....)
def hypothesis_func(X, theta):
    return np.dot(X, theta)

# Write a function to compute the cost -
def cost_func(X, y, theta):
    hypothes = hypothesis_func(X, theta)
    cost = 1/2 * np.sum((hypothes - y) ** 2)
    return cost

def main():
    print(hypothesis_func(X, theta))
    print(cost_func(X, y, theta))

if __name__ == "__main__":
    main()
