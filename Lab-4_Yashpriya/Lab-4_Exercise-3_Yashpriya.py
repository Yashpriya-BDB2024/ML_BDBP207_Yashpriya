# Implement normal equations method from scratch and compare your results on a simulated dataset (disease score fluctuation as target)
# and the admissions dataset (https://www.kaggle.com/code/erkanhatipoglu/linear-regression-using-the-normal-equation ).
# You can compare the results with scikit-learn and your own gradient descent implementation.

import pandas as pd
import numpy as np

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
# def design_matrix(data):
#     X = data.drop(columns=['disease_score', 'disease_score_fluct'])
#     X_transpose = np.transpose(X)
#     return X_transpose
# y = data['disease_score_fluct'].values
# print(y)

def hypothesis_func(theta, X):
    hypothes = np.dot(np.transpose(theta), X)
    return hypothes

def cost_func(X, theta, y):
    cost = 0.5 * np.transpose((X*theta - y)) * (X*theta - y)
    return cost

def compute_gradient(X, theta, y):
    gradient = np.transpose(X) * X * theta - np.transpose(X) * y
    return gradient

def compute_theta(X, y):
    theta = (np.transpose(X) * y) / (np.transpose(X) * X)
    return theta

def update_param():


# def main():
#     print(design_matrix(data))
# if __name__ == "__main__":
#     main()
