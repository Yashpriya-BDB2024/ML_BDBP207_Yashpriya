### PLOT FOR 'SIMULATED DATASETS' (X='age', y='disease_score_fluct')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data['age'].values.reshape(-1, 1)
y = data['disease_score_fluct'].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=999)


### Using Gradient Descent -

def hypothesis_func(X, theta):
    return np.dot(X, theta)

def cost_func(X, y, theta):
    hypothes = hypothesis_func(X, theta)
    cost = (1 / (2 * len(y))) * np.sum((hypothes - y) ** 2)
    return cost

def compute_gradient(X, y, theta):
    m = len(y)
    hypoth = hypothesis_func(X, theta)
    gradient = (1 / m) * np.dot(X.T, (hypoth - y))
    return gradient

def gradient_descent(X, y, learning_rate, num_of_iterations):
    theta = np.zeros((X.shape[1], 1))
    cost_history = []
    for i in range(num_of_iterations):
        gradient = compute_gradient(X, y, theta)
        theta -= learning_rate * gradient
        cost = cost_func(X, y, theta)
        cost_history.append(cost)
    return theta, cost_history

learning_rate = 0.01
num_of_iterations = 1000
theta_gd, cost_history = gradient_descent(X_train, y_train, learning_rate, num_of_iterations)
y_pred_gd = hypothesis_func(X_test, theta_gd)
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)
print("Gradient Descent - MSE:", mse_gd)
print("Gradient Descent - R^2 score:", r2_gd)


### Using Scikit-learn -

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sk = model.predict(X_test)
mse_sk = mean_squared_error(y_test, y_pred_sk)
r2_sk = r2_score(y_test, y_pred_sk)
print("Scikit-learn - MSE:", mse_sk)
print("Scikit-learn - R^2 score:", r2_sk)


### Using Normal Equation -

def compute_theta(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]      # Add bias term to X (since normal equation requires it)
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]
theta_closed_form = compute_theta(X_train_bias, y_train)
y_pred_closed_form = hypothesis_func(X_test_bias, theta_closed_form)
mse_closed_form = mean_squared_error(y_test, y_pred_closed_form)
r2_closed_form = r2_score(y_test, y_pred_closed_form)
print(f"Closed-form coefficients: {theta_closed_form.ravel()}")
print(f"Closed-form MSE: {mse_closed_form}")
print(f"Closed-form R^2: {r2_closed_form}")
