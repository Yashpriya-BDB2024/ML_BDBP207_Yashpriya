# Use your implementation and train ML models for simulated datasets and compare your results with the scikit-learn models.

### For simulated datasets -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load and pre-process data -
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data[['age', 'BMI', 'BP', 'blood_sugar', 'Gender']].values
y_disease_score = data['disease_score'].values.reshape(-1, 1)
y_disease_score_fluct = data['disease_score_fluct'].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

# Split the data -
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_disease_score, test_size=0.3, random_state=999)
# for 'disease_score_fluct'
X_train_dsf, X_test_dsf, y_train_dsf, y_test_dsf = train_test_split(X_scaled, y_disease_score_fluct, test_size=0.3, random_state=999)

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

# Train the gradient descent model -
learning_rate = 0.01
num_of_iterations = 1000
theta_gd, cost_history = gradient_descent(X_train, y_train, learning_rate, num_of_iterations)     # for disease_score
theta_gd_dsf, cost_history_dsf = gradient_descent(X_train_dsf, y_train_dsf, learning_rate, num_of_iterations)    # for disease_score_fluct

# Predictions for 'disease_score' -
y_pred_gd = hypothesis_func(X_test, theta_gd)
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

# Predictions for 'disease_score_fluct' -
y_pred_gd_dsf = hypothesis_func(X_test_dsf, theta_gd_dsf)
mse_gd_dsf = mean_squared_error(y_test_dsf, y_pred_gd_dsf)
r2_gd_dsf = r2_score(y_test_dsf, y_pred_gd_dsf)

print("Simulated dataset (target = disease_score): ")
print("Gradient Descent - MSE:", mse_gd)
print("Gradient Descent - R^2 score:", r2_gd)
print("Simulated dataset (target = disease_score_fluct): ")
print("Gradient Descent - MSE:", mse_gd_dsf)
print("Gradient Descent - R^2 score:", r2_gd_dsf)

# Train the scikit-learn model (for disease_score) -
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sk = model.predict(X_test)
mse_sk = mean_squared_error(y_test, y_pred_sk)
r2_sk = r2_score(y_test, y_pred_sk)

# Scikit-learn Linear Regression (for disease_score_fluct) -
model_dsf = LinearRegression()
model_dsf.fit(X_train_dsf, y_train_dsf)
y_pred_sk_dsf = model_dsf.predict(X_test_dsf)
mse_sk_dsf = mean_squared_error(y_test_dsf, y_pred_sk_dsf)
r2_sk_dsf = r2_score(y_test_dsf, y_pred_sk_dsf)

print("Simulated dataset (target = disease_score): ")
print("Scikit-learn - MSE:", mse_sk)
print("Scikit-learn - R^2 score:", r2_sk)
print("Simulated dataset (target = disease_score_fluct): ")
print("Scikit-learn - MSE:", mse_sk_dsf)
print("Scikit-learn - R^2 score:", r2_sk_dsf)
