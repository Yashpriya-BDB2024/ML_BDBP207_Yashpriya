# Use your implementation and train ML models for california housing datasets and compare your results with the scikit-learn models.

### For California housing datasets -

from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

### Scikit-learn implementation -
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred_sk = model.predict(X_test_scaled)

mse_sk = mean_squared_error(y_test, y_pred_sk)
r2_sk = r2_score(y_test, y_pred_sk)
print("Scikit-learn implementation:")
print("MSE:", mse_sk)
print("R^2 score: %0.2f" % r2_sk)


### Gradient descent implementation -
# Add bias term to X for Gradient Descent
X_train_bias = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
X_test_bias = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

def hypothesis_func(X, theta):
    return np.dot(X, theta)

def cost_func(X, y, theta):
    hypoth = hypothesis_func(X, theta)
    cost = (1 / (2 * len(y))) * np.sum((hypoth - y) ** 2)
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
theta_gd, cost_history = gradient_descent(X_train_bias, y_train, learning_rate, num_of_iterations)

y_pred_gd = hypothesis_func(X_test_bias, theta_gd)
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

print("Gradient Descent implementation:")
print("MSE:", mse_gd)
print("R^2 score: %0.2f" % r2_gd)
