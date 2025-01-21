### Admission test dataset - normal equation -

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def hypothes_func(X, theta):
        return np.dot(X, theta)

def compute_cost(X, y, theta):
    error = hypothes_func(X, theta)-y.reshape(-1)
    transp = error.transpose()
    dot_pro = np.dot(transp, error)
    return 0.5 * dot_pro

def compute_theta(X, y):
    X_transpose = X.transpose()
    dot_pro1 = np.dot(X_transpose, X)
    inverse = np.linalg.inv(dot_pro1)
    dot_prod2 = np.dot(inverse, X_transpose)
    theta = np.dot(dot_prod2, y)
    return theta.reshape(-1)

train_data = pd.read_csv("Admission_Predict_Ver1.1.csv")
X = train_data.drop(['Chance of Admit ', 'Serial No.'], axis=1)        # Chance of Admit is the target variable
y = pd.DataFrame(data = train_data['Chance of Admit ']).to_numpy()
y = y*100
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 0)
print(X_train.head())
m_train=len(X_train)
print(m_train)
m_test=len(X_test)
print(m_test)
num_of_features=len(X_train.axes[1])+1
print(num_of_features)
ones_train=[1]*m_train     # Adding a column of x0=1
ones_test=[1]*m_test
X_train.insert(0, "X_0", ones_train, True)
X_test.insert(0, "X_0", ones_test, True)
theta_train = compute_theta(X_train, y_train)

cost_train = compute_cost(X_train, y_train, theta_train)
print("Training cost: ")
print(cost_train)
cost_test = compute_cost(X_test, y_test, theta_train)
print("Testing cost: ")
print(cost_test)
theta_train = compute_theta(X_train, y_train)
print("Parameters: ")
print(theta_train)
actual_y = y_test
print("Ground truth (actual 'y'): ")
print(actual_y)
pred_y = hypothes_func(X_test, theta_train)
print("Predicted 'y': ")
print(pred_y)
r2_closed_form = r2_score(actual_y, pred_y)
print("r^2 score: ")
print(r2_closed_form)
mse_closed_form = mean_squared_error(actual_y, pred_y)
print("MSE: ")
print(mse_closed_form)

### Comparison with scikit-learn -

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train.to_numpy()[:, 1:], y_train)
y_pred_train_sklearn = model.predict(X_test.to_numpy()[:, 1:])
print("Predicted 'y' (scikit-learn): ")
print(y_pred_train_sklearn)
mse_sklearn = mean_squared_error(y_test, y_pred_train_sklearn)
print("MSE (scikit-learn): ")
print(mse_sklearn)
r2_sklearn = r2_score(y_test, y_pred_train_sklearn)
print("r^2 score (scikit-learn): ")
print(r2_sklearn)
