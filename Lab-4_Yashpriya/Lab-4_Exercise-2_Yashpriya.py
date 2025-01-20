# Use your implementation and train ML models for both california housing and simulated datasets
# and compare your results with the scikit-learn models.

### Simulated datasets (if target = disease_score) -

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Lab_4_Exercise_1_Yashpriya import gradient_descent

# load the data -
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data[['age', 'BMI', 'BP', 'blood_sugar', 'Gender']].values
y = data['disease_score'].values

# Split the data [70% - training, 30% - test] -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

# Train the model using gradient descent algorithm -
wt_gd, cost_gd = gradient_descent(X_train, y_train, learning_rate=0.1, n_iterations=1000 )

# Predict the model (from gradient descent) -
y_pred_gd = np.dot(X_test, wt_gd)
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

# Train the model using scikit-learn -
from sklearn.linear_model import LinearRegression
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

# Comparison b/w gradient descent & scikit-learn models -
print("Gradient descent - MSE: ", mse_gd)
print("Gradient descent - R^2 score:", r2_gd)
print("Scikit-learn model - MSE: ", mse_sklearn)
print("Scikit-learn model - R^2 score:", r2_sklearn)
