# Implement a linear regression model using scikit-learn for the simulated dataset - simulated_data_multiple_linear_regression_for_ML.csv
# to predict the disease score from multiple clinical parameters.

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# loading the data
data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data.loc[:, 'age':'Gender']
y = data['disease_score']

# EDA
print(data)
print(data.describe())
print(data.isnull().sum())
data.hist(bins=30, figsize=(10, 8))
plt.suptitle("Feature distributions")
plt.plot()
plt.show()

# split data - train = 70%, test = 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

# scale the data
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train a model
print("-----TRAINING----")
# training a linear regression
model = LinearRegression()
# train the model
model.fit(X_train_scaled, y_train)

# prediction on a test set
y_pred = model.predict(X_test_scaled)
print("Predicted 'y' values are:")
print(y_pred)

# compute the r2 score (r2 score closer to 1 is considered to be good)
r2 = r2_score(y_test, y_pred)
print("r2 score (target = disease_score) is %0.2f" % r2)
print("Successfully done!")

# compute the mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")

# plot the scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual disease_score")
plt.ylabel("Predicted disease_score")
plt.plot()
plt.show()

# compute the coefficient values of all the features
# A positive coefficient means that as the feature increases, the target variable tends to increase, and vice versa for a negative coefficient.
# Features with higher absolute coefficients are usually considered more impactful.
theta_values=model.coef_
theta_0=model.intercept_
print(theta_0)
print(theta_values)


