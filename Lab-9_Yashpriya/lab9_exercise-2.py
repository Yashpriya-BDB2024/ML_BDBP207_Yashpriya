# Implement a regression decision tree algorithm using scikit-learn for the simulated dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

X = data.drop(columns=["disease_score_fluct", "disease_score"])
y = data["disease_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

regressor = DecisionTreeRegressor(max_depth=3, random_state=99)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"r2 score: {r2}")

# Visualization of regression decision tree -
plt.figure(figsize=(20, 10))
plot_tree(regressor, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.show()


