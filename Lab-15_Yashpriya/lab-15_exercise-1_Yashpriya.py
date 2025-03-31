### Implement Gradient Boost Regression using scikit-learn. Use the Boston housing dataset from the ISLP package for the regression problem.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('Boston.csv')    # Load the data

# Exploratory Data Analysis
print("Overview of the dataset: ")
print(df.info() )   # Overview of the dataset
print("Statistical summary of dataset: ")
print(df.describe())   # Statistical summary of the dataset
print("Unique values of dataset: ")
print(df.nunique())    # displays the unique values
print("Column names of dataset: ")
print(df.columns)    # gives feature column names
print("Duplicate values of dataset: ")
print(df.duplicated().sum())
print("Null values in the dataset: ")
print(df.isnull().sum())

df.hist(bins=30, figsize=(10,6))    # Feature distribution (histogram plot)
plt.show()
df.boxplot(figsize=(10,6))     # Outlier detection (Boxplot)
plt.show()
pd.plotting.scatter_matrix(df, figsize=(10,6))    # Scatter matrix - to visualize the pairwise relationships of the features
plt.show()

# Drop the unnecessary column -
df = df.drop(columns=["Unnamed: 0"])
print(df)

# Define features (X) and target variable (y) -
X = df.drop(columns=["medv"])
y = df["medv"]

# Split the data into 70% train set & 30% test set -
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)   # random_state=42 is set as seed for reproducibility.

# n_estimators: the number of boosting stages that will be performed, max_depth: limits the number of nodes in the tree
# learning_rate: how much the contribution of each tree will shrink, loss="squared_error": loss function to optimize (here, mean squared error).
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, loss="squared_error", random_state=42)
gb_reg.fit(X_train, y_train)    # Train the model

# Evaluating the model's performance -
y_pred = gb_reg.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")

# K-fold cross validation -
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_r2_scores = cross_val_score(gb_reg, X, y, cv=kf, scoring="r2")
print("R2 scores for each fold:", cv_r2_scores)
print("Mean R2 score:", cv_r2_scores.mean())
print("Standard Deviation of R2:", cv_r2_scores.std())

# Feature importance visualization -
plt.figure(figsize=(10, 6))
plt.barh(X.columns, gb_reg.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Gradient Boosting Regression")
plt.show()
