### Implement Gradient Boost Classification using scikit-learn. Use the weekly dataset from the ISLP package and use Direction as the target variable for the classification.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Weekly.csv')    # Load the data

# Exploratory Data Analysis -
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

# Label Encoding -
df["Direction"] = LabelEncoder().fit_transform(df["Direction"])

# Define features (X) and target variable (y) -
X = df.drop(columns=["Direction"])
y = df["Direction"]

# Splitting the dataset (70% - train set, 30% - test set) -
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gb_clf.fit(X_train, y_train)    # Train the model

y_pred = gb_clf.predict(X_val)
print(classification_report(y_val, y_pred))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(gb_clf, X, y, cv=kf, scoring="accuracy")
print("Accuracy for each fold:", cv_accuracy)
print("Mean Accuracy:", cv_accuracy.mean())
print("Standard Deviation:", cv_accuracy.std())