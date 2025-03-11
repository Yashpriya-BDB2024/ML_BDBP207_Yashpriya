### Implement a classification decision tree algorithm using scikit-learn for the sonar dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("sonar.csv")

# print("Overview: ")
# print(data.info())
# print("Missing / null values: ")
# print(data.isnull().sum())
# print("Duplicate rows: ")
# print(data.duplicated().sum())

X = data.iloc[:, :-1].values     # All columns except the last one
y = data.iloc[:, -1].values      # Last column (target variable)

# Encode the target variable (R/M) to numerical values
label_encoder = LabelEncoder()    # R: 0, M: 1
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=99)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, rounded=True, fontsize=10,class_names=label_encoder.classes_)      #class_names=label_encoder.classes_: assigns original class labels (R, M) to the decision tree nodes
plt.show()
