# Implement a classification decision tree algorithm using scikit-learn for the breast cancer dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("wdbc.data", header=None)
data.drop(columns=[0], inplace=True)    # Drop the ID column
label_encoder = LabelEncoder()
data[1] = label_encoder.fit_transform(data[1])     # M: 1, B: 0

X = data.drop(columns=[1])    # Drop Diagnosis column
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=99)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(classifier, filled=True, rounded=True, fontsize=10)
plt.show()



