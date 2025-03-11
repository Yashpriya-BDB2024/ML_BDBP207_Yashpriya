### Implement a regression decision tree algorithm using scikit-learn for the simulated dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree    # Importing decision tree regressor and visualization tool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score     # For evaluating model performance

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")    # Load the dataset from a CSV file

X = data.drop(columns=["disease_score_fluct", "disease_score"])      # Define feature matrix (X) by dropping the target variable(s)
y = data["disease_score"]        # Define target variable (y) - 'disease_score' as the dependent variable

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)      # Setting random_state ensures reproducibility, meaning the model will produce the same results every time it's run.   

# Initialize the Decision Tree Regressor with a maximum depth of 3 for better interpretability
regressor = DecisionTreeRegressor(max_depth=3, random_state=99)      # A smaller max_depth prevents overfitting, ensuring the model generalizes well to new data.
regressor.fit(X_train, y_train)      # Train the decision tree model on the training dataset

y_pred = regressor.predict(X_test)    # Predict the target variable on the test dataset
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)     # Calculate R^2 score to evaluate how well the model explains the variance in the target variable
print(f"r2 score: {r2}")

# Visualization of regression decision tree -
plt.figure(figsize=(20, 10))    # Set figure size for better readability
plot_tree(regressor, feature_names=X.columns, filled=True, rounded=True, fontsize=10)     # Plot the tree structure
# feature_names=X.columns: Labels for features in the tree nodes  
# filled=True: Colors the nodes based on prediction values  
# rounded=True: Rounds the edges of nodes for better visualization  
# fontsize=10: Sets the font size of text inside nodes
plt.show()     # Display the plot
