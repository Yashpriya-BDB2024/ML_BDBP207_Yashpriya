### Write a Python program to aggregate predictions from multiple trees to output a final prediction for a regression problem.

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def aggregate_predictions(X, y, n_trees=40, max_depth=6, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    trees=[]    # to store the trained decision tree models
    for i in range(n_trees):
        # Generates a bootstrapped sample: randomly selects training data with replacement (some samples may repeat).
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_sample, y_sample = X_train[indices], y_train[indices]   # Uses the randomly selected indices to form a new training subset.
        tr = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state+i)   # Using the same random_state for all trees makes them behave similarly — so it's better to vary it using random_state + i.
        tr.fit(X_sample, y_sample)   # Trains the decision tree on the bootstrapped data.
        trees.append(tr)
    predictions = np.array([tr.predict(X_test) for tr in trees])   # For each trained tree, generate predictions on the same test set.
    final_pred = predictions.mean(axis=0)   # Averages the predictions of all trees across axis 0 (i.e., across all trees) to get the final ensemble prediction.
    r2 = r2_score(y_test, final_pred)
    print(f"r2 score: {r2}")
    mse = mean_squared_error(y_test, final_pred)
    print(f"MSE (aggregated tree regressor): {round(mse, 2)}")

def main():
    calif_housing =  fetch_california_housing()
    X, y = calif_housing.data, calif_housing.target
    aggregate_predictions(X, y)

if __name__ == "__main__":
    main()
