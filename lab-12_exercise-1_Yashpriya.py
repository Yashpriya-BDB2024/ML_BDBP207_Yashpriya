### Implement a decision regression tree algorithm without using scikit-learn using the diabetes dataset. Fetch the dataset from scikit-learn library.

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

def load_data_and_eda():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    # print("Shape of dataset:", df.shape)
    # print("Missing values:", df.isnull().sum().sum())
    # print("Duplicated rows:", df.duplicated().sum())
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_thresholds(X):    # Partition logic
    sorted_features = np.sort(X, axis=0)    # Sorts each feature column independently.
    threshold = (sorted_features[:-1] + sorted_features[1:]) / 2    # Computes midpoints between every consecutive pair of values.
    return threshold

def sse(y):
    return np.sum((y-np.mean(y))**2) if len(y)>0 else 0

def BuildTree(I, K, X, y):
    if len(I) < K:     # Termination condition: If no. of data points (|I|) less than no. of nodes (K)
        y_hat = np.mean(y[I])    # Compute the average target value
        return y_hat    # return a leaf node
    # Initialization part -
    best_feature, best_threshold = None, None
    min_error = float('inf')
    thresholds=get_thresholds(X[I])
    for j in range(X.shape[1]):    # Iterate over all the features
        for s in thresholds[:, j]:    # Iterate over all the possible split values
            I_plus = I[X[I, j]>=s]     # Indices where j >= s
            I_minus = I[X[I, j]<s]     # Indices where j < s
            if len(I_plus)==0 or len(I_minus)==0:    # Ensure valid split
                continue
            y_plus, y_minus = np.mean(y[I_plus]), np.mean(y[I_minus])
            total_error=sse(y[I_plus])+sse(y[I_minus])    # Compute the sum of squared error (SSE)
            if total_error < min_error:
                min_error = total_error
                best_feature, best_threshold = j, s    # Selecting the optimum feature with optimum threshold value bec. of the least error.
    if best_feature is None:
        return np.mean(y[I])
    # Recursively build left & right subtrees
    I_plus = I[X[I, best_feature] >= best_threshold]
    I_minus = I[X[I, best_feature] < best_threshold]
    return {"feature": best_feature, "threshold": best_threshold, "left": BuildTree(I_minus, K, X, y), "right": BuildTree(I_plus, K, X, y)}

def predict_one(x, tree):
    while isinstance(tree, dict):    # Keep moving down until reaching a leaf
        if x[tree["feature"]] < tree["threshold"]:
            tree = tree["left"]   # Move left of the tree
        else:
            tree = tree["right"]   # Move right of the tree
    return tree   # Return the final leaf value

def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])   # Apply to all the samples

def main():
    X_train, X_test, y_train, y_test = load_data_and_eda()     # Data splitting
    I_train=np.arange(len(y_train))
    K=10       # Set minimum node size
    reg_tree=BuildTree(I_train, K, X_train, y_train)   # Build the regression tree
    y_pred = predict(X_test, reg_tree)
    print(f"Predicted y-values: {y_pred}")
    print(f"SSE: {sse(y_test - y_pred)}")
    print(f"r^2 score: {r2_score(y_test, y_pred)}")

if __name__ == "__main__":
    main()