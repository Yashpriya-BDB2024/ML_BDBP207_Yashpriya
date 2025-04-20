import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

def bagging_scratch(X_train, y_train, t):   # Takes X_train (features), y_train (targets), and t (no. of trees) , & returns t decision trees trained on bootstrapped datasets.
    n = X_train.shape[0]   # n is the no. of training examples (rows).
    trees = []  # Initialize an empty list to store all decision trees.
    for i in range(t):  # Loop t times to build t tree.
        bootstrap_sample = X_train.sample(n, replace=True)   # Creates a bootstrap sample of size n (same as original), and sampling with replacement means some rows may repeat.
        y_bootstrap = y_train.loc[bootstrap_sample.index]   # Gets the corresponding y values for the bootstrap sample using .loc
        tree = build_tree(bootstrap_sample, y_bootstrap)   # Calls build_tree() on this bootstrapped dataset.
        trees.append(tree)  # Adds the built tree to the list.
    return trees   # Returns the list of all trees.

def build_tree(X, y, k=10):   # k is the minimum no. of samples to stop splitting further.
    i = len(y)  # i is the no. of samples.
    # These variables store the best split found during the loop.
    min_error = float('inf')   # the smallest sum of squared errors
    best_feature = None   # feature column to split on
    best_threshold = None   # value to split at
    best_split = None  # indices for left and right subtrees
    if i <= k:
        return np.mean(y)   # If the no. of samples is small, stop splitting & return the average target value — a leaf node.
    for feature in X.columns:   # Loop through each feature (column).
        thresholds = np.unique(X[feature])  # Get all unique values of this feature to try as split points.
        for threshold in thresholds:   # Try every threshold as a split point.
            left_indices = X[feature] < threshold
            right_indices = X[feature] >= threshold
            if sum(left_indices) == 0 or sum(right_indices) == 0:  # Skip splits where all data goes to one side.
                continue
            # Predictions = mean of y values in each group (since it's regression).
            y_left_pred = np.mean(y[left_indices])
            y_right_pred = np.mean(y[right_indices])
            error = sum((y_left_pred - y[left_indices]) ** 2) + sum((y_right_pred - y[right_indices]) ** 2)  # Compute sum of squared errors (SSE) for this split
            if error < min_error:  # Lower SSE means better split ; if this split is better than any previous one, update the best values.
                min_error = error
                best_feature = feature
                best_threshold = threshold
                best_split = left_indices, right_indices
    if best_feature is None:   # If no good split found, then return a leaf with the average target.
        return np.mean(y)
    left_indices, right_indices = best_split   # Unpack the saved split.
    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": build_tree(X.loc[left_indices].reset_index(drop=True), y.loc[left_indices].reset_index(drop=True), k),   # Recursively build the left and right subtrees.
        "right": build_tree(X.loc[right_indices].reset_index(drop=True), y.loc[right_indices].reset_index(drop=True), k),   # reset_index(drop=True) cleans up row indices.
    }

def predict(tree, sample):   # Takes a tree and one sample (row) and returns a prediction.
    if isinstance(tree, dict):   # If not a leaf node, it's a dictionary.
        feature, threshold = tree["feature"], tree["threshold"]   # Get the feature and threshold to split on.
        if sample[feature] < threshold:  # Traverse the left or right subtree based on feature value.
            return predict(tree["left"], sample)
        else:
            return predict(tree["right"], sample)
    return tree   # If it’s a leaf (no.), return it directly.

def bagging_predict(trees, X_test):   # Predicts on every row in X_test using every tree.
    predictions = np.array([   # For each tree, get predictions for all rows.
        X_test.apply(lambda row: predict(tree, row), axis=1) for tree in trees
    ])
    prediction = predictions.mean(axis=0)  # Average predictions from all trees.
    return prediction

def diabetes_loading_preprocessing():
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    return X, y

def kfold_bagging(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        trees = bagging_scratch(X_train, y_train, t=10)
        y_pred = bagging_predict(trees, X_val)
        scores.append(r2_score(y_val, y_pred))
    return np.mean(scores)

def main():
    X, y = diabetes_loading_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    avg_scores = kfold_bagging(X_train, y_train, n_splits=10)
    print(f"R2 score for Bagging with Decision Trees (K-Fold) from scratch: {avg_scores:.4f}")

if __name__ == "__main__":
    main()