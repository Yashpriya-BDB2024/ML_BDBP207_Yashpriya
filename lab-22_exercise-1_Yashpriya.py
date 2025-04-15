### Work on NCI data - build classification model after reducing the gene expression features using hierarchical clustering. Compare this with the PCA approach.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ISLP import load_data

# Load NCI60 dataset
nci_data = load_data("NCI60")   # contains gene expression data for 60 cancer cell lines.
# This dataset is a dictionary with 'data' and 'labels'
X_raw = nci_data['data']
y_raw = nci_data['labels']
# Convert to DataFrame
X = pd.DataFrame(X_raw, columns=[f"gene_{i}" for i in range(X_raw.shape[1])])   # This generates column names for each gene feature; X_raw.shape[1] gets the no. of columns (features, i.e., genes)
y = pd.Series(y_raw.values.flatten(), name="label")  # .flatten(): reshapes the numpy array into a 1D array, ensuring it can be used properly as a target variable, and then converted into a Pandas series.

# Train-test split (70% train set, 30% test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data after splitting (to avoid data leakage, i.e., the model has access to info. from the test set during training, which leads to overfitting.)
# Apply scaling (or normalization) only on the training data, and use the parameters (mean, standard deviation, etc.) derived from the training data to scale the test data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on the training data
X_test_scaled = scaler.transform(X_test)  # Only transform on the test data

# Hierarchical Clustering Approach -
# n_clusters=5: to reduce the data into 5 clusters based on gene expression patterns.
hc = AgglomerativeClustering(n_clusters=5, linkage='complete')   # Complete linkage considers max. distance b/w clusters
hc_labels = hc.fit_predict(X_train_scaled)   # fits the model and returns cluster labels for each sample.

# Logistic Regression on Hierarchical Clustering Features
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
accuracy_hc = accuracy_score(y_test, y_pred)
print(f"Accuracy using hierarchical clustering features: {accuracy_hc*100:.2f} %")
print("Confusion matrix for hierarchical clustering: ")
print(confusion_matrix(y_test, y_pred))
print("Classification report for hierarchical clustering: ")
print(classification_report(y_test, y_pred, zero_division=0))   # zero_division=0: set precision/recall to 0 for those classes where predictions are missing.

# PCA Approach -
pca = PCA(n_components=10)   # to reduce the data to 10 components that retain most of the variance in the original data.
X_train_pca = pca.fit_transform(X_train_scaled)   # Apply PCA to the training data
X_test_pca = pca.transform(X_test_scaled)   # Apply the same PCA transformation to test data

# Logistic Regression on PCA Features
clf_pca = LogisticRegression(max_iter=5000)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy using PCA features: {accuracy_pca*100:.2f} %")
print("Confusion matrix for PCA approach: ")
print(confusion_matrix(y_test, y_pred_pca))
print("Classification report for PCA approach: ")
print(classification_report(y_test, y_pred_pca, zero_division=0))

# Comparison -
if accuracy_hc > accuracy_pca:
    print("\nHierarchical Clustering performed better than PCA for Logistic Regression classification.")
else:
    print("\nPCA performed better than Hierarchical Clustering for Logistic Regression classification.")
