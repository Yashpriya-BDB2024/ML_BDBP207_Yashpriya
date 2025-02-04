# Compute SONAR classification results with and without data pre-processing (data normalization).
# Perform data pre-processing with your implementation and with scikit-learn methods and compare the results.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = pd.read_csv("sonar.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
label_encoder = LabelEncoder()    # R: 0, M: 1
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


### Without data pre-processing -

model_raw = LogisticRegression(max_iter=1000)
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)
accuracy_raw = accuracy_score(y_test, y_pred_raw)
print(f"Accuracy without pre-processing: {accuracy_raw:.2f}")


### With data-preprocessing -

### Manual implementation -
# Standardization
X_train_standardized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test_standardized = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
model_standardized = LogisticRegression(max_iter=1000)
model_standardized.fit(X_train_standardized, y_train)
y_pred_standardized = model_standardized.predict(X_test_standardized)
accuracy_standardized = accuracy_score(y_test, y_pred_standardized)
print(f"\nAccuracy after manual standardization: {accuracy_standardized:.2f}")

# Normalization
X_train_minmax_manual = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
X_test_minmax_manual = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0))
model_minmax_manual = LogisticRegression(max_iter=1000)
model_minmax_manual.fit(X_train_minmax_manual, y_train)
y_pred_minmax_manual = model_minmax_manual.predict(X_test_minmax_manual)
accuracy_minmax_manual = accuracy_score(y_test, y_pred_minmax_manual)
print(f"\nAccuracy after manual normalization: {accuracy_minmax_manual:.2f}")


### Using Scikit-learn -
# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"\nAccuracy after StandardScaler standardization: {accuracy_scaled:.2f}")

# Normalization
minmax_scaler = MinMaxScaler()
X_train_minmax_sk = minmax_scaler.fit_transform(X_train)
X_test_minmax_sk = minmax_scaler.transform(X_test)
model_minmax_sk = LogisticRegression(max_iter=1000)
model_minmax_sk.fit(X_train_minmax_sk, y_train)
y_pred_minmax_sk = model_minmax_sk.predict(X_test_minmax_sk)
accuracy_minmax_sk = accuracy_score(y_test, y_pred_minmax_sk)
print(f"\nAccuracy after MinMaxScaler normalization: {accuracy_minmax_sk:.2f}")
