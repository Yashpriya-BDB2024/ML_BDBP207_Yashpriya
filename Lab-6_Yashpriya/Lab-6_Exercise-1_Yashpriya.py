### K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.

# Using scikit-learn -

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv("breast_cancer_data.csv")
data = data.drop(['id', 'Unnamed: 32'], axis=1)

encoder = OneHotEncoder(sparse_output=False, drop='first')    # Creating the OneHotEncoder instance
diagnosis_reshaped = data['diagnosis'].values.reshape(-1, 1)    # Reshape it to 2D
encoded = encoder.fit_transform(diagnosis_reshaped)
data['Malignant'] = encoded[:, 0]    # Adding the encoded column back to the dataset
data = data.drop('diagnosis', axis=1)

y = data['Malignant']
X = data.drop('Malignant', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_reg = LogisticRegression(max_iter=1000)

scores = cross_val_score(log_reg, X_scaled, y, cv=10, scoring='accuracy')
print(f"Accuracy of 10-folds are: {scores * 100} %")

# From scratch -

