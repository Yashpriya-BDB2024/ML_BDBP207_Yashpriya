# Perform 10-fold cross validation for SONAR dataset in scikit-learn using logistic regression.
# SONAR dataset is a binary classification problem with target variables as Metal or Rock. i.e. signals are from metal or rock.

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv("sonar.csv")
print(data)

print("Overview: ")
print(data.info())
print("Missing / null values: ")
print(data.isnull().sum())
print("Duplicate rows: ")
print(data.duplicated().sum())

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

label_encoder = LabelEncoder()    # R: 0, M: 1
y_encoded = label_encoder.fit_transform(y)

pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
cv_scores = cross_val_score(pipeline, X, y_encoded, cv=10, scoring='accuracy')
print(cv_scores.mean())
print(cv_scores.std())
print(f"Accuracy of 10-folds are: {cv_scores * 100} %")
