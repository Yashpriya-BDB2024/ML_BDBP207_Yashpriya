### Implement XGBoost classifier and regressor using scikit-learn.

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, classification_report

# Classifier: Iris dataset
iris = load_iris()
X_cls, y_cls = iris.data, iris.target    # X: features, y: labels (0, 1, 2 for 3 flower types)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)    # 70% train, 30% test
# n_estimators: total no. of boosting trees to train, max_depth: controls overfitting, learning_rate: step size shrinkage used in update to prevent overfitting
# eval_metric='mlogloss': metric for multi-class classification (logarithmic loss)
clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='mlogloss')
clf.fit(X_train_cls, y_train_cls)   # train the classifier on training data
y_pred_cls = clf.predict(X_test_cls)   # make predictions on the test set
acc = accuracy_score(y_test_cls, y_pred_cls)
print("XGBoost Classifier Accuracy (Iris):", acc*100,"%")
print(classification_report(y_test_cls, y_pred_cls))

# Regressor: California housing dataset
calif_housing = fetch_california_housing()
X_reg, y_reg = calif_housing.data, calif_housing.target
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print("r2 score (XGBoost Regressor): ", r2)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print("XGBoost Regressor MSE (California housing):", round(mse, 2))