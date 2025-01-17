from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def load_data():
    # X: feature matrix (predictor variables) like income, house age, etc.
    # Y: target variable like median house value
    # This function uses fetch_california_housing dataset.
    # return_X_y=True: directly returns X and y as separate numpy arrays.
    [X, y] = fetch_california_housing(return_X_y=True)
    return (X, y)

def main():
    # load california housing data (calls the function to get features X and target y)
    [X, y] = load_data()

    # train_test_split(X,y,...): splits the dataset into training (to train the model) and testing sets (to evaluate the model performance on unseen data).
    # test_size=0.30: Specifies that 30% of the data will be used for testing, and the remaining 70% for training.
    # random_state=999: Sets a random seed to ensure the data split is reproducible. So, using the same random state will always produce the same split.
    # Thus, variables are created, i.e., X_train (70% of X), X_test (30% of X), y_train (70% of y) and y_test (30% of y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    # Scaling - gradient descent converges faster, improves model performance & prevent bias towards features.
    # scaler=StandardScaler(): standardizes features by removing the mean & scaling to unit variance (X_scaled = (X - mean) / standard deviation)
    # scaler.fit(X_train): Calculates the mean & standard deviation for each feature only on the training set.
    # scaler.transform(X_train): This ensures that all the features in X_train_scaled have a mean of 0 and a standard deviation of 1.
    # scaler.transform(X_test): Uses the training data's mean & standard deviation to scale the test set. This ensures fair model evaluation.
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("-----TRAINING----")
    # creates a linear regression model from sklearn.linear_model, model finds the best-fitting line to predict y from X, minimises the mean squared error b/w predicted& actual y.
    model=LinearRegression()
    # The model learns from the training data (X_train_scaled and y_train); calculates the optimal coefficients (weights) that minimize the prediction error.
    model.fit(X_train_scaled, y_train)

    # prediction on a test set (applies the learned coefficients to the test features)
    y_pred=model.predict(X_test_scaled)

    # r^2 score (Coefficient of Determination) measures how well the model explains the variance in the target variable.
    # r^2 score: closer to 1 (better model), = 1 (perfect prediction), = 0 (performs no better than predicting the mean) & < 0 (worse)
    r2 = r2_score(y_test, y_pred)
    print("r2 score is %0.2f" % r2)
    print("Successfully done!")

if __name__ == "__main__":
    main()
