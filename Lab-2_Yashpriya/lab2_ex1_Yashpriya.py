from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def load_data():
    [X, y] = fetch_california_housing(return_X_y=True)
    return (X, y)

def main():
    # load california housing data
    [X, y] = load_data()

    # split data - train = 70%, test = 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    # scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train a model
    print("-----TRAINING----")
    # training a linear regression
    model=LinearRegression()
    # train the model
    model.fit(X_train, y_train)

    # prediction on a test set
    y_pred=model.predict(X_test)

    # compute the r2 score (r2 score closer to 1 is considered to be good)
    r2 = r2_score(y_test, y_pred)
    print("r2 score is %0.2f" % r2)
    print("Successfully done!")

if __name__ == "__main__":
    main()