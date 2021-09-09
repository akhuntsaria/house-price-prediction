import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys

def get_linear_regression_and_test_data():
    df = pd.read_csv("data.csv")

    X = df[['sqm']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    return reg, X_test, y_test

def run_automatic_test(reg, X_test, y_test):
    print("------- Automatic test -------")
    print("Test values:\n", X_test.values)
    print("Prediction:", reg.predict(X_test))
    print("Actual:", y_test.values)
    print("Score", reg.score(X_test, y_test), "\n")

def run_manual_test(reg):
    print("------- Manual test -------")
    sqmInput = input('Enter sqm (or q!): ')
    if sqmInput == 'q!':
        sys.exit()

    sqmInput = float(sqmInput)

    # Convert input to 2D array and get prediction
    print("Prediction:", reg.predict([[sqmInput]])[0])

reg, X_test, y_test = get_linear_regression_and_test_data()

run_automatic_test(reg, X_test, y_test)

while True:
    run_manual_test(reg)
