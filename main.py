import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

X = df[['sqm']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, y_train)

print("------- Automatic test -------")
print("Prediction:", reg.predict(X_test))
print("Actual:", y_test.values)
print("Score", reg.score(X_test, y_test))
print("------------------------------\n")
