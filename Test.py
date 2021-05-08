import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("../Salary_data.csv")
print("The Dataset")
print("-----------")
print(dataset)
# Take all the rows, and all of the columns but the last one
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_prediction = regressor.predict(X_test)
print("y_prediction")
print("----------")
print(y_prediction)

print("actual")
print("------")
print(Y_test)

print("percentages")
print("-----------")
percentages = []
for index in range(len(y_prediction)):
  percentages.append(abs(y_prediction[index] - Y_test[index])/(y_prediction[index] + Y_test[index])*100)
print(percentages)

# Visualizing the training set and model
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train))
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualizing the test set and model
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_test, y_prediction)
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Single prediction example, an employee with 12 years experience
twelveYearPrediction = regressor.predict([[12]])
print("Twelve Years Experience, Single Prediction")
print("------------------------------------------")
print(twelveYearPrediction)

# Retriving our predicted formulas coefficients
slope = regressor.coef_[0]
print("slope:", slope)
yIntercept = regressor.intercept_
print("Y-Intercept:", yIntercept)