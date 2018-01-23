# Machine Learning with Stock data

# Add parent
import os, sys
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent dir
sys.path.append(APP_DIR)

from app import Stock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
ticker = 'BOX'
df = Stock.query.filter_by(ticker=ticker).first().price_history
df.index = df.index.view('int64') // pd.Timedelta(1, unit='s')  # ns --> s
df = df[['Adj Close', 'Volume']].reset_index()
features = ['index', 'Volume']
X = df.loc[:, features].values
y = df.loc[:, 'Adj Close'].values
# print(X.shape)
# print(y.shape)
# print(X[:3])
# print(y[:3])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling not required

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluation:
import calendar, time
today_epoche = calendar.timegm(time.gmtime())
today_vol = 2159213
l = [today_epoche, today_vol]
X_now = np.array(l).reshape(-1, len(l))
print(X_now.shape)
print(regressor.predict(X_now))

# Results: today's price = 22.68
# - Simple Linear Regression (Date, Adj Close): 17.09
# - Multiple Linear Regression (Date, Volume, Adj Close): 17.32
# - Polynomial Regression

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# # Visualising the Training set results
# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# # Visualising the Test set results
# plt.scatter(X_test, y_test, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
