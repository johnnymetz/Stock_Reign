# Simple Linear Regression

# Add parent directory
import os, sys
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent dir
sys.path.append(APP_DIR)

# Importing the libraries
from app import Stock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Importing the dataset
ticker = 'BOX'
df = Stock.query.filter_by(ticker=ticker.upper()).first().price_history
df.index = df.index.view('int64') // pd.Timedelta(1, unit='s')  # ns --> s
df = df[['Adj Close']].reset_index().rename(columns={'index': 'Date'})
features = ['Date']
X = df.loc[:, features].values
y = df.loc[:, 'Adj Close'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling not required

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# print(regressor.score(X_test, y_test))  # 6%

# Evaluation:
import calendar, time
today_epoche = calendar.timegm(time.gmtime())
today_features = [today_epoche]
X_now = np.array(today_features).reshape(-1, len(today_features))
print("Today's predicted price: {}".format(regressor.predict(X_now)))

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Date vs Price (Training set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Date vs Price (Test set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()