# Decision Tree Regression

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

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=4, random_state=0)
regressor.fit(X_train, y_train)
# print(regressor.score(X_test, y_test))

# Evaluation:
import calendar, time
today_epoche = calendar.timegm(time.gmtime())
today_features = [today_epoche]
X_now = np.array(today_features).reshape(-1, len(today_features))
print("Today's predicted price: {}".format(regressor.predict(X_now)))

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Date vs Price (Decision Tree Regression)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Date vs Price (Decision Tree Regression)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()