# Multiple Linear Regression

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
df = df[['Adj Close', 'Volume']].reset_index().rename(columns={'index': 'Date'})
features = ['Date', 'Volume']
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
# print(regressor.score(X_test, y_test))  # 10%

# Evaluation:
import calendar, time
today_epoche = calendar.timegm(time.gmtime())
today_vol = 1091932
today_features = [today_epoche, today_vol]
X_now = np.array(today_features).reshape(-1, len(today_features))
print("Today's predicted price: {}".format(regressor.predict(X_now)))

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# # Building the optimal model using Backward Elimination
# import statsmodels.formula.api as sm
# X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)
# # X is an array of ones (constant) followed by the x values
# X_opt = X[:, [0, 1, 2]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
# # All p-values are below the "significant level" BUT but Adj R^2 (0.119) isn't close to 1 so bad model
# # The two variables which best determine Price are: Date + Volume