# Polynomial Regression

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

# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)  # col0 = 1. col1 = x. col2 = x^2. col3 = x^3. col4 = x^4.
poly_reg.fit(X_poly, y_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)
# print(regressor.score(poly_reg.fit_transform(X_test), y_test))  # 80%

# Evaluation:
import calendar, time
today_epoche = calendar.timegm(time.gmtime())
today_features = [today_epoche]
X_now = np.array(today_features).reshape(-1, len(today_features))
print("Today's predicted price: {}".format(regressor.predict(poly_reg.fit_transform(X_now))))

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X, regressor.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Date vs Price (Training set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(sorted(X_train), regressor.predict(poly_reg.fit_transform(sorted(X_train))), color='blue')
plt.title('Date vs Price (Test set)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()