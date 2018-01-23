# Add parent
import os, sys
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent dir
sys.path.append(APP_DIR)

from matplotlib import style, dates, pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
import datetime
import sys

style.use('ggplot')
pd.options.mode.chained_assignment = None

# GOAL
# Prediction task is to generate a predictive regression model.
# Create a regression model which can predict a stock's Adj Close price given the Date, Volume, etc.

# Seven Steps:
# 1. Gather Data
# 2. Prepare Data
# 3. Choose a Model
# 4. Train Model (using train set)
# 5. Evaluate Model (using test set)
# 6. Tune Parameters (e.g. learning rate (how much we adjust model during each training step))
# 7. Create Final Model (using all data)
# 8. Predict

# DATA
df = pd.read_pickle('data/pickles/fb.pickle').reset_index()
if df.isnull().values.any():
    print('DataFrame contains NaN values. Please remove.')
    sys.exit(1)


def get_spread(row, col1, col2):
    return row[col1] - row[col2]

# df['HL Spread'] = df.apply(func=get_spread, axis=1, args=('High', 'Low',))
# df['OC Spread'] = df.apply(func=get_spread, axis=1, args=('Open', 'Adj Close',))
# df['Pct Change back'] = df['Adj Close'].pct_change(periods=1)  # pct change since close 1 row back
# df['Pct Change forw'] = df['Adj Close'].pct_change(periods=-3)  # pct change since close 1 row back
# ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Pct Change back', 'Pct Change forw']
df = df[['Date', 'Volume', 'Adj Close']]
# print(df[['Date', 'Adj Close', 'Pct Change 20', 'Pct Change -20']])
# If the stock price increases 1.5% over 1 trading day, what's the average pct change over the next 3 trading days?
# applicable_change = df[df['Pct Change back'] > .015]
# print(applicable_change.shape[0]/df.shape[0])
# print(applicable_change['Pct Change forw'].mean() * -1)
# Separate independent from dependent values
# X = df.index.values.reshape(-1, 1)
X = df.reset_index()[['index']].values
# Adding volume causes the slope to steepen a bit
# X3 = df.reset_index()[['index', 'Volume', 'HL Spread']].values
Y = df[['Adj Close']].values
# print(X.shape, Y.shape)
# X_possibilities = [X1, X2, X3]

# DATA VISUALIZATION
# df.plot(kind='box', subplots=True)  # Box and Whisker
# df.hist()
# scatter_matrix(df)
# df.plot.line(x='Date', y='Adj Close', label='Price')
# df.plot.scatter(x='OC Spread', y='Volume', label='Data')
# plt.show()


# CHOOSE MODEL
# Split Data into Train and Test sets
# for X in X_possibilities:
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)

# # TESTING LINEAR REGRESSION
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
m = model.coef_[0][0]
b = model.intercept_[0]
# print(m, b)
# print('Accuracy:', model.score(X_test, Y_test))
# # Time vs. Close Plot (Method 1: pure matplotlib)
# plt.scatter(x=df.Date.tolist(), y=df['Adj Close'].tolist(), s=2)
# plt.plot(df.Date.tolist(), [x * m + b for x in range(len(X))], 'b-')
# plt.title('y = {0}x + {1}'.format(round(m, 2), round(b, 2)))
# plt.xlabel('Date')
# plt.ylabel('Price')
# # Time vs. Close Plot (Method 2: pandas)
df['Days Since Epoch'] = df.Date.map(dates.date2num)
df['Predicted Price'] = [x * m + b for x in range(len(X))]
# pandas_plot = df.plot.scatter(x='Days Since Epoch', y='Adj Close', s=2, title='y = {0}x + {1}'.format(round(m, 2), round(b, 2)))
# df.plot.line(x='Days Since Epoch', y='Predicted Price', ax=pandas_plot)
# pandas_plot.xaxis_date()
# # Time vs. Close Plot (Method 3: only test set - only works if X contains index values)
plt.scatter(X_test[:, 0], Y_test, s=2)
plt.plot(sorted(X_train), sorted(model.predict(X_train)), 'b-')  # can put either X_train or X_test here
plt.title('y = {0}x + {1}'.format(round(m, 2), round(b, 2)))
plt.show()


# # Potential Models
# models = [
#     ('LR', LinearRegression())
# ]

# # Evaluate Models
# scoring = 'accuracy'
# results, names = [], []
# for name, model in models:
#
#     k = 6
#     # K-Fold: k train / test groups (k sets, each set is used as test set once)
#     # kfold = KFold(n_splits=k, random_state=seed)
#
#     # Stratified K-Fold: k train / test groups
#     skfold = StratifiedKFold(n_splits=k, random_state=seed)
#
#     # Leave-P-Out: leave p data points out as test; train with rest of data
#     # leave_one_out = LeaveOneOut()  # n groups
#
#     # Calculate accuracy scores for each test / train group
#     cv_results = cross_val_score(model, X_train, Y_train, cv=skfold, scoring=scoring)
#     print(len(cv_results))
#
#     # Report model error estimate as mean of accuracy scores
#     results.append(cv_results)
#     names.append(name)
#     print('{}: {:f} +/- {:f}'.format(name, cv_results.mean(), cv_results.std()))

# Chosen Model:






