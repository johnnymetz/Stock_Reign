# Naive Bayes

# Add parent directory
import os, sys
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent dir
sys.path.append(APP_DIR)

# Importing the libraries
from app import Stock
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import sys

# Importing the dataset
ticker = 'BOX'
df = Stock.query.filter_by(ticker=ticker.upper()).first().price_history
df['pct_change_daily'] = df['Adj Close'].pct_change(periods=1)  # pct change for 1 day
df['Up or Down'] = (df['pct_change_daily'] > 0).astype(int)
df['Days since Epoch'] = df.index.map(dates.date2num)
df = df.iloc[1:]  # remove first day (IPO)
features = ['Days since Epoch', 'Volume']
X = df.loc[:, features].values
y = df.loc[:, 'Up or Down'].values

# # Initial Visualization (There's no correlation unfortunately)
# df.plot(kind='box', subplots=True)  # Box and Whisker
# df.hist()
# scatter_matrix(df)
# plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))  # 59%

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Test today:
from datetime import datetime
jan22 = (datetime(2018, 1, 22) - datetime(1970, 1, 1)).days
jan22_vol = 1091932
jan22_features = [jan22, jan22_vol]
X_now = np.array(jan22_features).reshape(-1, len(jan22_features))
print("Jan 22 up or down (up, 1): {}".format(classifier.predict(X_now)))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.05),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.05))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Days since Epoch')
plt.ylabel('Volume')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Days since Epoch')
plt.ylabel('Volume')
plt.legend()
plt.show()