'''
Example here
https://pythonprogramming.net/features-labels-machine-learning-tutorial/?completed=/regression-introduction-machine-learning-tutorial/
'''
import math
import pandas as pd
import numpy as np
import quandl
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = quandl.get('WIKI/GOOGL')

# Select a few columns
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])
# This function scales to -1 1
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Support Vector Regression
clf = svm.SVR()

clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)

print('Using the SVR the score was {0:.1f} and the MAE was {1:.1f}'.format(confidence, mean_absolute_error(y_test, clf.predict(X_test))))

# Linear Regression
# By specificing -1 the function will use all the threads avaible
lr = LinearRegression(n_jobs = -1)

lr.fit(X_train, y_train)

print('Using the Linear Regression the score was {0:.1f} and the MAE was {1:.1f}'.format(lr.score(X_test, y_test), mean_absolute_error(y_test, lr.predict(X_test))))

'''
To save the algorithm do:
import pickle

with open('lr_algo.pickle', 'wb') as f:
    pickle.dump(lr, f)

And to bring it back up just:

pickle_in = open('lr_algo.pickle', 'rb')
lr = pickle.load(pickle_in)
'''

