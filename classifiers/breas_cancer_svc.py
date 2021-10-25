'''
The goal of this script is to use Support Vector Machine to 
classify data from a research about breast cancer

Tutorial in:
://pythonprogramming.net/k-nearest-neighbors-application-machine-learning-tutorial/?completed=/k-nearest-neighbors-intro-machine-learning-tutorial/

'''

import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/pedro/Documents/ML/classifiers/data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
# By droping this column we increase the precision of
# the prediction from 65% to 95%
df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svc = svm.SVC()

svc.fit(X_train, y_train)

acc = svc.score(X_test, y_test)
print(acc)

