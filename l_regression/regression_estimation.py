'''
The goal of this script is measure the efficiency of
different regression and approximation methods.
'''
import numpy as np
from yaml import load
import pandas as pd
from itertools import product
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class regression_test:
    def __init__(self, conf, X_train, X_test, y_train, y_test):
        # Set the random_state
        rng = np.random.RandomState(0)
        self.conf = conf
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = pd.DataFrame(columns=['Regressor', 'Param', 'Score', 'MAE'])

    def test(self):
        regressors = {
                'DecisionTreeRegressor': self.DTR,
                'LinearRegression': self.LR,
                'Ridge': self.RD,
                'SGDRegressor': self.SGDR,
                'ElasticNet': self.ENET
                }
        for k, v in regressors.items():
            if self.conf.get(k).get('Test'):
                self.conf[k].pop('Test')
                v()

        # Print the best results
        print('The best 5 results were:')
        self.results.sort_values(by=['Score'], ascending=False, inplace=True)
        print(self.results[:5])
        
    def add_result(self, res):
        self.results.loc[len(self.results.index)] = res


    # DecisionTree Regression
    def DTR(self):
        states = [v for v in self.conf['DecisionTreeRegressor'].values()]
        scenarios = list(product(*states))
        from sklearn import tree
        for scene in scenarios:
            clf = tree.DecisionTreeRegressor(
                    criterion=scene[0],
                    splitter=scene[1],
                    max_depth=scene[2])
            clf = clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)
            mae = mean_absolute_error(self.y_test, clf.predict(self.X_test))
            self.add_result(['DecisionTreeRegressor', 
                             '{}-{}-{}'.format(scene[0], scene[1], scene[2]), 
                             score, 
                             mae])

    # LinearRegression
    def LR(self):
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression(n_jobs=-1, copy_X=True)
        clf.fit(self.X_train, self.y_train)
        score = clf.score(self.X_test, self.y_test)
        mae = mean_absolute_error(self.y_test, clf.predict(self.X_test))
        self.add_result(['LinearRegresion', '', score, mae])

    # Ridge
    def RD(self):
        states = [v for v in self.conf['Ridge'].values()]
        scenarios = list(product(*states))
        from sklearn.linear_model import Ridge
        for scene in scenarios:
            clf = Ridge(alpha=scene[0], copy_X=True)
            clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)
            mae = mean_absolute_error(self.y_test, clf.predict(self.X_test))
            self.add_result(['Ridge', '{}'.format(scene[0]), 
                             score, mae])
            

    # SGDRegressor
    def SGDR(self):
        states = [v for v in self.conf['SGDRegressor'].values()]
        scenarios = list(product(*states))
        from sklearn.linear_model import SGDRegressor
        for scene in scenarios:
            clf = SGDRegressor(loss=scene[0], penalty=scene[1],
                               alpha=scene[2], max_iter=scene[3],
                               learning_rate=scene[4], power_t=scene[5])
            clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)
            mae = mean_absolute_error(self.y_test, clf.predict(self.X_test))
            self.add_result(['SGDRegressor', '{}-{}-{}-{}-{}'.format(
                            scene[0], scene[1], scene[2], scene[3]
                            , scene[4], scene[5]),
                            score, mae])

    # ElasticNet
    def ENET(self):
        states = [v for v in self.conf['ElasticNet'].values()]
        scenarios = list(product(*states))
        from sklearn.linear_model import ElasticNet
        for scene in scenarios:
            clf = ElasticNet(alpha=scene[0], l1_ratio=scene[1],
                             max_iter=scene[2])
            clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)
            mae = mean_absolute_error(self.y_test, clf.predict(self.X_test))
            self.add_result(['ElasticNet', '{}-{}-{}'.format(
                            scene[0], scene[1], scene[2]),
                            score, mae])
# RandomForest
# SVR
# TheilSenRegressor
# RANSACRegressor
# HuberRegressor
# GaussianNB

def test():
    with open('reg_est.conf', 'r') as f:
        conf=load(f, Loader=Loader)
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    X, y=load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test=train_test_split(X, y)
    tester=regression_test(conf, X_train, X_test, y_train, y_test)
    tester.test()
    tester.results.to_csv('results.csv')

if __name__ == '__main__':
    test()
'''
There's also the possibilitie of using a preprocessing tool
to transform the features points, like:
    - PolynomialFeatures
    - SplineTransformer
    - Radial Basis Function
Implement the ensemble algorithms
'''
