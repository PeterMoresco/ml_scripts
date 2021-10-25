'''
The goal of this script is measure the efficiency of
different regression and approximation methods.
'''
from yaml import load
from itertools import product
from sklearn.metrics import r2_score
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class regression_test:
    def __init__(self, conf, X_train, X_test, y_train, y_test):
        self.conf = conf
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def test(self):
        if self.conf.get('DTR'):
            self.DTR()
        if self.conf.get('LR'):
            self.LR()

    # DecisionTree Regression
    def DTR(self):
        states = [v for v in self.conf['DTR'].values()]
        scenarios = list(product(*states))
        from sklearn import tree
        for scene in scenarios:
            clf = tree.DecisionTreeRegressor(
                    criterion=scene[0],
                    splitter=scene[1],
                    max_depth=scene[2],
                    random_state=scene[3])
            clf = clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)
            print('''Decision Tree Regressor with crit {}
                    splitter {} max depth {} rnd state {} 
                    got a score of {}'''.format(scene[0], scene[1], scene[2], scene[3], score))

    # LinearRegression
    def LR(self):
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression(n_jobs=-1)
        clf.fit(self.X_train, self.y_train)
        score = clf.score(self.X_test, self.y_test)
        print('The Linear Regression got a score of {}'.format(score))

# RandomForest
# SVR
# TheilSenRegressor
# RANSACRegressor
# HuberRegressor
# GaussianNB
# Ridge

def test():
    with open('reg_est.conf', 'r') as f:
        conf=load(f, Loader=Loader)
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    X, y=load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test=train_test_split(X, y)
    tester=regression_test(conf, X_train, X_test, y_train, y_test)
    tester.test()

if __name__ == '__main__':
    test()
'''
There's also the possibilitie of using a preprocessing tool
to transform the features points, like:
    - PolynomialFeatures
    - SplineTransformer
    - Radial Basis Function
'''
