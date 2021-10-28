'''
The goal of this script is measure the efficiency of
different regression and approximation methods.
'''
import click
import numpy as np
from yaml import load, safe_load
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
                states = [v for v in self.conf[k].values()]
                scenarios = list(product(*states))
                for scene in scenarios:
                    clf = v(scene)
                    clf.fit(self.X_train, self.y_train)   
                    score = clf.score(self.X_test, self.y_test)
                    mae = mean_absolute_error(self.y_test, clf.predict(self.X_test))
                    # The join(map(***)) is due to the fact that join
                    # method only works in strings
                    self.add_result([k, '-'.join(map(str, scene)), score, mae])

        # Print the best results
        print('The best 5 results were:')
        self.results.sort_values(by=['Score'], ascending=False, inplace=True)
        print(self.results[:5])
        
    def add_result(self, res):
        self.results.loc[len(self.results.index)] = res


    # DecisionTree Regression
    def DTR(self, scene):
        from sklearn import tree
        return tree.DecisionTreeRegressor(
                    criterion=scene[0],
                    splitter=scene[1],
                    max_depth=scene[2])

    # LinearRegression
    def LR(self, scene):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1, copy_X=True)

    # Ridge
    def RD(self, scene):
        from sklearn.linear_model import Ridge
        return Ridge(alpha=scene[0], copy_X=True)

    # SGDRegressor
    def SGDR(self, scene):
        from sklearn.linear_model import SGDRegressor
        return SGDRegressor(loss=scene[0], penalty=scene[1],
                               alpha=scene[2], max_iter=scene[3],
                               learning_rate=scene[4], power_t=scene[5])

    # ElasticNet
    def ENET(self, scene):
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=scene[0], l1_ratio=scene[1],
                             max_iter=scene[2], copy_X=True)
     
# RandomForest
# SVR
# TheilSenRegressor
# RANSACRegressor
# HuberRegressor
# GaussianNB

@click.command()
@click.option('-cf', '--config_file',
              default='',
              help='The yaml file with the configurations for the run.')
@click.option('')
@click.option('-of', '--output_file',
              default='',
              help='The file to save the results and parameters of the run.')
@click.option('-br', '--best_results',
              default=1,
              help='1(True) or 0(False) to print the 5 best results.')
def test(config_file, output_file, best_results):
    '''
    This script simulate various regressor algorithms based on the parameters described in the yaml file.
    The results could be outputed to a csv file or/and printed to the screen.
    '''
    #TODO alther this part to 
    with open('reg_est.yaml', 'r') as f:
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
