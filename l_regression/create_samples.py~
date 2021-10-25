'''
This algorithm aims to create sample datasets.

Tutorial in:
https://pythonprogramming.net/sample-data-testing-machine-learning-tutorial/?completed=/how-to-program-r-squared-machine-learning-tutorial/
'''

import random
import numpy as np

def create_dataset(hm, variance, step=2, correlation=False):
    '''
    hm - The amount of data points to be generated
    variance - The variance in the dataset
    step - The default steping between points
    correlation - Indication if wether one want some correlation
    '''
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = list(range(hm))

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40, 40, 2, correlation='pos')

