'''
From: https://pythonprogramming.net/how-to-program-best-fit-line-slope-machine-learning-tutorial/?completed=/simple-linear-regression-machine-learning-tutorial/
'''

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from create_samples import create_dataset as cds

def best_fit_slope(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs)**2 - mean(xs* xs))
    b = mean(ys) - m * mean(xs)
    return m, b

def squared_error(ys_o, ys_l):
    return sum((ys_l - ys_o) * (ys_l - ys_o))

def coef_determination(ys_o, ys_l):
    y_mean_line = [mean(ys_o) for y in ys_o]
    squared_error_regr = squared_error(ys_o, ys_l)
    squared_error_mean = squared_error(ys_o, y_mean_line)
    return 1 - (squared_error_regr / squared_error_mean)

'''
Theres a obvious relationship between the size of the sample
and the variance when it comes to precision.
'''

xs, ys = cds(1000, 700, correlation='pos') 

m, b = best_fit_slope(xs, ys)

reg_line = [(m * x) + b for x in xs]

style.use('ggplot')

plt.scatter(xs, ys, color='#003F72')
plt.plot(xs, reg_line)
plt.show()

r_squared = coef_determination(ys, reg_line)
print(r_squared)

