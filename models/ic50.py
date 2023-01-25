import numpy as np
from scipy.optimize import minimize

# Sample numbers somewhat loggy
x = np.array([10, 8, 9, 4, 6, 1, 3])
y = np.array([0.9, 0.85, 0.88, 0.5, 0.7, .1, .2])


def formula(X, A, B, C, D):
    return D + (A - D) / (1 + (X / C)**B)


def formula_inverted(Y_mid, A, B, C, D):
    return ((A - D) / (Y_mid - D) - 1)**(1/B) * C


def RSS(init_params):
    predictions = formula(x, *init_params)
    return sum((y - predictions)**2)


#TODO: find a method for getting these numbers. 
initial_predictions = np.array([1, -1.5, 400, 0])

# Using the scipy minimize function to optimize from initial predicted parameters. 
result = minimize(RSS, initial_predictions, method='Powell')
fitted_params = result.x

model_predictions = formula(x, *fitted_params)

abs_error = y - model_predictions

# Uses a very large number to find roughly the upper bound for Y. Can probably be improved mathematically. 
y_mean = formula(1000000, *fitted_params) / 2

ic50 = formula_inverted(y_mean, *fitted_params)
