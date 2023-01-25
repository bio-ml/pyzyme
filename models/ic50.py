import numpy as np
from scipy.optimize import minimize

# Sample numbers, somewhat loggy
x = np.array([10, 8, 9, 4, 6, 1, 3])
y = np.array([0.9, 0.85, 0.88, 0.5, 0.7, .1, .2])


# Use of a three parameter model here sets the min at zero.
def formula(X, A, B, C):
    return A / (1 + (X / C)**B)


def formula_inverted(Y_mid, A, B, C):
    return (A / Y_mid - 1)**(1/B) * C


def RSS(init_params):
    predictions = formula(x, *init_params)
    return sum((y - predictions)**2)

#TODO: create a method for determining these based on the data. E.g. max value can be the maximum observed value from x.
initial_predictions = np.array([1, -1.5, 400])

# Using the scipy minimize function to optimize from initial predicted parameters. 
result = minimize(RSS, initial_predictions, method='Powell')
fitted_params = result.x

model_predictions = formula(x, *fitted_params)

# Calculation of accuracy-based metrics
abs_error = y - model_predictions
std_err = np.square(abs_error)
MSE = np.mean(std_err)
Rsquared = 1.0 - (np.var(abs_error) / np.var(y))

# Uses a very large number to find roughly the upper bound for Y. Can probably be improved mathematically. 
y_mean = formula(1000000, *fitted_params) / 2

ic50 = formula_inverted(y_mean, *fitted_params)

print(f'The IC50 occurs at {np.round(ic50, 2)} units of x. The R squared of the regression is {np.round(R2, 2)}.')
