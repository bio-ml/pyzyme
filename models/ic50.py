import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Sample numbers, somewhat loggy
x = np.array([10, 8, 9, 4, 6, 1, 3])
y = np.array([0.9, 0.85, 0.88, 0.5, 0.7, .1, .2])


# Use of a three parameter model here sets the min at zero.
def formula(X, max_y, hill_coeff, ic50):
    return max_y / (1 + (X / ic50)**(-hill_coeff))


# The formula that will be used to determine IC50 from the derived parameters. 
def formula_inverted(Y_mid, max_y, hill_coeff, ic50):
    return (max_y / Y_mid - 1)**(1/hill_coeff) * ic50


def RSS(init_params):
    predictions = formula(x, *init_params)
    return sum((y - predictions)**2)

# Predicts likely initial parameters to minimize from.
ip_max_y = max(y)
ip_ic50 = np.median(x)
initial_predictions = np.array([ip_max_y, -1.5, ip_ic50])

# Using the scipy minimize function to optimize from initial predicted parameters. 
result = minimize(RSS, initial_predictions, method='Powell')
fitted_params = result.x

model_predictions = formula(x, *fitted_params)

# Calculation of accuracy-based metrics
abs_error = y - model_predictions
std_err = np.square(abs_error)
MSE = np.mean(std_err)
R2 = 1.0 - (np.var(abs_error) / np.var(y))

# Uses a very large number to find roughly the upper bound for Y. Can probably be improved mathematically. 
y_mean = formula(1000000, *fitted_params) / 2
ic50_from_plot = formula_inverted(y_mean, *fitted_params)

ic50_calc = fitted_params[2]

#plotting
plt.scatter(x, y)
plt.plot(x, model_predictions)
plt.show() 

print(f'The IC50 occurs at {np.round(ic50_calc, 2)} units of x. The R squared of the regression is {np.round(R2, 2)}.')
return fitted_params

