import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class IC50:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Predicts likely initial parameters to optimize from.
        self.ip_min_y = min(y)
        self.ip_max_y = max(y)
        self.ip_ic50 = np.median(x)
        self.initial_predictions = np.array([self.ip_min_y, self.ip_max_y, 1, self.ip_ic50])

    # Four-parameter function to calculate IC50 regression.
    def formula(self, X, min_y, max_y, hill_coeff, ic50):
        return min_y + ((max_y - min_y) / (1 + np.power((X / ic50), hill_coeff)))

    # Returns the residual sum of squares between predicted and expected values.
    def RSS(self, init_params):
        predictions = self.formula(self.x, *init_params)
        return sum((self.y - predictions) ** 2)

    # Using the scipy minimize function to optimize from initial predicted parameters.
    def fit(self):
        result = minimize(self.RSS, self.initial_predictions, method='Powell')
        fitted_params = result.x
        return fitted_params

    # Currently, returns R-squared value of the determined regression.
    def accuracy_metrics(self):
        fitted_params = self.fit()
        model_predictions = self.formula(self.x, *fitted_params)
        abs_error = self.y - model_predictions
        std_err = np.square(abs_error)
        MSE = np.mean(std_err)
        R2 = 1.0 - (np.var(abs_error) / np.var(y))
        return 'The R-squared value of the regression is ' + str(np.round(R2, 2)) + '.'

    # Returns the value for the calculated IC50.
    def ic50_val(self):
        fitted_params = self.fit()
        return fitted_params[3]

    # Cleans up the parameters, labels them, and returns them.
    def parameters(self):
        fitted_params = self.fit()
        minimum = fitted_params[0]
        maximum = fitted_params[1]
        hill_coeff = fitted_params[2]
        ic50 = fitted_params[3]
        params = [minimum, maximum, hill_coeff, ic50]
        params = [str(np.round(x, 2)) for x in params]
        labels = ['min: ', 'max: ', 'Hill coeff: ', 'IC50: ']
        params = [labels[i] + params[i] for i in range(len(params))]
        return params

    # Plots a comparison of the raw data with the calculated regression.
    def chart(self):
        # Plotting the raw data.
        plt.scatter(self.x, self.y)

        # Plotting the regression.
        fitted_params = self.fit()
        x_vals = np.arange(0.1, max(self.x)+1, 0.1)
        y_preds = self.formula(x_vals, *fitted_params)
        plt.plot(x_vals, y_preds)
        plt.show()

    # Allows the user to enter a single value or a list to obtain predicted y value(s).
    def predict(self, user_input):
        fitted_params = self.fit()
        prediction = self.formula(user_input, *fitted_params)
        return prediction


# Sample numbers, somewhat loggy
x = [10, 8, 9, 4, 6, 1, 3]
y = [0.9, 0.85, 0.88, 0.5, 0.7, .1, .2]

example = IC50(x, y)
print(example.ic50_val())
print(example.accuracy_metrics())
print(example.parameters())
print(example.predict(np.array([2, 3, 4])))
example.chart()
