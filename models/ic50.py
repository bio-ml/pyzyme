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

        # Using the scipy minimize function to optimize from initial predicted parameters.
        initial_predictions = np.array([self.ip_min_y, self.ip_max_y, 1, self.ip_ic50])
        result = minimize(self.RSS, initial_predictions, method='Powell')
        self.fitted_params = result.x

    # Four-parameter function to calculate IC50 regression.
    def formula(self, X, min_y, max_y, hill_coeff, ic50):
        return min_y + ((max_y - min_y) / (1 + np.power((X / ic50), hill_coeff)))

    # Returns the residual sum of squares between predicted and expected values.
    def RSS(self, init_params):
        predictions = self.formula(self.x, *init_params)
        return sum((self.y - predictions) ** 2)

    # Returns R-squared value of the determined regression.
    def r_2(self, verbose=True):
        model_predictions = self.formula(self.x, *self.fitted_params)
        abs_error = self.y - model_predictions
        r2 = 1.0 - (np.var(abs_error) / np.var(self.y))
        if verbose:
            return 'The R-squared value of the regression is ' + str(np.round(r2, 5)) + '.'
        else:
            return str(np.round(r2, 5))

    # Returns the value for the calculated IC50.
    def ic50_val(self, verbose=True):
        ic50 = self.fitted_params[3]
        if verbose:
            return 'The IC50 value of the regression is ' + str(np.round(ic50, 5)) + '.'
        else:
            return np.round(ic50, 5)

    # Cleans up the parameters, labels them, and returns them.
    def parameters(self):
        minimum = self.fitted_params[0]
        maximum = self.fitted_params[1]
        hill_coeff = self.fitted_params[2]
        ic50 = self.fitted_params[3]
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
        x_vals = np.arange(min(self.x), max(self.x), 0.1)
        y_preds = self.formula(x_vals, *self.fitted_params)
        plt.plot(x_vals, y_preds, label=f'R2: {self.r_2(verbose=False)}\n IC50: {np.round(self.ic50_val(verbose=False), 3)}')
        plt.title('4 parameter logistic fit')
        plt.xlabel('x-units')
        plt.ylabel('y-units')
        plt.legend()
        return plt

    # Allows the user to enter a single value or a list to obtain predicted y value(s).
    def predict(self, user_input):
        prediction = self.formula(user_input, *self.fitted_params)
        return prediction

    # Allows the user to add values of their own to the plotted regression.
    def chart_predictions(self, user_input):
        predictions = self.predict(user_input)

        # Calls the plotted regression and adds the user input predictions.
        self.chart()
        plt.scatter(user_input, predictions, c='r', s=50)
        plt.title('4 parameter logistic fit')
        plt.xlabel('x-units')
        plt.ylabel('y-units')
        plt.show()
        return self


# Sample numbers, somewhat loggy
x = [10, 8, 9, 4, 6, 1, 3]
y = [0.9, 0.85, 0.88, 0.5, 0.7, .1, .2]

example = IC50(x, y)
print(example.ic50_val())
print(example.r_2())
print(example.parameters())
print(example.predict(np.array([1, 3, 4])))
example.chart_predictions(np.array([1, 3, 4]))
#example.chart().show()
