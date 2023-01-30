import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class michaelisMenten:
    '''
    Convenience function thats takes in
    velocities and substrate concentrations
    and returns Km and Vmax.
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

        #Predict likely initial parameters
        self.Km_guess = np.median(self.x)
        self.Vm_guess = np.max(self.y)

        #Using the scipy minimize function to optimize from initial predicted parameters.
        initial_predictions = np.array([self.Vm_guess, self.Km_guess])
        result = minimize(self.RSS, initial_predictions, method='Powell')
        self.fitted_params = result.x

    #define Michaelis-Menten equation
    def formula(self, x, Vmax, Km):
        return (Vmax * x)/(Km + x)

    #Returns the residual sum of squares between predicted and expected values.
    def RSS(self, init_params):
        predictions = self.formula(self.x, *init_params)
        return sum((self.y - predictions) ** 2)

    #Returns R-squared value of the determined regression.
    def accuracy_metrics(self):
        model_predictions = self.formula(self.x, *self.fitted_params)
        abs_error = self.y - model_predictions
        R2 = 1.0 - (np.var(abs_error) / np.var(self.y))
        return str(np.round(R2, 5))

    #Return Vmax and Km values
    def parameters(self):
        return self.fitted_params[0], self.fitted_params[1]

    def chart(self, x_labels=None, y_labels=None, plot_title=None):
    #plot raw data
        plt.scatter(self.x, self.y)

        #plot regression
        x_values = np.arange(np.min(self.x), np.max(self.x), 0.1)
        y_values = self.formula(x_values, *self.fitted_params)
        plt.plot(x_values, y_values, label=f'Km: {self.fitted_params[1]} \n Vmax: {self.fitted_params[0]}')
        if plot_title:
            plt.title(plot_title)
        if x_labels:
            plt.xlabel(x_labels)
        if y_labels:
            plt.ylabel(y_labels)
        plt.legend()
        plt.show()

#example
x_test = np.array([5, 10, 20, 40, 80])
y_test = np.array([9.6, 16.56, 22.646, 27.180, 30.178])

#create instance
model = michaelisMenten(x_test, y_test)
model.chart('[substrate]', 'Rate (mOD/min)')