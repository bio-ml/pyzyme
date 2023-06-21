import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class BasePlot:
    '''
    Base class to define plotting for all functions. 
    x: x values from parent class
    y: y values from parent class
    formula: formula from parent class
    fitted_params: fitted_params from parent class
    '''
    def __init__(self, x, y, formula, fitted_params): 
        self.x = x
        self.y = y
        self.formula = formula
        self.fitted_params = fitted_params
        
        
    def chart(self):
        plt.scatter(self.x, self.y, c='k')
        
        #plot regression
        x_values = np.arange(np.min(self.x), np.max(self.x), 0.1)
        y_values = self.formula(x_values, *self.fitted_params)
        
        plt.plot(x_values, y_values)
        plt.show()


class FourParameterLogistic:
    '''
    Current problems: if a new y value is greater than 
    y_predicted, an error will be thrown. Numpy currently
    doesn't allow powers of negative numbers, even if it
    doesn't result in a complete number. We need to create 
    exception handling to catch this in the future. 
    '''
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
        return max_y + ((min_y - max_y) / (1 + np.power((X / ic50), hill_coeff)))

    #Define prediction formula
    def predict_formula(self, Y, min_y, max_y, hill_coeff, ic50):
        '''
        Solve formula for X to allow unknown signal values (y)
        to return x values based on standard curve
        '''
        return ic50 * np.power((((min_y - max_y)/(Y - max_y)) - 1), (1/hill_coeff))

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
    def chart(self, x_labels=None, y_labels=None, plot_title=None):
        # Plotting the raw data.
        plt.scatter(self.x, self.y, c='b', s=30)

        # Plotting the regression.
        x_vals = np.arange(min(self.x), max(self.x), 0.1)
        y_preds = self.formula(x_vals, *self.fitted_params)
        plt.plot(x_vals, y_preds, c='b', label=f'R2: {self.r_2(verbose=False)}')
        if plot_title:
            plt.title(plot_title)
        if x_labels:
            plt.xlabel(x_labels)
        if y_labels:
            plt.ylabel(y_labels)
        plt.legend()
        return plt

    # Allows the user to enter a single value or a list to obtain predicted y value(s).
    def predict(self, user_input):
        prediction = self.predict_formula(user_input, *self.fitted_params)
        return prediction

    # Allows the user to add values of their own to the plotted regression.
    def chart_predictions(self, user_input, x_labels=None, y_labels=None, plot_title=None):
        predictions = self.predict(user_input)

        # Calls the plotted regression and adds the user input predictions.
        self.chart(x_labels, y_labels, plot_title)
        plt.scatter(predictions, user_input, c='r', s=30)
        plt.show()
        return self

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

class michaelisMenten:
    '''
    Convenience function thats takes in
    velocities and substrate concentrations
    and returns Km and Vmax.
    '''
    def __init__(self, x, y):
        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                self.x = x.to_numpy()
                self.y = y.mean(axis=1).to_numpy()
                self.stand_dev = y.std(axis=1).to_numpy()
            
            else:
                self.x = x.to_numpy()
                self.y = y.to_numpy()
                self.stand_dev = np.array(False)
                
        elif isinstance(x, np.ndarray):
            self.x = x
            self.y = y
            self.stand_dev = np.array(False)

        else:
            self.x = np.array(x)
            self.y = np.array(y)
            self.stand_dev = np.array(False)

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
        plt.scatter(self.x, self.y, c='k')

        #plot regression
        x_values = np.arange(np.min(self.x), np.max(self.x), 0.1)
        y_values = self.formula(x_values, *self.fitted_params)
        if self.stand_dev.any():
            plt.plot(x_values, y_values, c='k',
                      label=f'Km: {self.fitted_params[1]} \n Vmax: {self.fitted_params[0]}')
            plt.errorbar(self.x, self.y, yerr=self.stand_dev, ecolor='k', ls='none')
        else:
            plt.plot(x_values, y_values, c='k',
                     label=f'Km: {self.fitted_params[1]} \n Vmax: {self.fitted_params[0]}')
        if plot_title:
            plt.title(plot_title)
        if x_labels:
            plt.xlabel(x_labels)
        if y_labels:
            plt.ylabel(y_labels)
        plt.legend()
        plt.show()

class SubstrateInhibition:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Predicts likely initial parameters to optimize from.
        Vm_guess = np.max(self.y)
        Km_guess = np.median(self.x)
        Ki_guess = np.max(self.x)

        # Using the scipy minimize function to optimize from initial predicted parameters.
        initial_predictions = np.array([Vm_guess, Km_guess, Ki_guess])
        result = minimize(self.RSS, initial_predictions, method='Powell')
        self.fitted_params = result.x

        #initialize plotting
        BasePlot(self.x, self.y, self.formula, self.fitted_params )

    # Function to calculate substrate inhibition.
    def formula(self, X, Vmax, Km, Ki):
        return (Vmax * X) / (Km + X * (1 + X / Ki))

    # Returns the residual sum of squares between predicted and expected values.
    def RSS(self, init_params):
        predictions = self.formula(self.x, *init_params)
        return sum((self.y - predictions) ** 2)

    # Return Vmax, Km, and Ki
    def parameters(self):
        Vmax = np.round(self.fitted_params[0], 2)
        Km = np.round(self.fitted_params[1], 2)
        Ki = np.round(self.fitted_params[2], 2)

        params = [Vmax, Km, Ki]
        labels = ['Vmax ', 'Km ', 'Ki ']

        params = [l + str(p) for l, p in zip(labels, params)]
        return params