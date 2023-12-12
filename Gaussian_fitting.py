import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, a, b, c):
    """Define the Gaussian function."""
    return a * np.exp(-((x - b)**2) / (2 * c**2))

def fit_gaussian(x_data, y_data, initial_guess=[1, 0, 1]):
    """Fit a Gaussian function to the given data."""
    popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
    return popt, pcov

