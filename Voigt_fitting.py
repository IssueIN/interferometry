import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
import matplotlib.pyplot as plt
from scipy.special import wofz

# Define the Voigt function (normalized Voigt profile)
# def voigt(x, center, amplitude, sigma, gamma):
#     """
#     Voigt profile function.
    
#     Parameters:
#     - x: The independent variable where the Voigt profile is evaluated
#     - center: The center of the profile
#     - amplitude: The amplitude of the profile
#     - sigma: The standard deviation of the Gaussian distribution
#     - gamma: The half-width at half-maximum of the Lorentzian distribution
    
#     Returns:
#     - Voigt profile evaluated at x
#     """
#     # The Voigt profile is the convolution of a Lorentzian and a Gaussian profile
#     return amplitude * voigt_profile((x-center)/sigma, gamma/sigma)
def voigt(x, center, amplitude, sigma, gamma):
    """
    Voigt profile function defined using the Faddeeva function.
    """
    z = ((x-center) + 1j*gamma) / (sigma*np.sqrt(2))
    return amplitude * wofz(z).real / (sigma*np.sqrt(2*np.pi))


def fit_voigt(x_data, y_data, initial_guess):
    popt, pcov= curve_fit(voigt, x_data, y_data, p0=initial_guess)
    return popt, pcov
