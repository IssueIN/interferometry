import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps

soundfreq = 20
filename = f'2_{soundfreq}Hz_motor30000'
results = rd.read_data3(f'data\\{filename}.txt')
y = np.array(results[1])
increment = 1/200
x = np.array([i * increment for i in range(len(y))])
upperbond = soundfreq * 4
lowerbond = soundfreq * 0.5



# conversion = 2 * 1.89355e-11 #0.039771e-9 #1.36191e-11 #0.034191e-9 #0.039771
mask = (x>=-10e20) #(x >= 7.8e6) & (x <= 8.4e6)#(-5.6e6 <= x) & (1e6 >= x)
#(x >= 1.42e7) & (x <= 1.47e7) 

x_masked = x[mask]
y_masked = y[mask]

# Perform FFT on masked data
if len(x_masked) > 1:
    Δx = (x_masked[-1] - x_masked[0]) / (len(x_masked) - 1)
else:
    Δx = 1 

yf1 = spf.fft(y_masked)
xf1 = spf.fftfreq(len(y_masked), Δx)
xf1 = spf.fftshift(xf1)
yf1 = spf.fftshift(yf1)

mask = (xf1 > lowerbond) & (xf1 <  upperbond)
yf1=yf1[mask]
xf1=xf1[mask]
yf1=np.abs(yf1)/np.abs(yf1).max()

# normalization = 1.6e8#np.abs(yf1).max()
# Plot the FFT result
plt.plot(xf1, yf1, label='FFT of Signal')
plt.xlabel('frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT Result')
plt.grid()
plt.legend()

plt.show()


from Gaussian_fitting import gaussian, fit_gaussian
from Voigt_fitting import voigt, fit_voigt

# Your dataset
x_data =  xf1
y_data =  np.abs(yf1)

# Fit the Gaussian function to your data
popt_g, pcov_g = fit_gaussian(x_data, y_data, [1, soundfreq, 1])
popt_v, pcov_v = fit_voigt(x_data, y_data, [soundfreq, 1, 5e-5, 1e-2] )
x = np.linspace(lowerbond, upperbond, 1000)

plt.figure(figsize=(8, 4))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x, gaussian(x, *popt_g), color='blue', label='Gaussian Fit')
plt.plot(x, voigt(x, *popt_v), color='red', label='Voigt Fit')
plt.title('Gaussian Fit to Dataset')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(popt_g)
print(pcov_g)
print(popt_v)

from scipy.stats import chisquare

# Calculate chi-square for Gaussian fit
# chi_sq_gauss, p_value_g = chisquare(f_obs=y_data, f_exp=gaussian(x_data, *popt_g))

# Calculate chi-square for Voigt fit
# chi_sq_voigt, p_value_v = chisquare(f_obs=y_data, f_exp=voigt(x_data, *popt_v))

# # print('chisq_g:', chi_sq_gauss)
# print('chisq_v:', chi_sq_voigt)
# Calculate the residuals
residuals_g = y_data - gaussian(x_data, *popt_g)
std_residuals_g = np.std(residuals_g)
errors_g = np.full_like(y_data, std_residuals_g)

residuals_v = y_data - voigt(x_data, *popt_v)
std_residuals_v = np.std(residuals_v)
errors_v = np.full_like(y_data, std_residuals_v)



chi_sq_g = np.sum(((y_data - gaussian(x_data, *popt_g)) ** 2) / errors_g ** 2)
chi_sq_v = np.sum(((y_data - voigt(x_data, *popt_v)) ** 2) / errors_v ** 2)

# The number of degrees of freedom is the number of observed data points minus the number of fitted parameters
num_parameters_g = 3  # For example, if you fit with 3 parameters
num_parameters_v = 4
dof_g = len(y_data) - num_parameters_g
dof_v = len(y_data) - num_parameters_v
print(dof_g)
# The reduced chi-square value is the chi-square value divided by the degrees of freedom
reduced_chi_sq_g = chi_sq_g / dof_g
reduced_chi_sq_v = chi_sq_v / dof_v

print(f"Chi-squared (manual calculation) for guassion fit: {chi_sq_g}")
print(f"Reduced Chi-squared (manual calculation) for guassion fit: {reduced_chi_sq_g}")
print(f"Chi-squared (manual calculation) for voigt fit: {chi_sq_v}")
print(f"Reduced Chi-squared (manual calculation) for voigt fit: {reduced_chi_sq_v}")

"""
f_cutoff = 0.00005  # Set your cutoff frequency
cutoff_index = np.where(np.abs(xf1) < f_cutoff)  # Find indices where frequency is greater than the cutoff

filtered_yf1 = np.copy(yf1)
filtered_yf1[cutoff_index] = 0  # Zero out the frequencies above the cutoff

# Optionally, if you need the filtered signal back in time domain
filtered_signal = spf.ifft(spf.ifftshift(filtered_yf1))

# Plot the filtered FFT result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(conversion/xf1, np.abs(filtered_yf1)/4.8e8, label='Filtered FFT')
plt.xlabel('Frequency (units)')
plt.ylabel('Amplitude')
plt.title('Filtered FFT Result')
plt.legend()

# Optionally, plot the filtered signal
plt.subplot(1, 2, 2)
plt.plot(x_masked, np.abs(filtered_signal), label='Filtered Signal')  # Assuming x_masked is your time domain axis
plt.xlabel('Time (units)')
plt.ylabel('Amplitude')
plt.title('Filtered Signal in Time Domain')
plt.legend()

plt.tight_layout()
plt.show()
"""