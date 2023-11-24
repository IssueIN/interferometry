import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps

filename = 'ML_yellow_2'
results = rd.read_data3(f'data\\{filename}.txt')
y = np.array(results[1])
x = np.array(results[5])


conversion = 2 * 1.89355e-11 #0.039771e-9 #1.36191e-11 #0.034191e-9 #0.039771
mask = (x>=1e-12)#(x >= 7.8e6) & (x <= 8.4e6)#(-5.6e6 <= x) & (1e6 >= x)
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

# Plot the original signal
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) 
plt.plot(x_masked, y_masked, label='Original Signal')
plt.xlabel('x (usteps)')
plt.ylabel('y (intensity)')
plt.title('Original Signal')
plt.legend()

normalization = 1.4e4#np.abs(yf1).max()
wl1 = conversion/xf1
fft_mask = (-1e12 <= wl1)#(0.4e-6 <= wl1) & (wl1 <= 0.8e-6)
wl1_mask = wl1[fft_mask]
yf1_mask = yf1[fft_mask]
# Plot the FFT result
plt.subplot(1, 2, 2)  
plt.plot(wl1_mask, np.abs(yf1_mask)/normalization, label='FFT of Signal')
plt.xlabel('Wavelength (m)')
plt.ylabel('Amplitude')
plt.title('FFT Result')
plt.grid()
plt.legend()

plt.tight_layout()
plt.suptitle(filename)
plt.show()


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