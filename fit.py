import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
data = np.loadtxt('CS137_0_thickness.tsv', skiprows=24, usecols=(0, 1))
channels = data[:, 0]
counts = data[:, 1]

def model(x, N, mu, sigma, A, B):
    gaussian = (N / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    linear = A * x + B
    return gaussian + linear

# Initial guesses for the parameters
initial_guess = [1000, 650, 10, -0.1, 50]

# Perform the fit with y errors = sqrt(y + 1)
y_errors = np.sqrt(counts + 1)
popt, pcov = curve_fit(model, channels, counts, p0=initial_guess, sigma=y_errors, absolute_sigma=True)

# Extract the fitted parameters and their errors
N_fit, mu_fit, sigma_fit, A_fit, B_fit = popt
N_err, mu_err, sigma_err, A_err, B_err = np.sqrt(np.diag(pcov))

# Print the results
print("Fitted Parameters:")
print(f"N = {N_fit:.2f} ± {N_err:.2f}")
print(f"μ = {mu_fit:.2f} ± {mu_err:.2f}")
print(f"σ = {sigma_fit:.2f} ± {sigma_err:.2f}")
print(f"A = {A_fit:.2f} ± {A_err:.2f}")
print(f"B = {B_fit:.2f} ± {B_err:.2f}")

# Plot the data and the fit
plt.errorbar(channels, counts, yerr=y_errors, fmt='o', label='Data', markersize=3, capsize=2)
plt.plot(channels, model(channels, *popt), 'r-', label='Fit')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.legend()
plt.show()