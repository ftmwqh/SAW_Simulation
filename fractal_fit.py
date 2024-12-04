import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

#TODO: sandbox

# Load the data from the file
lnN = np.loadtxt("log_count.txt")
lnr = np.loadtxt("log_r.txt")

# Linear fit
slope, intercept, r_value, p_value, std_err = linregress(lnr, lnN)

# Print the results of the linear fit and gamma
print(f"Slope (exponent): {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")

# Plot the data points and the fit result
plt.figure(figsize=(8, 6))
plt.plot(lnr, lnN, 'o', label="Data")  # data points
plt.plot(lnr, slope * lnr + intercept, 'r-')  # Fit result
plt.xlabel("ln(N)")
plt.ylabel("ln(r)")
plt.title("Linear Fit of ln(N) ~ ln(r)")
plt.legend()
plt.grid()
plt.show()

#TODO: Radius of Gyration

# Load the data from the file
lnN = np.loadtxt("n_eff.txt")
lnR2 = np.loadtxt("Rg.txt")
lnRg = 0.5*(lnR2-lnN)

# Linear fit
slope, intercept, r_value, p_value, std_err = linregress(lnRg, lnN)

# Print the results of the linear fit and gamma
print(f"Slope (exponent): {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")

# Plot the data points and the fit result
plt.figure(figsize=(8, 6))
plt.plot(lnRg, lnN, 'o', label="Data")  # data points
plt.plot(lnRg, slope * lnRg + intercept, 'r-')  # Fit result
plt.xlabel("ln(lnN)")
plt.ylabel("ln(R^2)")
plt.title("Linear Fit of ln(lnN) ~ ln(lnRg)")
plt.legend()
plt.grid()
plt.show()

