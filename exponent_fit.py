import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the data from the file
data = np.loadtxt("SAW_3D.txt", skiprows=1)  # Skip header row
walk_length = data[:, 0]
B = data[:, 1]
R_square = data[:, 2]

#TODO: calculate gamma

# Pre-Process with log
ln_walk_length = np.log(walk_length)
ln_B = np.log(B)

# Linear fit
slope, intercept, r_value, p_value, std_err = linregress(ln_walk_length, ln_B)

# Print the results of the linear fit and gamma
print(f"Slope (exponent): {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f"gamma: {1-slope}")

# Plot the data points and the fit result
plt.figure(figsize=(8, 6))
plt.plot(ln_walk_length, ln_B, 'o', label="Data")  # data points
plt.plot(ln_walk_length, slope * ln_walk_length + intercept, 'r-')  # Fit result
plt.xlabel("ln(N)")
plt.ylabel("ln(BN)")
plt.title("Linear Fit of ln(BN) ~ ln(N)")
plt.legend()
plt.grid()
plt.show()

#TODO: calculate nu

# Pre-Process with log
ln_R_square = np.log(R_square)

# Linear fit
slope, intercept, r_value, p_value, std_err = linregress(ln_walk_length, ln_R_square)

# Print the results of the linear fit and gamma
print(f"Slope (exponent): {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f"nu: {slope/2}")

# Plot the data points and the fit result
plt.figure(figsize=(8, 6))
plt.plot(ln_walk_length, ln_R_square, 'o', label="Data")  # data points
plt.plot(ln_walk_length, slope * ln_R_square + intercept, 'r-')  # Fit result
plt.xlabel("ln(N)")
plt.ylabel("ln(R^2)")
plt.title("Linear Fit of ln(R^2) ~ ln(N)")
plt.legend()
plt.grid()
# plt.show()
