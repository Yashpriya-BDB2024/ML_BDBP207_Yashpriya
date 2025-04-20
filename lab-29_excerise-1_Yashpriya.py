### Simulate a dataset of 1000 points from a Normal distribution with mu=10, sd=3.
### Write a log-likelihood function and optimize it to find the mu and sigma.

import numpy as np
import matplotlib.pyplot as plt

# Simulate data from N(10, 3) -
np.random.seed(42)
data = np.random.normal(10, 3, 1000)

def negative_log_likelihood(mu, sigma, data):
    # For normal distribution, negative log likelihood formula is: ((n/2)*log(2*pi)) + (n*log(sigma)) + (1/(2*(sigma^2)) * summation(xi-mu)^2)
    n = len(data)
    part1 = n * 0.5 * np.log(2 * np.pi)
    part2 = n * np.log(sigma)
    part3 = np.sum((data - mu)**2) / (2 * sigma**2)
    return part1 + part2 + part3

# Try different values of mu and sigma (brute force).
mu_values = np.linspace(5,15,100)
sigma_values = np.linspace(1,5,100)

# Initialization -
min_nll = float('inf')   # Stores the lowest negative log-likelihood seen so far.
best_mu = None  # Stores the mean that gave this minimum.
best_sigma = None  # Stores the standard deviation that gave this minimum.

# Loops via every combination of mu & sigma from the generated ranges (100 × 100 = 10,000 combinations).
for mu in mu_values:
    for sigma in sigma_values:
        nll = negative_log_likelihood(mu, sigma, data)
        if nll < min_nll:
            min_nll = nll
            best_mu = mu
            best_sigma = sigma
# Prints the estimated parameters that gave the maximum likelihood (i.e., minimum negative log-likelihood).
print(f"Estimated mu (MLE): {best_mu:.4f}")
print(f"Estimated sigma (MLE): {best_sigma:.4f}")

# Plots to visualize the optimization -
# Fix sigma and vary mu
fixed_sigma = 3
nll_mu = [negative_log_likelihood(mu, fixed_sigma, data) for mu in mu_values]
plt.figure(figsize=(8, 4))
plt.plot(mu_values, nll_mu, label=f'sigma = {fixed_sigma}')
plt.xlabel('mu (Mean)')
plt.ylabel('NLL (Negative Log-Likelihood)')
plt.title('NLL vs mu (for fixed sigma, i.e., 3)')
plt.legend()
plt.show()

# Fix mu and vary sigma
fixed_mu = 10
nll_sigma = [negative_log_likelihood(fixed_mu, sigma, data) for sigma in sigma_values]
plt.figure(figsize=(8, 4))
plt.plot(sigma_values, nll_sigma, label=f'mu = {fixed_mu}')
plt.xlabel('Sigma (Standard Deviation)')
plt.ylabel('NLL (Negative Log-Likelihood)')
plt.title('NLL vs Sigma (for fixed mu, i.e., 10)')
plt.legend()
plt.show()