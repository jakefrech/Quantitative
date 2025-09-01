# Black-Scholes Model: Determine valuation of stock option (Call, Put)
# Takes in 6 Variables
# Assumptions:
# 1. Lognormal Distributions, asset price cannot take negative value
# 2. No dividends
# 3. Expiration Date, option only exercised on its expiration or maturity date (Euro Options)
# 4. Frictionless Market, no transaction costs
# 5. Risk-Free Interest Rate, interest rate is constant, underlying asset is a risk-free one
# 6. Normal Distribution, stock returns are normally distributed. Constant volatility overtime
# 7. No arbitrage opportunities. Cannot make risk-less profit

import numpy as np
from scipy.stats import norm

# Class to price European Call and Put options using the Black-Scholes Formula
# supports non-dividend and dividend paying stocks
class BlackScholesModel:
    def __init__(self, s, k, t, r, sigma, delta=0):
        self.s = s  # spot price of the underlying asset
        self.k = k  # strike price of the option
        self.t = t  # time to expiration in years (e.g., 240/365)
        self.r = r  # interest rate (constant, risk-free)
        self.sigma = sigma  # annual volatility of the underlying (standard deviation)
        self.delta = delta  # dividend yield or cost of carry (default 0)

    # Method to calculate d1 and d2
    @staticmethod
    def calculate_d1_d2():
        # Delta related factor
        # Expected gain if exercised
        d1 = (np.log(s/k) + (r - delta + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))

        # Exercise probability factor
        # risk adjusted probability
        d2 = (np.log(s/k) + (r - delta - 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
        return d1, d2

    # Calculate Black-Scholes price of a European Call or Put Option
    def price_option(self):

        d1, d2 = self.calculate_d1_d2()
        # cdf of standard normal distribution, mean = 0, std = 1
        # N(d1): delta of call option, sensitivity to underlying asset
        # N(d2): risk-neutral probability option will be in the money at expiration

        # Call option formula
        call = s*np.exp(-delta*t)*norm.cdf(d1,0,1) - k*np.exp(-r*t)*norm.cdf(d2,0,1)

        # Put option formula
        put = k*np.exp(-r*t)*norm.cdf(-d2,0,1) - s*norm.cdf(-d1,0,1)

        return call, put

# Define model parameters
r = 0.01         # Risk-free rate
s = 25           # Underlying asset price
k = 40           # Strike price
t = 240 / 365    # Time to maturity in years
sigma = 0.30     # Volatility (30%)
delta = 0        # Dividend yield (0 for non-dividend paying)

# Create a Black-Scholes model instance
bs_model = BlackScholesModel(s=s, k=k, t=t, r=r, sigma=sigma, delta=delta)

# Compute option prices
call, put = bs_model.price_option()

# Print results
print("European Call Option Price:", round(call, 2))
print("European Put Option Price :", round(put, 2))
