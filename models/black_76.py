"""

Black's (1976) caplet / floorlet pricer (Black-76).


This module implements the Black model for interest-rate caplets and floorlets.

Assumptions:
1. It assumes the forward rate F is lognormally distributed and uses the standard


Black formula:

    Caplet = DF * tau * [ F * N(d1) - K * N(d2) ]
    Floorlet = DF * tau * [ K * N(-d2) - F * N(-d1) ]

Price a caplet and a floorlet using Black's model (Black-76).
Caplet provides buyer protection against an increase in forward interest rate
Floorlet provides buyer protection against a decrease in forward interest rate

"""

def black_caplet_price(F, K, DF, T, sigma):

    # Input validation for negative inputs
    if F <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return 0.0, 0.0                                 # Return 0 if invalid inputs

    # d1 and d2 are the standard Black (and Black-Scholes) intermediates.
    # d1 = (ln(F/K) + 0.5*sigma^2 * T) / (sigma*sqrt(T))
    # d2 = d1 - sigma*sqrt(T)
    # These capture the moneyness adjusted for volatility and time to expiry.
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Black's formula for caplet and floorlet:
    caplet = DF * (F * norm.cdf(d1) - K * norm.cdf(d2)) * T
    floorlet = DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1)) * T

    return round(float(caplet),5), round(float(floorlet),5)

# Example
if __name__ == "__main__":
    F = 0.03        # F, Forward Interest Rate
    K = 0.025       # K, Strike Rate
    DF = 0.95       # DF, discount factor to payment date
    T = 1.0         # T, Time to maturity
    sigma = 0.2     # Sigma, Volatility

    caplet, floorlet = black_caplet_price(F, K, DF, T, sigma)

    print(f"Inputs: F={F}, K={K}, DF={DF}, T={T}, sigma={sigma}")
    print(f"Caplet price  : {caplet}")
    print(f"Floorlet price: {floorlet}")
