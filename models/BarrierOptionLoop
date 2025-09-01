import time
import numpy as np

class BarrierOptionLoop:
    def __init__(self, so, k, t, h, r, vol, n, m):
        self.so = so        # Initial asset price
        self.k = k          # Strike price
        self.t = t          # Time to maturity (years)
        self.h = h          # Barrier level (knock-out if breached)
        self.r = r          # Risk-free interest rate
        self.vol = vol      # Volatility of underlying asset
        self.n = n          # Number of time steps
        self.m = m          # Number of Monte Carlo simulations

        # Constants
        self.dt = t / n                             # Time increment per simulation step
        self.nudt = (r - 0.5 * vol**2) * self.dt    # Drift component
        self.volsdt = vol * np.sqrt(self.dt)        # Diffusion
        self.erdt = np.exp(r * self.dt)             # risk-free growth multiplier

    # Function to simulate M asset price using geometric Brownian motion and check barrier
    # Returns discounted option value and standard error
    # Checks each path for a barrier breach, if breached option = $0, if not payoff is max (0, k-st)
    def loop_simulate(self):

        sum_ct = 0 # running total of all simulated payoffs
        sum_ct2 = 0 # running total of squared payoffs, used for variance estimation and SE

        # Simulated M options paths. M: number of Monte Carlo simulation path
        # Simulated asset price and compute the payoff
        for i in range(self.m):
            barrier = False # Track is barrier was breached
            st = self.so # initial spot price

            # inner loop. N: time steps
            for j in range(self.n):
                epsilon = np.random.normal() # standard normal random variables (Brownian shock)
                stn= st*np.exp(self.nudt + self.volsdt * epsilon) # GBM
                st = stn

                # Check for breach
                if st >= self.h:
                    barrier = True
                    break

            # Compute payoff path
            ct = 0 if barrier else max(0, self.k - st)
            sum_ct += ct # accumulate payoff statistics
            sum_ct2 += ct**2

        co = np.exp(-self.r * self.t) * sum_ct / self.m # Compute discounted option value
        # Calculate sigma
        sigma = np.sqrt((sum_ct2 - (sum_ct**2) / self.m) * np.exp(-2 * self.r * self.t) / (self.m - 1))
        se = sigma / np.sqrt(self.m)

        # Return CO: estimated price of down and out european put, SE: Standard error
        return co, se

if __name__ == "__main__":
    # Option parameters
    so = 100  # Initial stock price
    k = 100  # Strike price
    t = 1  # Time to maturity
    h = 125  # Barrier level (up-and-out)
    r = 0.01  # Risk-free interest rate
    vol = 0.2  # Volatility
    n = 100  # Time steps
    m = 1000  # Number of simulations

    # Start timer
    start_time = time.time()

    # Initialize pricer
    euro_option = BarrierOptionLoop(so, k, t, h, r, vol, n, m)

    # Run brute-force pathwise simulation
    co1, se1 = euro_option.loop_simulate()
    print("(Loop) Call Value is ${0} with SE +/- {1}".format(np.round(co1, 2), np.round(se1, 3)))
