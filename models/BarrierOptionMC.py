import time
import numpy as np
import matplotlib.pyplot as plt

# Class for Monte Carlo simulator for european up and out put options with discrete barrier monitoring
class BarrierOptionMC:
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

    
    # Function for vectorized version to simulate M asset price paths and apply barrier knockout in bulk
    # returns discounted option value and standard error
    def vector_simulate(self):

        z = np.random.normal(size=(self.n, self.m)) # Generate random normal shocks 2D array (N, M)
        delta_st = self.nudt + self.volsdt * z # relative return

        st = self.so * np.cumprod(np.exp(delta_st), axis=0) # initial price
        st = np.concatenate((np.full(shape=(1, self.m), fill_value=self.so), st)) #ST now have shape (N+1, M)

        s = np.copy(st) # Create copy and identify barrier breaches
        mask = np.any(st >= self.h, axis=0)  # Paths where barrier was breached. Boolean array shape (M,)
        st[:, mask] = 0  # Knocked-out paths become worthless

        # Terminal values of surviving paths
        ct = np.maximum(0, self.k - st[-1][st[-1] != 0]) # Compute payoffs of surviving paths
        co = np.exp(-self.r * self.t) * np.sum(ct) / self.m # Discount average payoff to present
        # Compute SE
        sigma = np.std(np.exp(-self.r * self.t) * ct)
        se = sigma / np.sqrt(self.m)

        return co, se, s, mask

    # Function to plot simulated price paths, knocked out (red) surviving (green)
    def plots(self, s, mask):
        plt.figure(figsize=(8, 6))
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 16
        time_grid = np.linspace(0, self.t, self.n + 1)
        plt.plot(time_grid, s[:, mask], 'r', alpha=0.5)
        plt.plot(time_grid, s[:, ~mask], 'g', alpha=0.5)
        plt.axhline(self.h, color='black', linewidth=1.0)
        plt.annotate('h', (0.05, self.h + 5))
        plt.xlim(0, self.t)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('European Up-and-Out Put Option')
        plt.show()

if __name__ == "__main__":
    # Option parameters
    so = 100     # Initial stock price
    k = 100      # Strike price
    t = 1        # Time to maturity
    h = 125      # Barrier level (up-and-out)
    r = 0.01     # Risk-free interest rate
    vol = 0.2    # Volatility
    n = 100      # Time steps
    m = 1000     # Number of simulations

    # Start timer
    start_time = time.time()

    # Initialize pricer
    euro_option = BarrierOptionMC(so, k, t, h, r, vol, n, m)

    # Run vectorized simulation
    co2, se2, s, mask = euro_option.vector_simulate()
    print("(Vectorized) Call Value is ${0} with SE +/- {1}".format(np.round(co2, 2), np.round(se2, 3)))
    print("Computation time is:", round(time.time() - start_time, 4), "seconds")

    # Plot simulated paths
    euro_option.plots(s, mask)
