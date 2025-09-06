"""

Bootstrapping discount factors from par (swap) rates and providing zero rates / interpolation.

Assumptions:
1. `tenors` is a strictly increasing sequence of maturities in years (e.g. [0.5, 1, 2, ...]).
2. `swap_rates` are par (fixed) swap rates for those maturities, expressed as decimals (e.g. 0.02 for 2%).
3.  Fixed-leg payments occur at each tenor listed (one payment per tenor). The code uses
   delta_k = t_k - t_{k-1} with t_0 = 0 for accrual periods.
      
Mathematical relation (par swap condition):
    S_i * sum_{k=1..i} delta_k * DF(t_k) = 1 - DF(t_i)

Rearranging for the new discount factor:
    DF(t_i) = (1 - S_i * sum_{k=1..i-1} delta_k * DF(t_k)) / (1 + S_i * delta_i)

This recursion lets us build discount factors tenor by tenor.

"""

from typing import Sequence, Union
import numpy as np
from scipy.interpolate import interp1d

# Class for building a discount curve from swap par rates
class BootstrappedYieldCurve:
    def __init__(self, tenors: Sequence[float], swap_rates: Sequence[float]):

        # Store inputs as numpy arrays for vectorized math
        self.tenors = np.asarray(tenors, dtype=float) # Maturities in years at which par rates are quoted
        self.swap_rates = np.asarray(swap_rates, dtype=float) # Par swap rates

        # Check input validity (shape, monotonicity, positivity, etc.)
        self.validate_inputs()

        # Compute accrual fractions: delta_i = t_i - t_{i-1}
        # For the first tenor, this is simply t_1 - 0 = t_1
        self.deltas = np.diff(np.concatenate(([0.0], self.tenors)))

        # Bootstrap discount factors sequentially from par swap equations
        self.discount_factors = self.bootstrap()

        # Build a linear interpolator for discount factors
        # Include DF(0) = 1.0 to guarantee correct behavior at t=0
        interp_times = np.concatenate(([0.0], self.tenors))
        interp_dfs = np.concatenate(([1.0], self.discount_factors))
        self._df_interp = interp1d(
            interp_times,
            interp_dfs,
            kind="linear",
            fill_value="extrapolate",   # allow queries beyond final tenor
            assume_sorted=True,         # inputs are sorted
        )

    def validate_inputs(self) -> None:
        # Check tenors and swap_rates are well-formed before bootstrapping
        if self.tenors.ndim != 1 or self.swap_rates.ndim != 1:
            raise ValueError("tenors and swap_rates must be 1-D sequences.")
        if len(self.tenors) != len(self.swap_rates):
            raise ValueError("tenors and swap_rates must have the same length.")
        if len(self.tenors) == 0:
            raise ValueError("tenors must contain at least one maturity.")
        if not np.all(np.isfinite(self.tenors)) or not np.all(np.isfinite(self.swap_rates)):
            raise ValueError("tenors and swap_rates must be finite numbers.")
        if not np.all(self.tenors > 0):
            raise ValueError("All tenors must be positive.")
        if not np.all(np.diff(self.tenors) > 0):
            raise ValueError("tenors must be strictly increasing.")
        if not np.all(self.swap_rates >= 0):
            # Negative rates are possible in some markets, so adjust if needed.
            raise ValueError("swap_rates must be non-negative (adjust check if negatives expected).")

    # Method performing bootstrapping of discount factors
    def bootstrap(self) -> np.ndarray:

        """
        At each step i:
        1. We know all earlier DFs up to t_{i-1}
        2. The par swap equation links DF(t_i) to previous DFs and swap rate S_i
        3. Solve algebraically for DF(t_i)
        """
        n = len(self.tenors)
        dfs = np.empty(n, dtype=float)

        for i in range(n):
            S = float(self.swap_rates[i])
            delta_i = float(self.deltas[i])

            # Fixed leg contributions from earlier coupons:
            # A = sum_{k=0..i-1} delta_k * DF(t_k)
            A = 0.0 if i == 0 else float(np.dot(self.deltas[:i], dfs[:i]))

            # From par condition: DF(t_i)*(1 + S*delta_i) = 1 - S*A
            denom = 1.0 + S * delta_i
            numer = 1.0 - S * A

            if denom == 0:
                raise ZeroDivisionError(
                    f"Denominator zero at tenor {self.tenors[i]} (S={S}, delta={delta_i})"
                )

            df_i = numer / denom

            if df_i <= 0:
                # If DF <= 0 bad inputs or unrealistic swap rates
                raise ValueError(
                    f"Computed non-positive DF at tenor {self.tenors[i]:g}: {df_i}. "
                    "Check input rates and maturities."
                )

            dfs[i] = df_i

        return dfs

    # Method to return df at time t
    # Use linear interpolation between bootstrapped points
    # Extrapolate beyond final tenor
    # DF(0) defined as 1.0
    def get_df(self, t: Union[float, Sequence[float], np.ndarray]) -> Union[float, np.ndarray]:
        if np.isscalar(t):
            t_val = float(t)

            if t_val < 0:
                raise ValueError("t must be non-negative.")
            return float(self._df_interp(t_val))

        else:
            t_arr = np.asarray(t, dtype=float)
            if np.any(t_arr < 0):
                raise ValueError("t must be non-negative.")
            return self._df_interp(t_arr)

    # Method to compute zero(Spot) rates from df
    def get_zero_rate(
        self, t: Union[float, Sequence[float], np.ndarray], compounding: str = "continuous"
    ) -> Union[float, np.ndarray]:

        single_scalar = np.isscalar(t)
        t_arr = np.asarray(t, dtype=float) if not single_scalar else np.array([float(t)])
        if np.any(t_arr < 0):
            raise ValueError("t must be non-negative.")

        df_vals = np.asarray(self.get_df(t_arr))

        if np.any(df_vals <= 0):
            # Zero/negative DFs are invalid for rate computation
            idx = np.where(df_vals <= 0)[0][0]
            raise ValueError(f"Invalid DF at t={t_arr[idx]} -> DF={df_vals[idx]}")

        # Allocate output array
        out = np.empty_like(t_arr, dtype=float)

        if compounding == "continuous":
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(t_arr == 0.0, 0.0, -np.log(df_vals) / t_arr)
        elif compounding == "annual":
            out = np.where(t_arr == 0.0, 0.0, df_vals ** (-1.0 / t_arr) - 1.0)
        elif compounding == "simple":
            out = np.where(t_arr == 0.0, 0.0, (1.0 / df_vals - 1.0) / t_arr)
        else:
            raise ValueError("compounding must be 'continuous', 'annual', or 'simple'")

        return float(out[0]) if single_scalar else out

    def __repr__(self) -> str:
        return (
            f"BootstrappedYieldCurve(n={len(self.tenors)}, "
            f"tenors={self.tenors.tolist()}, swap_rates={self.swap_rates.tolist()})"
        )


if __name__ == "__main__":
    # Example usage
    tenors = [0.5, 1, 2, 3, 5, 7, 10]
    swap_rates = [0.02, 0.022, 0.025, 0.027, 0.030, 0.032, 0.035]

    yc = BootstrappedYieldCurve(tenors, swap_rates)

    # Print bootstrapped discount factors
    print("Discount factors (matching tenors):")
    for t, df in zip(tenors, yc.discount_factors):
        print(f"  t={t:>4g}y -> DF={df:.10f}")

    # Query zero rates
    print("\nZero rate at 5y (continuous):", yc.get_zero_rate(5.0, compounding="continuous"))
    print("Zero rates (vectorized):", yc.get_zero_rate([0.5, 1.0, 5.0]))

    # Validation: recompute par rates from DFs and confirm match
    deltas = np.diff(np.concatenate(([0.0], tenors)))
    dfs = yc.discount_factors
    for i, S_in in enumerate(swap_rates):
        numer = 1.0 - dfs[i]
        denom = float(np.dot(deltas[: i + 1], dfs[: i + 1]))
        S_out = numer / denom
        assert abs(S_in - S_out) < 1e-12, f"Mismatch at tenor {tenors[i]}: {S_in} vs {S_out}"
    print("\nSelf-check: par rates reproduced correctly.")
