"""STEP 4: stable covariance estimation using only formation-period data."""

import os
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

from config import TICKERS, DATA_FOLDER, OUTPUTS_FOLDER, TRAIN_END_DATE, MIN_FORMATION_ROWS


def main():
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    prices = {}
    for ticker in TICKERS:
        path = os.path.join(DATA_FOLDER, f"{ticker}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            prices[ticker] = df["Close"]

    # Union of all dates first (do NOT drop rows yet) -- recently listed /
    # spun-off tickers (e.g. a 2025 IPO) will have NaN before their listing
    # date, and that is expected, not an error.
    prices_df = pd.DataFrame(prices)
    prices_df = prices_df.dropna(how="all")

    formation = prices_df[prices_df.index <= TRAIN_END_DATE]

    # Only keep tickers that actually have enough formation-period history
    # to be usable (e.g. drop 2024/2025 IPOs and spin-offs from the
    # formation-period covariance -- they simply didn't exist yet).
    enough_history = formation.count() >= MIN_FORMATION_ROWS
    usable_tickers = enough_history[enough_history].index.tolist()
    dropped = sorted(set(prices_df.columns) - set(usable_tickers))
    if dropped:
        print(f"Excluding {len(dropped)} tickers from the covariance/backtest universe "
              f"(insufficient formation-period history, e.g. recent IPOs/spin-offs): {dropped}")

    prices_df = prices_df[usable_tickers]
    prices_df.to_csv(os.path.join(OUTPUTS_FOLDER, "combined_prices.csv"))

    formation = prices_df[prices_df.index <= TRAIN_END_DATE].dropna()
    daily_returns = formation.pct_change().dropna()
    if daily_returns.empty:
        raise ValueError(
            "No overlapping formation-period returns after filtering. "
            "Check TRAIN_END_DATE / START_DATE in config.py."
        )

    # Ledoit-Wolf shrinkage is more stable than a raw sample covariance
    # when many assets are correlated.
    lw = LedoitWolf().fit(daily_returns.values)
    raw_cov = lw.covariance_ * 252

    # Force exact symmetry (tiny floating-point asymmetry can appear after
    # matrix algebra) and add a small ridge/diagonal-loading term. This is
    # standard, disclosed practice in portfolio optimization: with hundreds
    # of stocks, some pairs are near-duplicates in price behavior (e.g.
    # dual share classes like GOOGL/GOOG, FOX/FOXA, NWS/NWSA all move
    # almost identically), which makes the raw covariance matrix nearly
    # singular / numerically unstable and can make optimizers fail with a
    # false "infeasible" result even though a solution exists. A tiny ridge
    # term (proportional to the average variance) fixes this numerical
    # issue while changing the actual risk estimates by a negligible amount.
    raw_cov = (raw_cov + raw_cov.T) / 2
    ridge = 1e-4 * np.trace(raw_cov) / raw_cov.shape[0]
    raw_cov = raw_cov + np.eye(raw_cov.shape[0]) * ridge

    cov = pd.DataFrame(raw_cov, index=daily_returns.columns, columns=daily_returns.columns)
    cov.to_csv(os.path.join(OUTPUTS_FOLDER, "covariance_matrix.csv"))

    eigvals = np.linalg.eigvalsh(raw_cov)
    print(f"\nSaved Ledoit-Wolf annualized covariance matrix "
          f"(formation period only, {len(usable_tickers)} tickers, ridge={ridge:.2e}).")
    print(f"Smallest eigenvalue: {eigvals.min():.2e} (should be positive)")
    print(cov.round(4))


if __name__ == "__main__":
    main()