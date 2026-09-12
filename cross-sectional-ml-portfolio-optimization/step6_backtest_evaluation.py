"""STEP 6: fixed development holdout evaluation.

This experiment uses the static 2023-12-31 formation cutoff, but only evaluates
through 2025-09-30. Q4 2025 is reserved exclusively for Step 8 and is therefore
not touched by this development experiment.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUTS_FOLDER, TRAIN_END_DATE, WALK_FORWARD_END_DATE, TRANSACTION_COST


def main():
    weights_df = pd.read_csv(os.path.join(OUTPUTS_FOLDER, "optimal_allocation.csv"), index_col="Ticker")
    weights = weights_df["Weight"]

    prices_df = pd.read_csv(
        os.path.join(OUTPUTS_FOLDER, "combined_prices.csv"), index_col=0, parse_dates=True
    )[weights.index]
    backtest_prices = prices_df.loc[
        (prices_df.index > pd.Timestamp(TRAIN_END_DATE)) &
        (prices_df.index <= pd.Timestamp(WALK_FORWARD_END_DATE))
    ]
    if backtest_prices.empty:
        raise ValueError("No development holdout data found. Check the configured dates.")

    returns_df = backtest_prices.pct_change().dropna()
    optimized_daily = (returns_df * weights).sum(axis=1)
    optimized_daily.iloc[0] -= TRANSACTION_COST * float(weights.abs().sum())
    equal_weights = pd.Series(1 / len(weights), index=weights.index)
    equal_daily = (returns_df * equal_weights).sum(axis=1)
    equal_daily.iloc[0] -= TRANSACTION_COST

    optimized_cum = (1 + optimized_daily).cumprod()
    equal_cum = (1 + equal_daily).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(optimized_cum.index, optimized_cum.values, label="Optimized ML + Risk")
    plt.plot(equal_cum.index, equal_cum.values, label="Equal Weight")
    plt.title("Development Holdout: 2024-01-01 to 2025-09-30")
    plt.xlabel("Date"); plt.ylabel("Growth of $1"); plt.legend(); plt.grid(True, alpha=0.3)
    out_path = os.path.join(OUTPUTS_FOLDER, "backtest_comparison.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    print("\n--- DEVELOPMENT HOLDOUT ---")
    print(f"Formation cutoff: {TRAIN_END_DATE}")
    print(f"Evaluation: {backtest_prices.index.min().date()} to {backtest_prices.index.max().date()}")
    print(f"Optimized final growth: {optimized_cum.iloc[-1]:.3f}x")
    print(f"Equal-weight final growth: {equal_cum.iloc[-1]:.3f}x")
    print("This is a development experiment; the final holdout is evaluated only in Step 8.")


if __name__ == "__main__":
    main()
