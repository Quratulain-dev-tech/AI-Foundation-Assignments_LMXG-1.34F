"""STEP 8: FINAL UNTOUCHED HOLDOUT.

This is the final model evaluation after all development decisions are fixed.
The model is trained at 2025-09-30 and evaluated only on Q4 2025.
Do NOT use this result to tune features, models, weights, or dates.

CHANGED FROM PREVIOUS VERSION: uses the pooled cross-sectional ranking
model (cross_sectional_model.py) instead of per-stock absolute regression.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from pypfopt import EfficientFrontier

from config import (
    TICKERS, DATA_FOLDER, FEATURES_FOLDER, OUTPUTS_FOLDER,
    FINAL_HOLDOUT_START_DATE, FINAL_HOLDOUT_END_DATE,
    MIN_WEIGHT, MAX_WEIGHT, RISK_FREE_RATE, TRANSACTION_COST,
    MIN_FORMATION_ROWS, MAX_ANNUAL_EXPECTED_RETURN, MIN_ANNUAL_EXPECTED_RETURN,
    CROSS_SECTIONAL_MIN_ROWS, CROSS_SECTIONAL_TILT_SCALE,
)
from cross_sectional_model import train_cross_sectional, scores_to_expected_returns

CUTOFF = pd.Timestamp("2025-09-30")


def compute_mu(features, cutoff):
    result = train_cross_sectional(features, cutoff, min_rows=CROSS_SECTIONAL_MIN_ROWS)
    if result is None:
        return None, None, None

    hist_mean = {}
    for ticker, df in features.items():
        d = df[df.index <= cutoff]
        if len(d) > 0:
            hist_mean[ticker] = float(d["daily_return"].mean() * 252)

    mu = scores_to_expected_returns(
        result["scores"], result["ic"], hist_mean,
        tilt_scale=CROSS_SECTIONAL_TILT_SCALE,
        min_ret=MIN_ANNUAL_EXPECTED_RETURN, max_ret=MAX_ANNUAL_EXPECTED_RETURN,
    )
    return mu, result["ic"], result["n_train_rows"]


def clean_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    """Symmetrize and lightly regularize (same fix used in step5/step7 --
    prevents a spurious 'infeasible' solver status from a numerically
    near-singular covariance matrix)."""
    vals = cov.values
    vals = (vals + vals.T) / 2.0
    eps = 1e-6 * np.trace(vals) / vals.shape[0]
    vals = vals + eps * np.eye(vals.shape[0])
    return pd.DataFrame(vals, index=cov.index, columns=cov.columns)


def optimize(mu, cov):
    cov = clean_covariance(cov)
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        return pd.Series(ef.clean_weights()).reindex(mu.index).fillna(0)
    except Exception as exc:
        print(f"max_sharpe failed ({exc}).")
        return None


def main():
    prices = {}
    features = {}
    for ticker in TICKERS:
        p = os.path.join(DATA_FOLDER, f"{ticker}.csv")
        f = os.path.join(FEATURES_FOLDER, f"{ticker}_features.csv")
        if os.path.exists(p) and os.path.exists(f):
            prices[ticker] = pd.read_csv(p, index_col=0, parse_dates=True)["Close"]
            features[ticker] = pd.read_csv(f, index_col=0, parse_dates=True)
    prices_df = pd.DataFrame(prices).dropna(axis=1)

    mu, ic, n_rows = compute_mu(features, CUTOFF)
    if mu is None:
        raise ValueError("Not enough pooled data to train the final holdout model.")

    usable = [t for t in mu if t in prices_df.columns]
    formation = prices_df.loc[prices_df.index <= CUTOFF, usable]
    ret = formation.pct_change().dropna()
    if len(ret) < MIN_FORMATION_ROWS:
        raise ValueError("Not enough formation-period price history for the final holdout.")

    cov = pd.DataFrame(LedoitWolf().fit(ret.values).covariance_ * 252, index=usable, columns=usable)
    w = optimize(pd.Series(mu).reindex(usable), cov)
    if w is None:
        raise ValueError("Could not optimize final holdout portfolio.")

    holdout = prices_df.loc[
        (prices_df.index >= pd.Timestamp(FINAL_HOLDOUT_START_DATE)) &
        (prices_df.index <= pd.Timestamp(FINAL_HOLDOUT_END_DATE)), usable
    ]
    r = holdout.pct_change().dropna()
    if r.empty:
        raise ValueError("Final holdout has no usable return observations.")

    opt = (r * w).sum(axis=1)
    turnover = float(np.abs(w).sum())
    opt.iloc[0] -= TRANSACTION_COST * turnover
    eqw = pd.Series(1 / len(usable), index=usable)
    eq = (r * eqw).sum(axis=1)

    bh_growth = holdout.div(holdout.iloc[0]).mean(axis=1)
    bh = bh_growth.pct_change().dropna()

    cum = pd.DataFrame({"Cross-Sectional ML + Risk": (1 + opt).cumprod(),
                        "Equal Weight": (1 + eq).cumprod(),
                        "Equal Weight Buy & Hold": bh_growth / bh_growth.iloc[0]})
    cum.to_csv(os.path.join(OUTPUTS_FOLDER, "final_holdout_equity_curves.csv"))

    summary = pd.DataFrame([
        {"Strategy": name, "Total_Growth": curve.iloc[-1], "Return": curve.iloc[-1] - 1}
        for name, curve in cum.items()
    ])
    summary.to_csv(os.path.join(OUTPUTS_FOLDER, "final_holdout_summary.csv"), index=False)

    plt.figure(figsize=(11, 6))
    for col in cum.columns:
        plt.plot(cum.index, cum[col], label=col)
    plt.title("FINAL HOLDOUT — Q4 2025 (Untouched Before Evaluation)")
    plt.xlabel("Date"); plt.ylabel("Growth of $1"); plt.legend(); plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(OUTPUTS_FOLDER, "final_holdout_backtest.png"), dpi=180, bbox_inches="tight")
    plt.close()

    print("\n--- FINAL UNTOUCHED HOLDOUT ---")
    print(f"Training/formation cutoff: {CUTOFF.date()}")
    print(f"Holdout: {r.index.min().date()} -> {r.index.max().date()} ({len(r)} daily returns)")
    print(f"Pooled cross-sectional training rows: {n_rows}")
    print(f"Validation Information Coefficient (IC): {ic:.4f}")
    print(f"Number of tickers in final portfolio (weight > 0): {(w > 0.0001).sum()} / {len(usable)}")
    print(summary.to_string(index=False, formatters={"Total_Growth": "{:.3f}x".format, "Return": "{:.2%}".format}))
    print("\nIMPORTANT: This holdout must remain untouched for honest reporting.")


if __name__ == "__main__":
    main()