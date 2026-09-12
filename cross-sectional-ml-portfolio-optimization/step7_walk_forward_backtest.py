"""STEP 7: research-grade development walk-forward backtest.

CHANGED FROM PREVIOUS VERSION: expected returns now come from the pooled
cross-sectional ranking model (cross_sectional_model.py) instead of
per-stock absolute-return regression. Everything else -- point-in-time
formation cutoffs, Ledoit-Wolf covariance from formation-period-only
prices, turnover costs, comparison against rebalanced and buy-and-hold
equal-weight baselines, and the untouched Q4 2025 final holdout -- is
unchanged, because that discipline was already sound.
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
    WALK_FORWARD_START_DATE, WALK_FORWARD_END_DATE,
    WALK_FORWARD_REBALANCE_MONTHS, WALK_FORWARD_MIN_FORMATION_MONTHS,
    MIN_WEIGHT, MAX_WEIGHT, RISK_FREE_RATE, TRANSACTION_COST,
    MIN_FORMATION_ROWS, MAX_ANNUAL_EXPECTED_RETURN, MIN_ANNUAL_EXPECTED_RETURN,
    CROSS_SECTIONAL_MIN_ROWS, CROSS_SECTIONAL_TILT_SCALE,
)
from cross_sectional_model import train_cross_sectional, scores_to_expected_returns


def build_cutoffs(index):
    start = max(
        pd.Timestamp(WALK_FORWARD_START_DATE),
        index.min() + pd.DateOffset(months=WALK_FORWARD_MIN_FORMATION_MONTHS),
    )
    end = min(pd.Timestamp(WALK_FORWARD_END_DATE), index.max())
    cuts = []
    c = start
    while c <= end:
        cuts.append(c)
        c += pd.DateOffset(months=WALK_FORWARD_REBALANCE_MONTHS)
    return cuts


def compute_mu(features, cutoff):
    """Runs the pooled cross-sectional model at this cutoff and converts
    its output into per-ticker expected annual returns. Returns (mu_dict, ic)
    or (None, None) if there wasn't enough pooled data yet."""
    result = train_cross_sectional(features, cutoff, min_rows=CROSS_SECTIONAL_MIN_ROWS)
    if result is None:
        return None, None

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
    return mu, result["ic"]


def clean_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    """Symmetrize and lightly regularize so the solver doesn't choke on
    tiny numerical asymmetry / near-singularity (same fix applied in
    step5 -- this is what turned step5's 'infeasible' error into a
    working optimization)."""
    vals = cov.values
    vals = (vals + vals.T) / 2.0
    eps = 1e-6 * np.trace(vals) / vals.shape[0]
    vals = vals + eps * np.eye(vals.shape[0])
    return pd.DataFrame(vals, index=cov.index, columns=cov.columns)


def optimize(mu, cov, window_label=""):
    cov = clean_covariance(cov)
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        return pd.Series(ef.clean_weights()).reindex(mu.index).fillna(0.0)
    except Exception as exc:
        # print (not warnings.warn) -- warnings with the same message/location
        # only show ONCE by default, so a repeated failure across many
        # walk-forward windows would otherwise go completely silent.
        print(f"  [{window_label}] max_sharpe failed ({exc}); skipping this window.")
        return None


def apply_turnover_cost(daily, old_w, new_w):
    turnover = float(np.abs(new_w - old_w).sum())
    cost = TRANSACTION_COST * turnover
    out = daily.copy()
    if len(out):
        out.iloc[0] -= cost
    return out, turnover, cost


def max_drawdown(cum):
    return (cum / cum.cummax() - 1).min()


def annualized_return(daily):
    growth = (1 + daily).prod()
    years = len(daily) / 252
    return growth ** (1 / years) - 1 if years > 0 else np.nan


def sharpe(daily):
    if daily.std() == 0:
        return np.nan
    return ((daily.mean() - RISK_FREE_RATE / 252) / daily.std()) * np.sqrt(252)


def main():
    prices = {}
    features = {}
    for ticker in TICKERS:
        p = os.path.join(DATA_FOLDER, f"{ticker}.csv")
        f = os.path.join(FEATURES_FOLDER, f"{ticker}_features.csv")
        if os.path.exists(p) and os.path.exists(f):
            prices[ticker] = pd.read_csv(p, index_col=0, parse_dates=True)["Close"]
            features[ticker] = pd.read_csv(f, index_col=0, parse_dates=True)

    prices_df = pd.DataFrame(prices).dropna(how="all")
    prices_df = prices_df.dropna(axis=1)  # keep only tickers with a complete history
    if len(prices_df.columns) < 30:
        raise ValueError(
            f"Only {len(prices_df.columns)} tickers have complete price history. "
            "The cross-sectional model needs a broad universe -- check step1/step2 output."
        )

    cutoffs = build_cutoffs(prices_df.index)
    print(
        f"Development walk-forward: {len(cutoffs)} rebalances, "
        f"{WALK_FORWARD_REBALANCE_MONTHS}-month OOS periods, "
        f">={WALK_FORWARD_MIN_FORMATION_MONTHS} months formation, "
        f"{len(prices_df.columns)} tickers in universe."
    )
    print(f"Development end: {WALK_FORWARD_END_DATE}; Q4 2025 is reserved for final holdout.")

    all_opt, all_eq = [], []
    rows = []
    prev_opt = None
    prev_eq = None
    first_oos_date = None

    for i, cutoff in enumerate(cutoffs, 1):
        next_cutoff = cutoff + pd.DateOffset(months=WALK_FORWARD_REBALANCE_MONTHS)

        mu, ic = compute_mu(features, cutoff)
        if mu is None:
            print(f"Window {i:02d}: skipped (not enough pooled data yet at {cutoff.date()})")
            continue

        usable = [t for t in mu if t in prices_df.columns]
        if len(usable) < 30:
            continue
        formation_prices = prices_df.loc[prices_df.index <= cutoff, usable]
        ret = formation_prices.pct_change().dropna()
        if len(ret) < MIN_FORMATION_ROWS:
            continue

        cov = pd.DataFrame(
            LedoitWolf().fit(ret.values).covariance_ * 252,
            index=usable, columns=usable,
        )
        w_opt = optimize(pd.Series(mu).reindex(usable), cov, window_label=f"Window {i:02d} ({cutoff.date()})")
        if w_opt is None:
            continue
        w_eq = pd.Series(1 / len(usable), index=usable)

        old_opt = pd.Series(0.0, index=usable) if prev_opt is None else prev_opt.reindex(usable).fillna(0.0)
        old_eq = pd.Series(0.0, index=usable) if prev_eq is None else prev_eq.reindex(usable).fillna(0.0)

        segment = prices_df.loc[(prices_df.index > cutoff) & (prices_df.index <= next_cutoff), usable]
        segment = segment[segment.index <= pd.Timestamp(WALK_FORWARD_END_DATE)]
        if len(segment) < 2:
            continue
        r = segment.pct_change().dropna()
        if r.empty:
            continue
        if first_oos_date is None:
            first_oos_date = r.index.min()

        opt_daily = (r * w_opt).sum(axis=1)
        eq_daily = (r * w_eq).sum(axis=1)
        opt_daily, opt_turnover, opt_cost = apply_turnover_cost(opt_daily, old_opt, w_opt)
        eq_daily, _, _ = apply_turnover_cost(eq_daily, old_eq, w_eq)

        all_opt.append(opt_daily)
        all_eq.append(eq_daily)
        prev_opt, prev_eq = w_opt, w_eq

        opt_ret = (1 + opt_daily).prod() - 1
        eq_ret = (1 + eq_daily).prod() - 1
        print(
            f"Window {i:02d} [{r.index.min().date()} -> {r.index.max().date()}] "
            f"N={len(usable)} IC={ic:.4f}: "
            f"optimized {opt_ret:+.2%} | equal {eq_ret:+.2%} | turnover {opt_turnover:.2f}"
        )
        rows.append({
            "Window": i, "Cutoff": cutoff.date(), "Start": r.index.min().date(), "End": r.index.max().date(),
            "N_Tickers": len(usable), "Validation_IC": ic,
            "Optimized_Return": opt_ret, "Equal_Weight_Return": eq_ret,
            "Optimized_Turnover": opt_turnover, "Transaction_Cost": opt_cost,
            "Beat_EqualWeight": opt_ret > eq_ret,
        })

    print(f"\n{len(all_opt)} of {len(cutoffs)} candidate cutoffs produced a usable OOS window "
          f"({len(cutoffs) - len(all_opt)} skipped -- see reasons printed above).")

    if not all_opt:
        raise ValueError("No valid walk-forward windows were produced.")

    opt = pd.concat(all_opt).sort_index()
    eq = pd.concat(all_eq).sort_index()
    opt_cum = (1 + opt).cumprod()
    eq_cum = (1 + eq).cumprod()

    bh_prices = prices_df.loc[first_oos_date:WALK_FORWARD_END_DATE]
    bh_growth = bh_prices.div(bh_prices.iloc[0]).mean(axis=1)
    bh = bh_growth.pct_change().dropna()
    bh_cum = bh_growth / bh_growth.iloc[0]

    summary = pd.DataFrame([
        {"Strategy": "Cross-Sectional ML + Risk", "Total_Growth": opt_cum.iloc[-1],
         "Annualized_Return": annualized_return(opt), "Volatility": opt.std()*np.sqrt(252),
         "Sharpe": sharpe(opt), "Max_Drawdown": max_drawdown(opt_cum)},
        {"Strategy": "Equal Weight Rebalanced", "Total_Growth": eq_cum.iloc[-1],
         "Annualized_Return": annualized_return(eq), "Volatility": eq.std()*np.sqrt(252),
         "Sharpe": sharpe(eq), "Max_Drawdown": max_drawdown(eq_cum)},
        {"Strategy": "Equal Weight Buy & Hold", "Total_Growth": bh_cum.iloc[-1],
         "Annualized_Return": annualized_return(bh), "Volatility": bh.std()*np.sqrt(252),
         "Sharpe": sharpe(bh), "Max_Drawdown": max_drawdown(bh_cum)},
    ])

    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS_FOLDER, "walk_forward_window_results.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUTS_FOLDER, "walk_forward_summary.csv"), index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(opt_cum.index, opt_cum.values, label="Cross-Sectional ML + Risk")
    plt.plot(eq_cum.index, eq_cum.values, label="Equal Weight Rebalanced")
    plt.plot(bh_cum.index, bh_cum.values, label="Equal Weight Buy & Hold")
    plt.title("Development Walk-Forward Out-of-Sample Comparison")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUTS_FOLDER, "walk_forward_backtest.png"), dpi=180, bbox_inches="tight")
    plt.close()

    print("\n--- DEVELOPMENT WALK-FORWARD SUMMARY ---")
    print(summary.to_string(index=False, formatters={
        "Total_Growth": "{:.2f}x".format,
        "Annualized_Return": "{:.2%}".format,
        "Volatility": "{:.2%}".format,
        "Sharpe": "{:.2f}".format,
        "Max_Drawdown": "{:.2%}".format,
    }))
    print(f"\nOOS windows evaluated: {len(rows)}")
    print(f"Optimized beat equal-weight in {sum(r['Beat_EqualWeight'] for r in rows)}/{len(rows)} windows.")
    avg_ic = np.mean([r['Validation_IC'] for r in rows])
    print(f"Average validation IC across windows: {avg_ic:.4f}")
    print("Final holdout: 2025-10-01 to 2025-12-31; do not tune the project using it.")


if __name__ == "__main__":
    main()