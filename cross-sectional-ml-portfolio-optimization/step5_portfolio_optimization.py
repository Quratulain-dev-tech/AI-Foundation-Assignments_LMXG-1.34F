"""STEP 5: constrained maximum-Sharpe portfolio optimization."""

import os
import numpy as np
import pandas as pd
import cvxpy as cp
from pypfopt import EfficientFrontier

from config import OUTPUTS_FOLDER, TRAIN_END_DATE, MIN_WEIGHT, MAX_WEIGHT, RISK_FREE_RATE


def load_inputs():
    pred = pd.read_csv(os.path.join(OUTPUTS_FOLDER, "predicted_returns.csv"), index_col="Ticker")
    cov = pd.read_csv(os.path.join(OUTPUTS_FOLDER, "covariance_matrix.csv"), index_col=0)
    common = pred.index.intersection(cov.index)
    mu = pred.loc[common, "Expected_Annual_Return"].astype(float)
    S = cov.loc[common, common].astype(float)

    bad = mu.index[mu.isna() | np.isinf(mu)]
    bad = bad.union(S.index[S.isna().any(axis=1) | np.isinf(S.values).any(axis=1)])
    if len(bad) > 0:
        print(f"Dropping {len(bad)} ticker(s) with invalid (NaN/Inf) values before optimizing: {list(bad)}")
        keep = mu.index.difference(bad)
        mu, S = mu.loc[keep], S.loc[keep, keep]

    Sv = S.values
    Sv = (Sv + Sv.T) / 2
    S = pd.DataFrame(Sv, index=S.index, columns=S.columns)
    return mu, S


def try_optimize(mu, S, objective, risk_free_rate, min_w, max_w):
    """Tries every solver cvxpy reports as installed, printing each outcome
    explicitly so a repeat failure is fully diagnosable from the log alone."""
    solvers = [None] + cp.installed_solvers()
    seen = set()
    last_error = None
    for solver in solvers:
        key = solver or "DEFAULT"
        if key in seen:
            continue
        seen.add(key)
        try:
            kwargs = {} if solver is None else {"solver": solver}
            ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w), **kwargs)
            if objective == "max_sharpe":
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            else:
                ef.min_volatility()
            print(f"  [{objective}] solver={key}: SUCCESS")
            return ef
        except Exception as exc:
            print(f"  [{objective}] solver={key}: FAILED -> {exc}")
            last_error = exc
    raise last_error


def main():
    print("cvxpy solvers available on this machine:", cp.installed_solvers())
    print(f"Raw config values -- MIN_WEIGHT: {MIN_WEIGHT!r} (type {type(MIN_WEIGHT).__name__}), "
          f"MAX_WEIGHT: {MAX_WEIGHT!r} (type {type(MAX_WEIGHT).__name__})")
    min_w = float(MIN_WEIGHT)
    max_w = float(MAX_WEIGHT)

    mu, S = load_inputs()
    n = len(mu)

    print(f"\nConfig check: MIN_WEIGHT={min_w}, MAX_WEIGHT={max_w}, universe size={n}")
    min_total = min_w * n
    max_total = max_w * n
    print(f"With these bounds, total portfolio weight can range from {min_total:.2f} to {max_total:.2f} "
          f"(must be able to reach exactly 1.0 to be feasible).")
    if min_total > 1.0 + 1e-9:
        raise RuntimeError(
            f"INFEASIBLE BY CONSTRUCTION: MIN_WEIGHT ({min_w}) x {n} tickers = {min_total:.2f}, "
            f"which is more than 100% -- there is no way to satisfy 'every ticker gets at least "
            f"{min_w:.2%}' while weights sum to 1. This means config.py still has an old/incorrect "
            f"MIN_WEIGHT value. Open config.py and confirm the line reads exactly: MIN_WEIGHT = 0.0"
        )
    if max_total < 1.0 - 1e-9:
        raise RuntimeError(
            f"INFEASIBLE BY CONSTRUCTION: MAX_WEIGHT ({max_w}) x {n} tickers = {max_total:.2f}, "
            f"which is less than 100% -- there is no way to reach full investment. Check config.py's "
            f"MAX_WEIGHT value."
        )

    eigvals = np.linalg.eigvalsh(S.values)
    print(f"Universe size: {len(mu)} tickers. Covariance smallest eigenvalue: {eigvals.min():.2e}, "
          f"condition number: {eigvals.max()/max(eigvals.min(),1e-12):.2e}")

    print("\nAttempting max_sharpe across all available solvers:")
    try:
        ef = try_optimize(mu, S, "max_sharpe", RISK_FREE_RATE, min_w, max_w)
        method = "max_sharpe"
    except Exception:
        print("\nmax_sharpe failed on every installed solver; attempting min_volatility "
              "(a real risk-minimizing portfolio, not a placeholder):")
        try:
            ef = try_optimize(mu, S, "min_volatility", RISK_FREE_RATE, min_w, max_w)
            method = "min_volatility (max_sharpe was numerically infeasible)"
        except Exception as exc2:
            print("\n--- DIAGNOSTICS (every solver failed for both objectives) ---")
            print(f"mu: min={mu.min():.4f} max={mu.max():.4f} NaN={mu.isna().sum()}")
            print(f"S: NaN={S.isna().sum().sum()} smallest eigenvalue={eigvals.min():.2e} "
                  f"condition number={eigvals.max()/max(eigvals.min(),1e-12):.2e}")
            raise RuntimeError(
                "Portfolio optimization failed on every installed solver, for both "
                "objectives. Please pip install osqp and/or clarabel (pip install osqp "
                "clarabel) and re-run -- these are more robust QP solvers than the older "
                "ECOS default for problems this size."
            ) from exc2

    weights = ef.clean_weights()
    expected_return, volatility, sharpe = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    print(f"\nOptimization method used: {method}")
    print("Portfolio Allocation:")
    for ticker, w in weights.items():
        if w > 0:
            print(f"  {ticker}: {w:.2%}")
    print(f"\nFormation-period expected return: {expected_return:.2%}")
    print(f"Formation-period volatility: {volatility:.2%}")
    print(f"Formation-period Sharpe: {sharpe:.2f}")
    print(f"Formation cutoff: {TRAIN_END_DATE}")
    print("WARNING: formation metrics are in-sample and are NOT final performance metrics.")

    pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"]).to_csv(
        os.path.join(OUTPUTS_FOLDER, "optimal_allocation.csv"), index=False
    )
    print(f"\nSaved: {os.path.join(OUTPUTS_FOLDER, 'optimal_allocation.csv')}")


if __name__ == "__main__":
    main()