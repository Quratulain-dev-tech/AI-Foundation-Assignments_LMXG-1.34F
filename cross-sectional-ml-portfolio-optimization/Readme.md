# Cross-Sectional ML Portfolio Optimization — A Research-Grade Walk-Forward Study

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-HistGradientBoosting-orange.svg)](https://scikit-learn.org/)
[![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-Max--Sharpe-green.svg)](https://pyportfolioopt.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

> A full, leakage-controlled ML pipeline that ranks ~500 S&P 500 stocks
> cross-sectionally, builds a risk-aware max-Sharpe portfolio around that
> ranking, and honestly reports how it performs out-of-sample — including
> on a final holdout quarter the model never saw during development.

This is not a "buy this stock" tool. It's an end-to-end demonstration of how
a real quantitative research pipeline is built: point-in-time data, a
cross-sectional ranking model, signal governance, shrinkage risk estimation,
constrained optimization, transaction costs, walk-forward validation, and a
locked final holdout — with every honest result reported, including the
ones that show the model *didn't* beat a naive benchmark.

---

## Table of Contents
- [Why cross-sectional ranking, not per-stock prediction](#why-cross-sectional-ranking-not-per-stock-prediction)
- [Pipeline](#pipeline)
- [Leakage controls](#leakage-controls)
- [Results](#results)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
- [Key engineering decisions](#key-engineering-decisions)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)

---

## Why cross-sectional ranking, not per-stock prediction

An earlier version of this project trained one regression model **per
stock** to predict that stock's own future return in isolation. That
approach consistently fails in practice, for a simple reason: a stock's
daily return is dominated by market-wide noise that is nearly impossible to
predict stock-by-stock.

Following the dominant approach in empirical asset-pricing ML research (Gu,
Kelly & Xiu, *Empirical Asset Pricing via Machine Learning*, Review of
Financial Studies, 2020, and its follow-ups), this project instead:

1. Pools **every stock's** feature history into one large dataset.
2. Trains **one** gradient-boosted model to answer *"which stocks will do
   relatively better than their peers over the next 20 trading days?"* —
   not "what will this stock's absolute return be?"
3. Cross-sectionally z-scores every feature **within each date**, so the
   model only ever sees each stock's *relative* standing on that day —
   the shared market-wide component is removed before the model even sees
   the data.

This is also why the universe was expanded from an initial ~20-stock list
to the full S&P 500 (~500 names): ranking needs many stocks per date to be
a statistically meaningful cross-section to learn from.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_data_collection.py` | Downloads daily OHLCV for ~500 S&P 500 tickers via `yfinance` (resumable, rate-limit-paced). |
| 2 | `step2_feature_engineering.py` | Builds momentum, moving-average gap, RSI, volume, and volatility features per ticker. |
| 3 | `step3_return_prediction.py` | Trains the pooled cross-sectional ranking model; reports validation Information Coefficient (IC). |
| 4 | `step4_risk_model.py` | Estimates a Ledoit-Wolf shrinkage covariance matrix from formation-period-only returns. |
| 5 | `step5_portfolio_optimization.py` | Solves a constrained max-Sharpe portfolio (position cap, long-only). |
| 6 | `step6_backtest_evaluation.py` | Evaluates that single static portfolio against an equal-weight benchmark. |
| 7 | `step7_walk_forward_backtest.py` | Rebuilds the model and portfolio every quarter (expanding formation window) and compares against equal-weight, rebalanced and buy-and-hold. |
| 8 | `step8_final_holdout.py` | Trains once at 2025-09-30, evaluates **only** on Q4 2025 — a period never touched during development. |

`cross_sectional_model.py` and `config.py` are shared across steps 3, 7,
and 8, so the exact same model logic and governance parameters are used
everywhere — no silently-different definitions between the development and
final-holdout code paths.

## Leakage controls

- At every rebalance date, training rows are restricted to observations
  whose **target end date is on or before the cutoff** — a 20-day-forward
  label can never cross the formation boundary.
- The risk model (Ledoit-Wolf covariance) is estimated using **only**
  prices available as of the cutoff.
- Portfolio weights for an out-of-sample segment are computed **before**
  that segment begins, never using information from within it.
- **Signal governance:** the model's real skill is measured with the
  Information Coefficient (IC) — the standard cross-sectional metric
  (daily Spearman rank correlation between predicted rank and actual
  forward return), not accuracy or MAE. Each stock's expected return is
  its own historical mean, tilted by the ranking signal **scaled by the
  validated IC**. An IC of 0 means exactly zero tilt: the code is
  structurally incapable of pretending to have found an edge it didn't
  validate.

## Results

### Development walk-forward (19 quarterly rebalances, 2021 → Sep 2025)

| Strategy | Total Growth | Annualized Return | Sharpe | Max Drawdown |
|---|---|---|---|---|
| Cross-Sectional ML + Risk | 2.02x | 16.27% | 1.00 | -21.33% |
| Equal Weight (Rebalanced) | 2.08x | 17.09% | 1.03 | -21.28% |
| Equal Weight (Buy & Hold) | 2.01x | 15.91% | 0.96 | -19.07% |

- Optimized strategy beat the equal-weight benchmark in **10 of 19
  windows (53%)** — essentially coin-flip odds, not a consistent edge.
- Average validation IC across windows: **0.0145** — a genuinely weak
  signal, consistent with the honest expectation set in this project's
  governance design.

### Final untouched holdout (Q4 2025, Oct 2 → Dec 31)

| Strategy | Total Growth | Return |
|---|---|---|
| Cross-Sectional ML + Risk | 0.992x | **-0.83%** |
| Equal Weight (Rebalanced) | 1.023x | **+2.25%** |
| Equal Weight (Buy & Hold) | 1.022x | **+2.19%** |

Validation IC on this final training window: **0.0262**. The final
portfolio held 31 of 469 eligible tickers (position cap 5%).

**Honest conclusion:** the pipeline is sound and leakage-controlled, but
across both the development walk-forward and the locked final holdout, the
model did not demonstrate a reliable, tradeable edge over a simple
equal-weight benchmark. This is reported as a finding, not hidden or
tuned away — which is the entire point of freezing the holdout in
advance.

## Repository structure

```
research_grade_v2/
├── config.py                       # All governance parameters, fixed before the final holdout
├── sp500_tickers.py                 # ~500-ticker research universe
├── step1_data_collection.py
├── step2_feature_engineering.py
├── step3_return_prediction.py
├── cross_sectional_model.py         # Shared ranking model (steps 3, 7, 8)
├── step4_risk_model.py
├── step5_portfolio_optimization.py
├── step6_backtest_evaluation.py
├── step7_walk_forward_backtest.py
├── step8_final_holdout.py
├── requirements.txt
├── data/                            # Raw OHLCV (generated, gitignored)
├── features/                        # Engineered features (generated, gitignored)
└── outputs/                         # All results, csvs, charts (generated, gitignored)
```

## How to run

```bash
pip install -r requirements.txt

python step1_data_collection.py        # ~500 tickers, ~2018-2026 daily OHLCV
python step2_feature_engineering.py
python step3_return_prediction.py
python step4_risk_model.py
python step5_portfolio_optimization.py
python step6_backtest_evaluation.py
python step7_walk_forward_backtest.py
python step8_final_holdout.py          # run this LAST, and only once
```

### Key outputs
- `outputs/walk_forward_summary.csv`, `outputs/walk_forward_backtest.png`
- `outputs/final_holdout_summary.csv`, `outputs/final_holdout_backtest.png`
- `outputs/optimal_allocation.csv`, `outputs/covariance_matrix.csv`

## Key engineering decisions

- **Ledoit-Wolf shrinkage covariance** instead of the raw sample
  covariance — far more stable when estimating a ~500x500 covariance
  matrix from a few years of daily data.
- **Position cap (5% max, 0% min)** instead of a fixed minimum-weight
  floor — with ~500 candidate assets, a naive "everyone gets at least 1%"
  constraint is mathematically infeasible past ~100 assets.
- **Covariance regularization** (symmetrization + a small numerical
  ridge) before every optimization call — a covariance matrix that
  round-trips through a CSV or gets sliced/reindexed can pick up tiny
  floating-point asymmetry, which is enough to make an off-the-shelf
  convex solver report "infeasible" even though the matrix is
  economically fine.
- **IC-scaled tilting, not raw model output** — expected returns are each
  stock's own historical mean, tilted by the model's rank signal scaled
  by validated out-of-sample IC. This is a deliberate governance choice
  so that a weak or unproven model cannot generate large, false-confidence
  return forecasts.
- **A frozen final holdout quarter** — Q4 2025 was never used to select
  features, tune hyperparameters, or adjust position limits. It was
  evaluated exactly once, after every other decision was already fixed.

## Limitations

- The universe is today's S&P 500 constituent list applied to historical
  dates — **survivorship bias is not corrected for** (stocks removed from
  the index during the sample period are simply absent).
- Yahoo Finance data is not a point-in-time institutional data feed.
- No bid/ask spread, market impact, taxes, borrow costs, or liquidity
  constraints are modeled; only a flat per-dollar transaction cost on
  turnover.
- Position limits (0%–5%) were a development-time judgment call, not
  tuned against the final holdout.
- A genuinely low validation IC (0.01–0.05, as found here) means the
  strategy behaves close to a risk-managed, diversified portfolio — not a
  demonstrated source of reliable alpha.

## Disclaimer

This project is a research and engineering exercise. It is **not
investment advice** and is not a guarantee of future performance. No
amount of methodological rigor — leakage controls, walk-forward testing,
or a frozen holdout — turns a backtest into a promise about the future.
Consult a licensed financial advisor before allocating real capital based
on anything in this repository.

## License

MIT — see [LICENSE](LICENSE).
