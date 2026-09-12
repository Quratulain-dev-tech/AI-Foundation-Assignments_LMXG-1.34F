"""Shared configuration for the research-grade portfolio pipeline.

IMPORTANT: parameters are fixed before the final holdout is evaluated.
"""

from sp500_tickers import SP500_TICKERS

# Full current S&P 500 universe (~500 names, from Wikipedia's constituent
# list). A cross-sectional ranking model needs many stocks per date to have
# a statistically meaningful "ranking" to learn from -- 20 stocks is too
# few. Downloading this many tickers from Yahoo Finance takes a while and
# can hit rate limits; step1 paces requests and retries automatically.
TICKERS = SP500_TICKERS
# Historical research universe. Yahoo Finance's adjusted price history is used.
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"

# Single fixed holdout used by Step 6.
TRAIN_END_DATE = "2023-12-31"

# Development walk-forward ends before the final untouched holdout.
WALK_FORWARD_START_DATE = "2021-01-01"
WALK_FORWARD_END_DATE = "2025-09-30"
FINAL_HOLDOUT_START_DATE = "2025-10-01"
FINAL_HOLDOUT_END_DATE = "2025-12-31"

DATA_FOLDER = "data"
FEATURES_FOLDER = "features"
OUTPUTS_FOLDER = "outputs"

# Quarterly rebalancing with at least 3 years of formation history.
WALK_FORWARD_REBALANCE_MONTHS = 3
WALK_FORWARD_MIN_FORMATION_MONTHS = 36

# Diversification / governance constraints.
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.05
# NOTE: with validation-based skill shrinkage, most predicted returns end up
# close to zero (the model is honestly reporting it found little edge).
# A 2% risk-free hurdle then makes max_sharpe infeasible almost every window.
# 0.0 lets the optimizer work with whatever small, validated signal exists,
# without pretending the model has more skill than the validation step found.
RISK_FREE_RATE = 0.0

# 10 bps per dollar traded, applied to actual turnover.
TRANSACTION_COST = 0.001

# Medium-horizon prediction target: next 20 trading days.
HORIZON_DAYS = 20
MIN_FORMATION_ROWS = 300

# Model-governance bound. This is a safety bound, not a tuning target.
MAX_ANNUAL_EXPECTED_RETURN = 0.50
MIN_ANNUAL_EXPECTED_RETURN = -0.50

# --- Cross-sectional ranking model settings ---
# Minimum pooled (date x ticker) rows required before the cross-sectional
# model is trusted to fit at all (guards against fitting on too little data
# early in the sample, when few tickers/dates are available).
CROSS_SECTIONAL_MIN_ROWS = 20000
# Maximum annualized tilt (up or down) applied around a stock's own
# historical-mean return, at full (IC=1.0) confidence. Actual tilt is
# scaled down by the model's validated Information Coefficient, so with
# no proven skill (IC=0) the tilt is exactly zero.
CROSS_SECTIONAL_TILT_SCALE = 0.15
