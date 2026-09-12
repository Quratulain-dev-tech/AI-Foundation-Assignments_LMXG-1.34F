"""
CROSS-SECTIONAL RANKING MODEL (shared by step3, step7, step8)
-----------------------------------------------------------------

WHY THIS CHANGED FROM THE PREVIOUS VERSION:

The previous version trained one model PER STOCK to predict that stock's
own absolute future return. Recent empirical asset-pricing research
(most notably Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine
Learning", Review of Financial Studies, 2020, and the large body of
follow-up work through 2024-2025) consistently finds that this approach
underperforms a CROSS-SECTIONAL approach: pool every stock's data
together and train ONE model to answer "which stocks will do relatively
BETTER than their peers over the next N days", rather than "what will
this one stock's return be in isolation".

Why cross-sectional ranking tends to work better:
  - Absolute return prediction has to fight market-wide noise (a stock's
    daily return is dominated by overall market moves that are almost
    impossible to predict). Ranking cancels out this shared component,
    because on any given day ALL stocks are exposed to it similarly.
  - Pooling stocks together turns "20 tiny per-stock datasets" into one
    large dataset, which lets a model actually learn stable patterns
    instead of overfitting noise.
  - This is also why the model needs a LARGE universe (hundreds of
    stocks): with only 20 stocks there are only 20 cross-sectional
    "observations" per day, which is too few for ranking to be
    statistically meaningful. Cross-sectional ML papers typically use
    500-3000+ stocks for exactly this reason.

HONESTY NOTE: this is still not guaranteed to find real, tradeable edge.
It is a better-supported approach than what came before, not a promise
of profit. The validation Information Coefficient (IC) computed below is
the metric to look at -- if IC is still ~0, the honest conclusion is
still "no edge found", and the code will (correctly) shrink positions
toward the historical-mean baseline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

FEATURE_COLS = [
    "daily_return", "return_5d", "return_20d", "return_60d",
    "volatility_20d", "volatility_60d", "ma_10_gap", "ma_50_gap",
    "ma_200_gap", "rsi_14", "volume_change_20d", "volume_z_20d",
]
TARGET_COL = "forward_return"


def _cross_sectional_zscore(wide):
    """Z-score each row (i.e. across all stocks on that date). This removes
    the market-wide/common component from each feature, leaving only each
    stock's RELATIVE standing on that day -- the signal cross-sectional
    models are actually trying to learn from."""
    mean = wide.mean(axis=1)
    std = wide.std(axis=1).replace(0, np.nan)
    return wide.sub(mean, axis=0).div(std, axis=0)


def build_panel(features, cutoff):
    """Pool every ticker's feature history (up to `cutoff`) into one long
    (Date, Ticker) panel with cross-sectionally normalized features and a
    cross-sectional percentile target (0=worst performer that day's cohort,
    1=best performer)."""
    raw = {}
    for ticker, df in features.items():
        d = df[df.index <= cutoff].copy()
        d["target_end_date"] = pd.to_datetime(d["target_end_date"])
        raw[ticker] = d

    if not raw:
        return pd.DataFrame()

    all_dates = sorted(set().union(*[set(d.index) for d in raw.values()]))
    idx = pd.DatetimeIndex(all_dates)

    wide_feats = {
        col: pd.DataFrame({t: d[col].reindex(idx) for t, d in raw.items()})
        for col in FEATURE_COLS
    }
    wide_target = pd.DataFrame({t: d[TARGET_COL].reindex(idx) for t, d in raw.items()})
    wide_target_end = pd.DataFrame({t: d["target_end_date"].reindex(idx) for t, d in raw.items()})

    z_feats = {col: _cross_sectional_zscore(wide_feats[col]) for col in FEATURE_COLS}
    target_pct = wide_target.rank(axis=1, pct=True)

    long_frames = []
    for col in FEATURE_COLS:
        s = z_feats[col].stack()
        s.index.names = ["Date", "Ticker"]
        long_frames.append(s.rename(col))
    feat_long = pd.concat(long_frames, axis=1)

    target_long = target_pct.stack()
    target_long.index.names = ["Date", "Ticker"]
    target_long = target_long.rename("target_percentile")

    raw_target_long = wide_target.stack()
    raw_target_long.index.names = ["Date", "Ticker"]
    raw_target_long = raw_target_long.rename("forward_return")

    end_long = wide_target_end.stack()
    end_long.index.names = ["Date", "Ticker"]
    end_long = end_long.rename("target_end_date")

    panel = feat_long.join(target_long).join(raw_target_long).join(end_long)
    panel = panel.dropna(subset=FEATURE_COLS)
    panel = panel.reset_index()
    return panel


def train_cross_sectional(features, cutoff, min_rows=2000, val_frac=0.2):
    """Trains the pooled cross-sectional model as of `cutoff`.

    Returns a dict with:
      scores: {ticker: predicted cross-sectional percentile today (0..1)}
      ic: validation-period average daily Spearman rank correlation
          between predicted score and actual forward return
          (the standard cross-sectional skill metric -- NOT MAE)
      n_train_rows: rows used in the final fit
    or None if there isn't enough pooled data yet to train responsibly.
    """
    panel = build_panel(features, cutoff)
    if panel.empty:
        return None

    labeled = panel.dropna(subset=["target_end_date", "forward_return", "target_percentile"])
    labeled = labeled[labeled["target_end_date"] <= cutoff]
    if len(labeled) < min_rows:
        return None

    dates = np.sort(labeled["Date"].unique())
    split_date = dates[int(len(dates) * (1 - val_frac))]
    train = labeled[labeled["Date"] <= split_date]
    val = labeled[labeled["Date"] > split_date]
    if len(val) < 200 or len(train) < min_rows * 0.5:
        return None

    def make_model():
        return HistGradientBoostingRegressor(
            max_depth=5, max_iter=300, learning_rate=0.05,
            l2_regularization=1.0, random_state=42,
        )

    model = make_model()
    model.fit(train[FEATURE_COLS], train["target_percentile"])
    val = val.copy()
    val["pred"] = model.predict(val[FEATURE_COLS])

    # Information Coefficient: computed WITHIN each date (proper
    # cross-sectional evaluation), then averaged across validation dates.
    ics = []
    for _, grp in val.groupby("Date"):
        if len(grp) >= 5 and grp["forward_return"].nunique() > 1:
            ic, _ = spearmanr(grp["pred"], grp["forward_return"])
            if not np.isnan(ic):
                ics.append(ic)
    mean_ic = float(np.mean(ics)) if ics else 0.0

    # Refit on all eligible labeled data (train+val) for the live prediction.
    model_final = make_model()
    model_final.fit(labeled[FEATURE_COLS], labeled["target_percentile"])

    latest = panel.sort_values("Date").groupby("Ticker").tail(1).dropna(subset=FEATURE_COLS)
    if latest.empty:
        return None
    latest_scores = model_final.predict(latest[FEATURE_COLS])
    scores = dict(zip(latest["Ticker"], latest_scores))

    return {"scores": scores, "ic": mean_ic, "n_train_rows": int(len(labeled))}


def scores_to_expected_returns(scores, ic, hist_annual_return, tilt_scale, min_ret, max_ret):
    """Converts cross-sectional percentile scores into small expected-return
    tilts around each stock's own historical-mean return.

    Rationale: cross-sectional ML gives a genuinely more reliable RANKING
    than an absolute return magnitude. Rather than pretend we know a
    precise expected return number, we start from each stock's own
    historical average (a stable baseline) and apply only a modest tilt,
    sized by how much validated skill (IC) the model actually showed.
    A validation IC of 0 means zero tilt -- ranking is only trusted to the
    extent it was shown to work out-of-sample.
    """
    confidence = max(0.0, ic)  # never tilt on negative/unproven skill
    out = {}
    for ticker, score in scores.items():
        centered = (score - 0.5) * 2.0  # -1 (worst-ranked) .. +1 (best-ranked)
        tilt = centered * confidence * tilt_scale
        base = hist_annual_return.get(ticker, 0.0)
        annual = float(np.clip(base + tilt, min_ret, max_ret))
        out[ticker] = annual
    return out
