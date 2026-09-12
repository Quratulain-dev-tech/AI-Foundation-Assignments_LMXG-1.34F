"""STEP 3: Cross-sectional ranking model (single formation cutoff).

CHANGED FROM PREVIOUS VERSION: instead of training one model per stock to
predict its own absolute return, this pools ALL stocks together and trains
one model to rank stocks against each other on a given day. See
cross_sectional_model.py for the full rationale (based on Gu, Kelly & Xiu
2020 and follow-up empirical asset-pricing ML research).
"""

import os
import numpy as np
import pandas as pd

from config import (
    TICKERS, FEATURES_FOLDER, OUTPUTS_FOLDER, TRAIN_END_DATE,
    MAX_ANNUAL_EXPECTED_RETURN, MIN_ANNUAL_EXPECTED_RETURN,
    CROSS_SECTIONAL_MIN_ROWS, CROSS_SECTIONAL_TILT_SCALE,
)
from cross_sectional_model import train_cross_sectional, scores_to_expected_returns


def main():
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    cutoff = pd.Timestamp(TRAIN_END_DATE)

    features = {}
    for ticker in TICKERS:
        path = os.path.join(FEATURES_FOLDER, f"{ticker}_features.csv")
        if os.path.exists(path):
            features[ticker] = pd.read_csv(path, index_col=0, parse_dates=True)

    if len(features) < 30:
        raise ValueError(
            f"Only {len(features)} tickers have feature files. A cross-sectional "
            "ranking model needs a broad universe (dozens to hundreds of stocks) "
            "to learn a meaningful ranking -- run step1 and step2 first on the "
            "full S&P 500 list in config.py."
        )

    print(f"Training pooled cross-sectional model on {len(features)} tickers "
          f"up to {TRAIN_END_DATE} ...")
    result = train_cross_sectional(features, cutoff, min_rows=CROSS_SECTIONAL_MIN_ROWS)
    if result is None:
        raise ValueError(
            "Not enough pooled formation data yet to train the cross-sectional "
            "model responsibly. Check TRAIN_END_DATE / START_DATE in config.py."
        )

    print(f"Pooled training rows: {result['n_train_rows']}")
    print(f"Validation Information Coefficient (IC): {result['ic']:.4f}  "
          f"(0 = no ranking skill found, ~0.02-0.05 is a realistic 'good' result "
          f"in this literature, 1.0 = perfect ranking)")

    # Historical mean return per ticker (formation period only) -- the
    # stable baseline that the cross-sectional signal tilts around.
    hist_mean = {}
    for ticker, df in features.items():
        d = df[df.index <= cutoff]
        if len(d) > 0:
            hist_mean[ticker] = float(d["daily_return"].mean() * 252)

    expected_returns = scores_to_expected_returns(
        result["scores"], result["ic"], hist_mean,
        tilt_scale=CROSS_SECTIONAL_TILT_SCALE,
        min_ret=MIN_ANNUAL_EXPECTED_RETURN, max_ret=MAX_ANNUAL_EXPECTED_RETURN,
    )

    rows = []
    for ticker, annual in sorted(expected_returns.items(), key=lambda kv: -kv[1]):
        score = result["scores"].get(ticker, np.nan)
        rows.append({
            "Ticker": ticker,
            "Expected_Annual_Return": annual,
            "Cross_Sectional_Score": score,
            "Validation_IC": result["ic"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUTPUTS_FOLDER, "predicted_returns.csv"), index=False)

    print(f"\nTop 10 ranked (highest expected relative performance):")
    print(out_df.head(10).to_string(index=False, formatters={
        "Expected_Annual_Return": "{:.2%}".format, "Cross_Sectional_Score": "{:.3f}".format,
    }))
    print(f"\nBottom 10 ranked (lowest expected relative performance):")
    print(out_df.tail(10).to_string(index=False, formatters={
        "Expected_Annual_Return": "{:.2%}".format, "Cross_Sectional_Score": "{:.3f}".format,
    }))
    print(f"\nSaved predictions for {len(out_df)} tickers. Formation cutoff: {TRAIN_END_DATE}.")


if __name__ == "__main__":
    main()
