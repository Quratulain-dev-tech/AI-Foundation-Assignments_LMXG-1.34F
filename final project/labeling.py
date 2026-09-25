"""
Step 3: Assign the target label (spike / crash / neutral).

Technique used here: threshold-based labeling on the NEXT day's return.
  - next_return > SPIKE_THRESHOLD   -> "spike"
  - next_return < CRASH_THRESHOLD   -> "crash"
  - otherwise                        -> "neutral"

This is the simplest, most explainable labeling technique and is a good
starting point for a classifier / assignment. If you want a more advanced
technique later, you could swap this for:
  - volatility-adjusted thresholds (z-score of return vs rolling std)
  - k-means clustering of returns into 3 clusters
  - a zero-shot LLM labeling pass on qualitative price-action descriptions

Usage:
    python labeling.py
"""

import os
import pandas as pd

import config


def label_next_day_move(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()

    # Next day's return, computed per ticker so labels don't leak across tickers
    df["next_return"] = df.groupby("Ticker")["Close"].pct_change().shift(-1)

    def to_label(r):
        if pd.isna(r):
            return None
        if r > config.SPIKE_THRESHOLD:
            return "spike"
        if r < config.CRASH_THRESHOLD:
            return "crash"
        return "neutral"

    df["target"] = df["next_return"].apply(to_label)

    # Drop the last row per ticker (no next-day label available)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    return df


def main():
    in_path = os.path.join(config.FEATURES_DATA_DIR, "features.csv")
    df = pd.read_csv(in_path, parse_dates=["Date"])

    labeled = label_next_day_move(df)

    out_path = os.path.join(config.FEATURES_DATA_DIR, "labeled_features.csv")
    labeled.to_csv(out_path, index=False)

    print(f"Saved {len(labeled)} labeled rows -> {out_path}")
    print("\nLabel distribution:")
    print(labeled["target"].value_counts())


if __name__ == "__main__":
    main()
