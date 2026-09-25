"""
Step 2: Extract features from raw OHLCV market data.

Features built per row (per ticker, per day):
  - daily return (close-to-close)
  - N-day rolling returns (momentum)
  - rolling volatility (std of returns)
  - high/low range as % of close
  - volume vs its rolling average (volume spike indicator)
  - gap between open and previous close

Usage:
    python feature_engineering.py
"""

import os
import numpy as np
import pandas as pd

import config


def add_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()

    # --- N-day rolling returns (momentum features, includes 1-day return since
    #     RETURN_WINDOWS starts at 1) ---
    for w in config.RETURN_WINDOWS:
        df[f"return_{w}d"] = df["Close"].pct_change(periods=w)

    # --- rolling volatility of daily returns ---
    df["volatility"] = df["return_1d"].rolling(config.VOLATILITY_WINDOW).std()

    # --- high/low range as a % of close (intraday volatility proxy) ---
    df["hl_range_pct"] = (df["High"] - df["Low"]) / df["Close"]

    # --- volume spike: today's volume vs rolling average volume ---
    df["volume_avg"] = df["Volume"].rolling(config.VOLUME_AVG_WINDOW).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_avg"]

    # --- overnight gap: open vs previous close ---
    df["prev_close"] = df["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"]

    return df


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ticker, group in raw_df.groupby("Ticker"):
        out.append(add_features_for_ticker(group))
    features = pd.concat(out, ignore_index=True)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def main():
    raw_path = os.path.join(config.RAW_DATA_DIR, "market_data.csv")
    raw_df = pd.read_csv(raw_path, parse_dates=["Date"])

    features = build_features(raw_df)

    os.makedirs(config.FEATURES_DATA_DIR, exist_ok=True)
    out_path = os.path.join(config.FEATURES_DATA_DIR, "features.csv")
    features.to_csv(out_path, index=False)
    print(f"Saved features for {len(features)} rows -> {out_path}")
    print(features.tail())


if __name__ == "__main__":
    main()
