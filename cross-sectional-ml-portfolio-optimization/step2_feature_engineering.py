"""STEP 2: Feature engineering for the stock-return model."""

import os
import numpy as np
import pandas as pd

from config import TICKERS, DATA_FOLDER, FEATURES_FOLDER, HORIZON_DAYS


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df):
    df = df.copy().sort_index()

    close = df["Close"]
    volume = df["Volume"]
    ret = close.pct_change()

    df["daily_return"] = ret
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)
    df["volatility_20d"] = ret.rolling(20).std()
    df["volatility_60d"] = ret.rolling(60).std()
    df["ma_10_gap"] = close / close.rolling(10).mean() - 1
    df["ma_50_gap"] = close / close.rolling(50).mean() - 1
    df["ma_200_gap"] = close / close.rolling(200).mean() - 1
    df["rsi_14"] = rsi(close, 14) / 100.0
    df["volume_change_20d"] = volume.pct_change(20)
    df["volume_z_20d"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

    # Future 20-trading-day return. It is the prediction target, never an input.
    df["forward_return"] = close.shift(-HORIZON_DAYS) / close - 1
    df["target_end_date"] = pd.Series(df.index, index=df.index).shift(-HORIZON_DAYS)

    # Remove warm-up rows and rows where the future target does not exist.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def main():
    os.makedirs(FEATURES_FOLDER, exist_ok=True)
    for ticker in TICKERS:
        path = os.path.join(DATA_FOLDER, f"{ticker}.csv")
        if not os.path.exists(path):
            print(f"Skipping {ticker}: missing {path}")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        feat = build_features(df)
        out = os.path.join(FEATURES_FOLDER, f"{ticker}_features.csv")
        feat.to_csv(out)
        print(f"Saved: {out} ({len(feat)} rows)")


if __name__ == "__main__":
    main()
