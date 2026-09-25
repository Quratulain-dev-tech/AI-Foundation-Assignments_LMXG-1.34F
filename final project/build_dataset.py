"""
Step 9: Combine the scraped news history with the price-based features into
ONE dataset the model can train on -- this is the "everything in one place"
step: yfinance price data + yfinance news + scraped Google News, merged by
(Ticker, Date).

For each (Ticker, Date) in labeled_features.csv, this adds:
  - news_sentiment_mean : average sentiment of headlines published that day
  - news_count          : how many headlines were published that day

Days with no scraped news for that ticker get news_sentiment_mean=0 (neutral)
and news_count=0 -- this is expected and fine; your news_history.csv will
only have real coverage for recent days until you've run news_scrape.py for
a while (see README).

Usage:
    python build_dataset.py
"""

import os
import pandas as pd

import config

NEWS_PATH = os.path.join("data/news", "news_history.csv")
PRICE_FEATURES_PATH = os.path.join(config.FEATURES_DATA_DIR, "labeled_features.csv")
OUTPUT_PATH = os.path.join(config.FEATURES_DATA_DIR, "labeled_features_with_news.csv")


def build_daily_news_agg(news_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse many headlines per (Ticker, Date) into one row of aggregates."""
    agg = (
        news_df.groupby(["Ticker", "Date"])
        .agg(news_sentiment_mean=("Sentiment", "mean"), news_count=("Sentiment", "count"))
        .reset_index()
    )
    return agg


def main():
    price_df = pd.read_csv(PRICE_FEATURES_PATH, parse_dates=["Date"])

    if not os.path.exists(NEWS_PATH):
        print(f"No news history found at {NEWS_PATH} yet -- run news_scrape.py first.")
        print("Filling news_sentiment_mean=0 and news_count=0 for all rows so training can still run.")
        price_df["news_sentiment_mean"] = 0.0
        price_df["news_count"] = 0
    else:
        news_df = pd.read_csv(NEWS_PATH, parse_dates=["Date"])
        daily_news = build_daily_news_agg(news_df)

        price_df = price_df.merge(daily_news, on=["Ticker", "Date"], how="left")
        price_df["news_sentiment_mean"] = price_df["news_sentiment_mean"].fillna(0.0)
        price_df["news_count"] = price_df["news_count"].fillna(0).astype(int)

    os.makedirs(config.FEATURES_DATA_DIR, exist_ok=True)
    price_df.to_csv(OUTPUT_PATH, index=False)

    covered_days = (price_df["news_count"] > 0).sum()
    print(f"Saved {len(price_df)} rows -> {OUTPUT_PATH}")
    print(f"{covered_days} of {len(price_df)} rows have real scraped news coverage "
          f"({covered_days / len(price_df):.1%}).")
    print(price_df[["Date", "Ticker", "news_sentiment_mean", "news_count", "target"]].tail(10))


if __name__ == "__main__":
    main()
