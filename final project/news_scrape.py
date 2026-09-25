"""
Step 7: Collect news headlines per ticker (with date + sentiment score) so
you can start building a REAL historical news dataset over time.

Combines TWO sources into one file (data/news/news_history.csv):
  1. yfinance's built-in news feed (Ticker.news)
  2. Real web scraping of Google News RSS (web_scraper.py)

IMPORTANT LIMITATION -- read this first:
  Both sources only expose RECENT headlines (no free way to pull 2 years of
  past headlines retroactively). This script therefore APPENDS today's
  headlines to a running CSV each time you run it (duplicates skipped). If
  you run it daily (e.g. via Windows Task Scheduler / cron), you'll
  gradually build up a real dated news history that build_dataset.py can
  merge with data/features/labeled_features.csv to train on real headline
  sentiment instead of only blending it in at prediction time.

Usage:
    python news_scrape.py
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

import config
from sentiment import score_sentiment
from web_scraper import scrape_all as scrape_google_news_all

NEWS_DIR = "data/news"
NEWS_PATH = os.path.join(NEWS_DIR, "news_history.csv")


def fetch_news_for_ticker(ticker: str) -> pd.DataFrame:
    """Pull whatever recent news Yahoo Finance currently has for `ticker`."""
    t = yf.Ticker(ticker)
    try:
        items = t.news or []
    except Exception as e:
        print(f"  WARNING: could not fetch news for {ticker}: {e}")
        return pd.DataFrame()

    rows = []
    for item in items:
        # yfinance nests the actual article fields under "content" in newer
        # versions; fall back to the flat structure used in older versions.
        content = item.get("content", item)
        title = content.get("title") or item.get("title", "")
        if not title:
            continue

        pub_time = (
            content.get("pubDate")
            or content.get("providerPublishTime")
            or item.get("providerPublishTime")
        )
        date_str = _parse_date(pub_time)

        rows.append({
            "Date": date_str,
            "Ticker": ticker,
            "Title": title,
            "Sentiment": round(score_sentiment(title), 2),
            "Link": (content.get("canonicalUrl") or {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else content.get("link", ""),
            "Source": "yfinance",
        })

    return pd.DataFrame(rows)


def _parse_date(pub_time) -> str:
    """Handle both unix-timestamp and ISO-string publish times, else today."""
    if pub_time is None:
        return datetime.today().strftime("%Y-%m-%d")
    try:
        # unix timestamp (older yfinance versions)
        return datetime.fromtimestamp(int(pub_time)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass
    try:
        # ISO string like "2026-09-24T12:30:00Z" (newer yfinance versions)
        return str(pub_time)[:10]
    except Exception:
        return datetime.today().strftime("%Y-%m-%d")


def scrape_all(tickers=None) -> pd.DataFrame:
    tickers = tickers or config.TICKERS
    frames = []
    for t in tickers:
        print(f"Fetching yfinance news for {t} ...")
        df = fetch_news_for_ticker(t)
        if not df.empty:
            frames.append(df)
    if not frames:
        yfinance_news = pd.DataFrame(columns=["Date", "Ticker", "Title", "Sentiment", "Link", "Source"])
    else:
        yfinance_news = pd.concat(frames, ignore_index=True)

    # Real web scraping (Google News RSS) -- a genuinely different source
    google_news = scrape_google_news_all(tickers)

    return pd.concat([yfinance_news, google_news], ignore_index=True)


def main():
    os.makedirs(NEWS_DIR, exist_ok=True)
    new_news = scrape_all()

    if os.path.exists(NEWS_PATH):
        existing = pd.read_csv(NEWS_PATH)
        combined = pd.concat([existing, new_news], ignore_index=True)
        # de-duplicate: same ticker + same title = same article, regardless of source
        combined = combined.drop_duplicates(subset=["Ticker", "Title"], keep="first")
    else:
        combined = new_news

    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    combined.to_csv(NEWS_PATH, index=False)

    print(f"\nSaved {len(combined)} total headlines ({len(new_news)} new this run) -> {NEWS_PATH}")
    print("By source:")
    print(combined["Source"].value_counts())
    print(combined.tail(10))


if __name__ == "__main__":
    main()
