"""
Step 8: Real web scraping of news headlines via Google News RSS.

Unlike news_scrape.py (which uses yfinance's built-in news feed), this
module does actual web scraping: it fetches Google News's public RSS
"search" feed per company and parses the XML directly with no API key.

RSS URL pattern:
    https://news.google.com/rss/search?q=<company>&hl=en-US&gl=US&ceid=US:en

Usage:
    python web_scraper.py
"""

import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from email.utils import parsedate_to_datetime

import config
from sentiment import score_sentiment

# ticker -> best search query (company name) to use against Google News.
# Extend this alongside config.TICKERS / ner_extract.TICKER_ALIASES.
COMPANY_NAMES = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketPredictorBot/1.0)"}


def scrape_google_news(company_name: str, ticker: str, max_items: int = 20, max_retries: int = 3) -> pd.DataFrame:
    """Scrape recent headlines for `company_name` from Google News RSS.
    Retries on transient connection/timeout errors (common with this
    endpoint) before giving up."""
    query = requests.utils.quote(company_name)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    resp = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=25)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == max_retries:
                print(f"  WARNING: could not fetch Google News for {company_name} "
                      f"after {max_retries} attempts: {e}")
                return pd.DataFrame()
            wait = 3 * attempt  # 3s, 6s, ... simple backoff
            print(f"  Attempt {attempt} failed for {company_name} ({e.__class__.__name__}), "
                  f"retrying in {wait}s ...")
            time.sleep(wait)

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  WARNING: could not parse RSS for {company_name}: {e}")
        return pd.DataFrame()

    rows = []
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue

        pub_date_raw = item.findtext("pubDate")
        link = item.findtext("link") or ""

        try:
            date_str = parsedate_to_datetime(pub_date_raw).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            date_str = datetime.today().strftime("%Y-%m-%d")

        rows.append({
            "Date": date_str,
            "Ticker": ticker,
            "Title": title,
            "Sentiment": round(score_sentiment(title), 2),
            "Link": link,
            "Source": "google_news",
        })

    return pd.DataFrame(rows)


def scrape_all(tickers=None) -> pd.DataFrame:
    tickers = tickers or config.TICKERS
    frames = []
    for t in tickers:
        name = COMPANY_NAMES.get(t, t)
        print(f"Scraping Google News for {name} ({t}) ...")
        df = scrape_google_news(name, t)
        if not df.empty:
            frames.append(df)
        time.sleep(2)  # be polite -- avoid hammering the RSS endpoint
    if not frames:
        return pd.DataFrame(columns=["Date", "Ticker", "Title", "Sentiment", "Link", "Source"])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = scrape_all()
    print(f"\nScraped {len(df)} headlines total")
    print(df.head(10))
