"""
Step 1: Fetch past N years of market data (OHLCV) with date/time index.

Why this version exists
-----------------------
Yahoo Finance sometimes answers "possibly delisted; no price data found" for a
perfectly healthy ticker (AAPL, NVDA ...) when the request hits a temporary
connection / cookie / rate-limit problem. It is NOT a real delisting. The old
script just skipped that ticker and silently saved 5 out of 6 companies, so
the whole pipeline trained without it.

This version:
  1. retries every ticker several times, alternating between yf.download()
     and yf.Ticker().history(), waiting a bit longer after each failure;
  2. if a ticker still fails, re-uses its rows from the previous
     market_data.csv (when there is one) and tells you so;
  3. prints a loud warning and exits with code 1 if any ticker is still
     missing, so you notice and simply run the script again.

Usage:
    python data_fetch.py
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

import config

MAX_RETRIES = 4            # attempts per ticker
BASE_WAIT_SECONDS = 4      # waits 4s, 8s, 12s ... between attempts
PAUSE_BETWEEN_TICKERS = 1.5

COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
OUT_PATH = os.path.join(config.RAW_DATA_DIR, "market_data.csv")


def _download_once(ticker: str, start, end, use_history_api: bool) -> pd.DataFrame:
    """One raw attempt. Two different Yahoo code paths, so if one is blocked
    or rate-limited the other often still works."""
    if use_history_api:
        return yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    return yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)


def _clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise whatever yfinance returned into the exact columns we need."""
    # Newer yfinance versions can return MultiIndex columns like
    # ('Close', 'AAPL') even for a single ticker. Flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})  # index is Date/Datetime
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:                 # Ticker.history() is tz-aware
        df["Date"] = df["Date"].dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize()
    df["Ticker"] = ticker
    return df[COLUMNS]


def fetch_ticker_data(ticker: str, start, end, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    """Fetch daily OHLCV for one ticker, retrying on empty/failed responses."""
    print(f"Fetching {ticker} from {start.date()} to {end.date()} ...")

    for attempt in range(1, max_retries + 1):
        use_history_api = attempt % 2 == 0  # 1: download, 2: history, 3: download ...
        try:
            raw = _download_once(ticker, start, end, use_history_api)
            if raw is not None and not raw.empty:
                clean = _clean(raw, ticker)
                if attempt > 1:
                    print(f"  OK for {ticker} on attempt {attempt}")
                return clean
        except Exception as e:
            print(f"  attempt {attempt}/{max_retries} raised {e.__class__.__name__}: {e}")

        if attempt < max_retries:
            wait = BASE_WAIT_SECONDS * attempt
            print(f"  attempt {attempt}/{max_retries}: no data for {ticker} "
                  f"(usually a temporary Yahoo/network problem, not a real delisting) "
                  f"- retrying in {wait}s ...")
            time.sleep(wait)

    print(f"  WARNING: no data for {ticker} after {max_retries} attempts")
    return pd.DataFrame(columns=COLUMNS)


def fetch_all(tickers=None, start=None, end=None):
    """Return (list_of_ticker_frames, list_of_tickers_that_failed)."""
    tickers = tickers or config.TICKERS
    start = start or config.START_DATE
    end = end or config.END_DATE

    frames, missing = [], []
    for i, t in enumerate(tickers):
        df = fetch_ticker_data(t, start, end)
        if df.empty:
            missing.append(t)
        else:
            frames.append(df)
        if i < len(tickers) - 1:
            time.sleep(PAUSE_BETWEEN_TICKERS)  # be polite to Yahoo
    return frames, missing


def _previous_rows(tickers) -> pd.DataFrame:
    """Rows for `tickers` from the last saved market_data.csv (if any)."""
    if not os.path.exists(OUT_PATH):
        return pd.DataFrame(columns=COLUMNS)
    old = pd.read_csv(OUT_PATH, parse_dates=["Date"])
    return old[old["Ticker"].isin(tickers)][COLUMNS]


def main():
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    frames, missing = fetch_all()

    # Fall back to previously saved rows for tickers that failed this time.
    still_missing = []
    if missing:
        old = _previous_rows(missing)
        for t in missing:
            rows = old[old["Ticker"] == t]
            if rows.empty:
                still_missing.append(t)
            else:
                print(f"  NOTE: fresh download failed for {t}; keeping OLD saved rows "
                      f"(last date {rows['Date'].max().date()}). Re-run later to refresh.")
                frames.append(rows)

    if not frames:
        print("\nERROR: no data could be fetched for any ticker. "
              "Check your internet connection and try again.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(combined)} rows for {combined['Ticker'].nunique()} tickers -> {OUT_PATH}")
    print(combined.groupby("Ticker")["Date"].agg(["count", "min", "max"]))

    if still_missing:
        print("\n" + "!" * 70)
        print(f"  INCOMPLETE DATA: {', '.join(still_missing)} could not be fetched.")
        print("  Do NOT continue the pipeline yet - just run   python data_fetch.py   again.")
        print("!" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
