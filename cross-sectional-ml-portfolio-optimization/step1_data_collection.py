"""
STEP 1: DATA COLLECTION
------------------------
Ye script company list (config.py mein, ab poori S&P 500 ~500 companies)
ka historical daily stock data yfinance (real market data, Yahoo Finance)
se download karta hai aur CSV files mein save karta hai.

NOTE ON RUNTIME: 500 tickers download hone mein kaafi time lag sakta hai
(pacing ki wajah se, taake Yahoo Finance rate-limit na kare). Ye script
RESUMABLE hai -- agar beech mein ruk jaye ya crash ho jaye, dobara chalane
par jo tickers pehle se saved hain unhe skip kar dega, sirf baaki wale
download honge.

Run: python step1_data_collection.py
"""

import yfinance as yf
import pandas as pd
import os
import time

from config import TICKERS, START_DATE, END_DATE, DATA_FOLDER

MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 1.5  # seconds -- avoids Yahoo Finance rate-limiting


def download_and_clean(ticker, start, end):
    for attempt in range(1, MAX_RETRIES + 1):
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.ffill().dropna()
            return df

        time.sleep(DELAY_BETWEEN_REQUESTS * attempt * 2)

    return None


def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    summary = []
    total = len(TICKERS)
    skipped_existing = 0

    for i, ticker in enumerate(TICKERS, 1):
        path = os.path.join(DATA_FOLDER, f"{ticker}.csv")

        # Resumable: skip tickers already downloaded from a previous run.
        if os.path.exists(path):
            try:
                existing = pd.read_csv(path, index_col=0, parse_dates=True)
                if len(existing) > 100:
                    skipped_existing += 1
                    summary.append({"Ticker": ticker, "Rows": len(existing)})
                    continue
            except Exception:
                pass  # fall through and re-download if the file is unreadable

        print(f"[{i}/{total}] Downloading: {ticker} ...")
        df = download_and_clean(ticker, START_DATE, END_DATE)
        if df is None:
            print(f"  Warning: no data for {ticker} after {MAX_RETRIES} tries, skipping.")
            continue

        df.to_csv(path)
        print(f"  Saved: {path} ({df.shape[0]} rows)")
        summary.append({"Ticker": ticker, "Rows": df.shape[0]})

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print("\n--- Summary ---")
    print(f"Already had: {skipped_existing} | Newly downloaded: {len(summary) - skipped_existing}")
    print(f"Done. {len(summary)}/{total} companies available in ./{DATA_FOLDER}/")

    missing = [t for t in TICKERS if t not in [s["Ticker"] for s in summary]]
    if missing:
        print(f"\n{len(missing)} tickers failed and were skipped: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print("Just re-run this script -- it will only retry the missing ones.")


if __name__ == "__main__":
    main()
