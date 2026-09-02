import pandas as pd
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# RETRY HTTP 429 URLs
# ============================================================

print("=" * 60)
print("HTTP 429 RETRY CHECK")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

UNREACHABLE_FILE = DATA_DIR / "unreachable_urls.csv"

RETRY_REACHABLE_FILE = DATA_DIR / "retry_reachable_urls.csv"
RETRY_STILL_UNREACHABLE_FILE = DATA_DIR / "retry_still_unreachable.csv"


# ============================================================
# SETTINGS
# ============================================================

# Lower than previous 20 threads
MAX_WORKERS = 5

# Longer timeout
TIMEOUT = 15

# Delay before retry
RETRY_DELAY = 2


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
}


# ============================================================
# CHECK ONE URL
# ============================================================

def retry_url(row):

    url = row["url"]

    # Small delay before request
    time.sleep(RETRY_DELAY)

    start_time = time.time()

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True
        )

        response_time = round(time.time() - start_time, 3)

        status_code = response.status_code

        if 200 <= status_code < 400:

            return {
                "url": url,
                "status_code": status_code,
                "reachable": True,
                "response_time": response_time,
                "error": ""
            }

        else:

            return {
                "url": url,
                "status_code": status_code,
                "reachable": False,
                "response_time": response_time,
                "error": f"HTTP {status_code}"
            }

    except requests.exceptions.Timeout:

        response_time = round(time.time() - start_time, 3)

        return {
            "url": url,
            "status_code": "",
            "reachable": False,
            "response_time": response_time,
            "error": "Timeout"
        }

    except requests.exceptions.RequestException as e:

        response_time = round(time.time() - start_time, 3)

        return {
            "url": url,
            "status_code": "",
            "reachable": False,
            "response_time": response_time,
            "error": str(e)
        }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not UNREACHABLE_FILE.exists():

        print("\n❌ File not found:")
        print(UNREACHABLE_FILE)

        return


    # --------------------------------------------------------
    # Load unreachable URLs
    # --------------------------------------------------------

    df = pd.read_csv(UNREACHABLE_FILE)

    # Only retry HTTP 429
    retry_df = df[
        df["error"].astype(str).str.strip() == "HTTP 429"
    ].copy()

    print(f"\nHTTP 429 URLs found: {len(retry_df)}")

    if len(retry_df) == 0:

        print("\nNo HTTP 429 URLs to retry.")
        return


    print(f"Parallel threads: {MAX_WORKERS}")
    print(f"Delay per request: {RETRY_DELAY} seconds")
    print(f"Timeout: {TIMEOUT} seconds")

    print("\nStarting retry...\n")


    # --------------------------------------------------------
    # Parallel retry
    # --------------------------------------------------------

    results = []

    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(retry_url, row)
            for _, row in retry_df.iterrows()
        ]

        for future in as_completed(futures):

            result = future.result()

            results.append(result)

            completed += 1

            if completed % 100 == 0 or completed == len(futures):

                print(
                    f"Progress: {completed}/{len(futures)}"
                )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)


    retry_reachable = results_df[
        results_df["reachable"] == True
    ].copy()

    retry_still_unreachable = results_df[
        results_df["reachable"] == False
    ].copy()


    # --------------------------------------------------------
    # Save successful retries
    # --------------------------------------------------------

    retry_reachable.to_csv(
        RETRY_REACHABLE_FILE,
        index=False,
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Save still failed
    # --------------------------------------------------------

    retry_still_unreachable.to_csv(
        RETRY_STILL_UNREACHABLE_FILE,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("429 RETRY COMPLETED")
    print("=" * 60)

    print(f"429 URLs retried       : {len(retry_df)}")
    print(f"Now reachable          : {len(retry_reachable)}")
    print(f"Still unreachable      : {len(retry_still_unreachable)}")

    print("\nFiles created:")

    print(f"✓ {RETRY_REACHABLE_FILE}")
    print(f"✓ {RETRY_STILL_UNREACHABLE_FILE}")

    print("\nRetry step completed.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()