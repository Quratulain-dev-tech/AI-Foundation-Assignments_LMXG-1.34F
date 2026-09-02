import pandas as pd
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# URL REACHABILITY CHECKER
# ============================================================

print("=" * 60)
print("URL REACHABILITY CHECK")
print("=" * 60)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "clean_urls.csv"

REACHABLE_FILE = DATA_DIR / "reachable_urls.csv"
UNREACHABLE_FILE = DATA_DIR / "unreachable_urls.csv"

# Number of parallel threads
MAX_WORKERS = 20

# Request timeout
TIMEOUT = 10


# ============================================================
# USER AGENT
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}


# ============================================================
# CHECK ONE URL
# ============================================================

def check_url(url):

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
    # Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print("\n❌ Input file not found:")
        print(INPUT_FILE)

        return


    # --------------------------------------------------------
    # Load URLs
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    urls = df["url"].dropna().astype(str).tolist()

    print(f"\nTotal URLs: {len(urls)}")

    print(f"Parallel threads: {MAX_WORKERS}")

    print(f"Timeout per URL: {TIMEOUT} seconds")


    # --------------------------------------------------------
    # Parallel checking
    # --------------------------------------------------------

    results = []

    completed = 0

    print("\nStarting reachability checking...\n")


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_url = {
            executor.submit(check_url, url): url
            for url in urls
        }

        for future in as_completed(future_to_url):

            result = future.result()

            results.append(result)

            completed += 1

            # Progress every 100 URLs
            if completed % 100 == 0 or completed == len(urls):

                print(
                    f"Progress: {completed}/{len(urls)}"
                )


    # --------------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)


    # --------------------------------------------------------
    # Separate reachable / unreachable
    # --------------------------------------------------------

    reachable_df = results_df[
        results_df["reachable"] == True
    ].copy()

    unreachable_df = results_df[
        results_df["reachable"] == False
    ].copy()


    # --------------------------------------------------------
    # Save reachable URLs
    # --------------------------------------------------------

    reachable_df.to_csv(
        REACHABLE_FILE,
        index=False,
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Save unreachable URLs
    # --------------------------------------------------------

    unreachable_df.to_csv(
        UNREACHABLE_FILE,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("REACHABILITY CHECK COMPLETED")
    print("=" * 60)

    print(f"Total URLs       : {len(urls)}")
    print(f"Reachable URLs   : {len(reachable_df)}")
    print(f"Unreachable URLs : {len(unreachable_df)}")

    print("\nFiles created:")

    print(f"✓ {REACHABLE_FILE}")
    print(f"✓ {UNREACHABLE_FILE}")

    print("\nReady for parallel scraping.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()