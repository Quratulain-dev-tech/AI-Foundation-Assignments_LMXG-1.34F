import json
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd


# ============================================================
# URL CLASSIFICATION & CLEANING
# ============================================================

print("=" * 60)
print("URL CLASSIFICATION & CLEANING")
print("=" * 60)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "urls.csv.json"

CLEAN_FILE = DATA_DIR / "clean_urls.csv"
INVALID_FILE = DATA_DIR / "invalid_urls.csv"
DUPLICATE_FILE = DATA_DIR / "duplicate_urls.csv"


# ============================================================
# LOAD URLs
# ============================================================

def load_urls(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Make sure we have a list
    if not isinstance(data, list):
        raise ValueError("JSON file does not contain a list of URLs.")

    return data


# ============================================================
# VALIDATE URL
# ============================================================

def is_valid_url(url):

    if not isinstance(url, str):
        return False

    url = url.strip()

    if not url:
        return False

    try:
        parsed = urlparse(url)

        # URL must have http/https
        if parsed.scheme not in ("http", "https"):
            return False

        # URL must have a domain
        if not parsed.netloc:
            return False

        return True

    except Exception:
        return False


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    urls = load_urls(INPUT_FILE)

    print(f"\nInput file:")
    print(INPUT_FILE)

    print(f"\nOriginal URLs: {len(urls)}")


    # --------------------------------------------------------
    # Remove empty values
    # --------------------------------------------------------

    non_empty_urls = []

    for url in urls:

        if isinstance(url, str) and url.strip():

            non_empty_urls.append(url.strip())


    print(f"After empty removal: {len(non_empty_urls)}")


    # --------------------------------------------------------
    # Find duplicates
    # --------------------------------------------------------

    seen = set()

    unique_urls = []

    duplicate_urls = []

    for url in non_empty_urls:

        if url in seen:

            duplicate_urls.append(url)

        else:

            seen.add(url)
            unique_urls.append(url)


    print(f"Unique URLs: {len(unique_urls)}")
    print(f"Duplicate URLs: {len(duplicate_urls)}")


    # --------------------------------------------------------
    # Validate URLs
    # --------------------------------------------------------

    valid_urls = []
    invalid_urls = []

    for url in unique_urls:

        if is_valid_url(url):

            valid_urls.append(url)

        else:

            invalid_urls.append(url)


    print(f"Valid URLs: {len(valid_urls)}")
    print(f"Invalid URLs: {len(invalid_urls)}")


    # ========================================================
    # SAVE CLEAN URLS
    # ========================================================

    clean_df = pd.DataFrame({
        "url": valid_urls
    })

    clean_df.to_csv(
        CLEAN_FILE,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # SAVE INVALID URLS
    # ========================================================

    invalid_df = pd.DataFrame({
        "url": invalid_urls
    })

    invalid_df.to_csv(
        INVALID_FILE,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # SAVE DUPLICATES
    # ========================================================

    duplicate_df = pd.DataFrame({
        "url": duplicate_urls
    })

    duplicate_df.to_csv(
        DUPLICATE_FILE,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETED")
    print("=" * 60)

    print(f"Original URLs       : {len(urls)}")
    print(f"After empty removal : {len(non_empty_urls)}")
    print(f"Unique URLs         : {len(unique_urls)}")
    print(f"Valid URLs          : {len(valid_urls)}")
    print(f"Invalid URLs        : {len(invalid_urls)}")
    print(f"Duplicate URLs      : {len(duplicate_urls)}")

    print("\nFiles created:")

    print(f"✓ {CLEAN_FILE}")
    print(f"✓ {INVALID_FILE}")
    print(f"✓ {DUPLICATE_FILE}")

    print("\nReady for URL reachability checking.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()