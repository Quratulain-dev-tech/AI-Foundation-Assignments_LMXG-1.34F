import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from datetime import datetime
import time


# ============================================================
# PARALLEL BBC + CNN SCRAPER
# ============================================================

print("=" * 70)
print("PARALLEL BBC + CNN WEB SCRAPER")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

REACHABLE_FILE = DATA_DIR / "reachable_urls.csv"
RETRY_REACHABLE_FILE = DATA_DIR / "retry_reachable_urls.csv"

OUTPUT_FILE = DATA_DIR / "scraped_articles.csv"
FAILED_FILE = DATA_DIR / "scraping_failed.csv"


# ============================================================
# SETTINGS
# ============================================================

MAX_WORKERS = 10
TIMEOUT = 15


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9"
}


# ============================================================
# LOAD URLS
# ============================================================

def load_urls():

    all_urls = []

    # First reachable URLs
    if REACHABLE_FILE.exists():

        df = pd.read_csv(REACHABLE_FILE)

        if "url" in df.columns:

            all_urls.extend(
                df["url"]
                .dropna()
                .astype(str)
                .tolist()
            )


    # URLs recovered after 429 retry
    if RETRY_REACHABLE_FILE.exists():

        df = pd.read_csv(RETRY_REACHABLE_FILE)

        if "url" in df.columns:

            all_urls.extend(
                df["url"]
                .dropna()
                .astype(str)
                .tolist()
            )


    # Remove duplicates
    unique_urls = list(dict.fromkeys(all_urls))

    return unique_urls


# ============================================================
# DETECT SOURCE
# ============================================================

def get_source(url):

    domain = urlparse(url).netloc.lower()

    if "bbc." in domain:
        return "BBC"

    if "cnn." in domain:
        return "CNN"

    return "Other"


# ============================================================
# SCRAPE ONE URL
# ============================================================

def scrape_url(url):

    start_time = time.time()

    source = get_source(url)

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True
        )

        response_time = round(
            time.time() - start_time,
            3
        )

        status_code = response.status_code

        # ----------------------------------------------------
        # HTTP error
        # ----------------------------------------------------

        if status_code >= 400:

            return {
                "url": url,
                "source": source,
                "title": "",
                "description": "",
                "content": "",
                "published_date": "",
                "author": "",
                "status_code": status_code,
                "response_time": response_time,
                "scraped_at": datetime.now().isoformat(),
                "status": "failed",
                "error": f"HTTP {status_code}"
            }


        # ----------------------------------------------------
        # Parse HTML
        # ----------------------------------------------------

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )


        # ----------------------------------------------------
        # Remove unnecessary elements
        # ----------------------------------------------------

        for element in soup(
            ["script", "style", "noscript"]
        ):

            element.decompose()


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title = ""

        if soup.title:

            title = soup.title.get_text(
                strip=True
            )


        # ----------------------------------------------------
        # DESCRIPTION
        # ----------------------------------------------------

        description = ""

        meta_description = soup.find(
            "meta",
            attrs={"name": "description"}
        )

        if meta_description:

            description = meta_description.get(
                "content",
                ""
            ).strip()


        # ----------------------------------------------------
        # PUBLISHED DATE
        # ----------------------------------------------------

        published_date = ""

        date_meta = (
            soup.find(
                "meta",
                attrs={
                    "property": "article:published_time"
                }
            )
            or
            soup.find(
                "meta",
                attrs={
                    "property": "og:published_time"
                }
            )
            or
            soup.find(
                "meta",
                attrs={
                    "name": "date"
                }
            )
        )

        if date_meta:

            published_date = (
                date_meta.get("content", "")
            )


        # ----------------------------------------------------
        # AUTHOR
        # ----------------------------------------------------

        author = ""

        author_meta = soup.find(
            "meta",
            attrs={"name": "author"}
        )

        if author_meta:

            author = author_meta.get(
                "content",
                ""
            ).strip()


        # ----------------------------------------------------
        # CONTENT
        # ----------------------------------------------------

        paragraphs = soup.find_all("p")

        content_parts = []

        for paragraph in paragraphs:

            text = paragraph.get_text(
                " ",
                strip=True
            )

            if text:

                content_parts.append(text)


        content = " ".join(content_parts)


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        return {
            "url": url,
            "source": source,
            "title": title,
            "description": description,
            "content": content,
            "published_date": published_date,
            "author": author,
            "status_code": status_code,
            "response_time": response_time,
            "scraped_at": datetime.now().isoformat(),
            "status": "success",
            "error": ""
        }


    except requests.exceptions.Timeout:

        return {
            "url": url,
            "source": source,
            "title": "",
            "description": "",
            "content": "",
            "published_date": "",
            "author": "",
            "status_code": "",
            "response_time": round(
                time.time() - start_time,
                3
            ),
            "scraped_at": datetime.now().isoformat(),
            "status": "failed",
            "error": "Timeout"
        }


    except requests.exceptions.RequestException as e:

        return {
            "url": url,
            "source": source,
            "title": "",
            "description": "",
            "content": "",
            "published_date": "",
            "author": "",
            "status_code": "",
            "response_time": round(
                time.time() - start_time,
                3
            ),
            "scraped_at": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        }


    except Exception as e:

        return {
            "url": url,
            "source": source,
            "title": "",
            "description": "",
            "content": "",
            "published_date": "",
            "author": "",
            "status_code": "",
            "response_time": round(
                time.time() - start_time,
                3
            ),
            "scraped_at": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load URLs
    # --------------------------------------------------------

    urls = load_urls()

    print(f"\nTotal URLs for scraping: {len(urls)}")
    print(f"Parallel threads: {MAX_WORKERS}")
    print(f"Timeout: {TIMEOUT} seconds")


    if len(urls) == 0:

        print("\n❌ No URLs found.")

        return


    # --------------------------------------------------------
    # Parallel scraping
    # --------------------------------------------------------

    results = []

    completed = 0

    print("\nStarting parallel scraping...\n")


    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_url = {
            executor.submit(
                scrape_url,
                url
            ): url

            for url in urls
        }


        for future in as_completed(
            future_to_url
        ):

            try:

                result = future.result()

                results.append(result)

            except Exception as e:

                url = future_to_url[future]

                results.append({
                    "url": url,
                    "source": get_source(url),
                    "title": "",
                    "description": "",
                    "content": "",
                    "published_date": "",
                    "author": "",
                    "status_code": "",
                    "response_time": "",
                    "scraped_at": datetime.now().isoformat(),
                    "status": "failed",
                    "error": str(e)
                })


            completed += 1


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                completed % 100 == 0
                or completed == len(urls)
            ):

                print(
                    f"Progress: "
                    f"{completed}/{len(urls)}"
                )


    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)


    # --------------------------------------------------------
    # Separate success / failed
    # --------------------------------------------------------

    successful_df = results_df[
        results_df["status"] == "success"
    ].copy()


    failed_df = results_df[
        results_df["status"] == "failed"
    ].copy()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    successful_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


    failed_df.to_csv(
        FAILED_FILE,
        index=False,
        encoding="utf-8"
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("PARALLEL SCRAPING COMPLETED")
    print("=" * 70)

    print(f"Total URLs       : {len(urls)}")
    print(f"Successful       : {len(successful_df)}")
    print(f"Failed           : {len(failed_df)}")

    print("\nSource breakdown:")

    if len(successful_df) > 0:

        print(
            successful_df["source"]
            .value_counts()
            .to_string()
        )


    print("\nFiles created:")

    print(f"✓ {OUTPUT_FILE}")
    print(f"✓ {FAILED_FILE}")

    print("\nScraping completed! 🎉")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()