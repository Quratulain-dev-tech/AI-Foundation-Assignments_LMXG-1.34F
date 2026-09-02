import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timezone
import pandas as pd
import re
import time
from pathlib import Path


# ==========================================
# SETTINGS
# ==========================================

CNN_HOME = "https://www.cnn.com"

TARGET_ARTICLES = 100

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    )
}


# ==========================================
# 1. GET ARTICLE LINKS
# ==========================================

def get_article_links():

    print("\nStarting CNN link discovery...\n")

    sections = [
        CNN_HOME,
        f"{CNN_HOME}/world",
        f"{CNN_HOME}/us",
        f"{CNN_HOME}/politics",
        f"{CNN_HOME}/business",
        f"{CNN_HOME}/sport",
        f"{CNN_HOME}/science",
        f"{CNN_HOME}/health",
        f"{CNN_HOME}/entertainment",
        f"{CNN_HOME}/travel",
        f"{CNN_HOME}/weather"
    ]

    links = set()

    pattern = (
        r"^https://www\.cnn\.com/"
        r"\d{4}/\d{2}/\d{2}/"
        r"[^/]+/[^/?#]+"
    )

    for page in sections:

        try:

            response = requests.get(
                page,
                headers=HEADERS,
                timeout=15
            )

            print(
                f"{page} -> Status: "
                f"{response.status_code}"
            )

            if response.status_code != 200:
                continue

            soup = BeautifulSoup(
                response.text,
                "lxml"
            )

            for a in soup.find_all(
                "a",
                href=True
            ):

                href = a["href"]

                full_url = urljoin(
                    CNN_HOME,
                    href
                )

                # Only CNN URLs
                if not full_url.startswith(
                    "https://www.cnn.com/"
                ):
                    continue

                # Exclude videos
                if "/video/" in full_url:
                    continue

                # Exclude live news
                if "/live-news/" in full_url:
                    continue

                # Only article URLs
                if re.match(
                    pattern,
                    full_url
                ):
                    links.add(
                        full_url
                    )

        except Exception as e:

            print(
                f"Could not access {page}: "
                f"{type(e).__name__}"
            )

    return sorted(links)


# ==========================================
# 2. GET META CONTENT
# ==========================================

def get_meta(soup, name):

    tag = soup.find(
        "meta",
        attrs={"name": name}
    )

    if not tag:

        tag = soup.find(
            "meta",
            attrs={"property": name}
        )

    if tag:

        return tag.get(
            "content",
            ""
        ).strip()

    return ""


# ==========================================
# 3. SCRAPE ONE ARTICLE
# ==========================================

def scrape_article(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:

            print(
                f"✗ HTTP {response.status_code}"
            )

            return None

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

        # -------------------------------
        # Title
        # -------------------------------

        title = get_meta(
            soup,
            "og:title"
        )

        if not title and soup.title:

            title = soup.title.get_text(
                " ",
                strip=True
            )

        # -------------------------------
        # Description
        # -------------------------------

        description = get_meta(
            soup,
            "description"
        )

        if not description:

            description = get_meta(
                soup,
                "og:description"
            )

        # -------------------------------
        # Published time
        # -------------------------------

        published_time = get_meta(
            soup,
            "article:published_time"
        )

        # -------------------------------
        # Updated time
        # -------------------------------

        updated_time = get_meta(
            soup,
            "article:modified_time"
        )

        # -------------------------------
        # Author
        # -------------------------------

        author = get_meta(
            soup,
            "author"
        )

        # -------------------------------
        # Article content
        # -------------------------------

        article = soup.find(
            "article"
        )

        content = ""

        if article:

            paragraphs = article.find_all(
                "p"
            )

            cleaned_paragraphs = []

            for p in paragraphs:

                text = p.get_text(
                    " ",
                    strip=True
                )

                if text:

                    cleaned_paragraphs.append(
                        text
                    )

            content = "\n".join(
                cleaned_paragraphs
            )

        # -------------------------------
        # Article ID
        # -------------------------------

        article_id = (
            url.rstrip("/")
            .split("/")[-1]
        )

        # -------------------------------
        # Word count
        # -------------------------------

        word_count = len(
            content.split()
        )

        # -------------------------------
        # Scraped timestamp
        # -------------------------------

        scraped_at = datetime.now(
            timezone.utc
        ).isoformat()

        # -------------------------------
        # Validation
        # -------------------------------

        if not title:

            print("✗ Missing title")

            return None

        if not content:

            print("✗ Empty content")

            return None

        return {

            "source": "CNN",

            "article_id": article_id,

            "title": title,

            "description": description,

            "published_time": published_time,

            "updated_time": updated_time,

            "author": author,

            "content": content,

            "url": url,

            "scraped_at": scraped_at,

            "word_count": word_count

        }

    except requests.exceptions.Timeout:

        print("✗ Request timeout - skipped")

        return None

    except requests.exceptions.RequestException as e:

        print(
            f"✗ Request error: {e}"
        )

        return None

    except Exception as e:

        print(
            f"✗ Error: "
            f"{type(e).__name__}: {e}"
        )

        return None


# ==========================================
# 4. MAIN CNN SCRAPER
# ==========================================

print("=" * 50)
print("STARTING CNN WEB SCRAPER")
print("=" * 50)


article_links = get_article_links()


print(
    f"\nTotal unique article links found: "
    f"{len(article_links)}"
)


if len(article_links) < TARGET_ARTICLES:

    print(
        f"\nWARNING: Only "
        f"{len(article_links)} links found."
    )


articles = []


for url in article_links:

    # Stop after exactly 100
    # successful articles

    if len(articles) >= TARGET_ARTICLES:

        break

    article_number = len(articles) + 1

    print(
        f"\nScraping article "
        f"{article_number}/{TARGET_ARTICLES}"
    )

    data = scrape_article(url)

    if data:

        articles.append(data)

        print(
            f"✓ Success: "
            f"{data['title']}"
        )

    else:

        print(
            "✗ Failed - moving to next article"
        )

    # Polite delay
    time.sleep(0.5)


# ==========================================
# 5. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(
    articles
)


# ==========================================
# 6. DATA QUALITY REPORT
# ==========================================

print("\n")
print("=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)


print(
    f"\nTotal records: {len(df)}"
)


if not df.empty:

    print("\nMissing values:")

    print(
        df.isna().sum()
    )

    print(
        "\nDuplicate URLs:",
        df["url"].duplicated().sum()
    )

    print(
        "Duplicate Article IDs:",
        df["article_id"].duplicated().sum()
    )

    print(
        "Empty content:",
        (
            df["content"]
            .str.strip()
            .eq("")
        ).sum()
    )

    print(
        "\nWord count statistics:"
    )

    print(
        df["word_count"].describe()
    )


# ==========================================
# 7. SAVE CNN DATASET
# ==========================================

BASE_DIR = Path(
    __file__
).resolve().parent


DATA_DIR = BASE_DIR / "data"


DATA_DIR.mkdir(
    exist_ok=True
)


output_file = (
    DATA_DIR /
    "cnn_articles.csv"
)


df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 8. FINAL MESSAGE
# ==========================================

print("\n")
print("=" * 50)
print("CNN SCRAPING COMPLETED")
print("=" * 50)


print(
    f"Articles collected: "
    f"{len(df)}"
)


print(
    f"CSV file: "
    f"{output_file}"
)


# ==========================================
# 9. DATASET PREVIEW
# ==========================================

if not df.empty:

    print(
        "\nDataset Preview:\n"
    )

    print(
        df[
            [
                "source",
                "title",
                "published_time",
                "author",
                "word_count"
            ]
        ].to_string(
            index=False
        )
    )