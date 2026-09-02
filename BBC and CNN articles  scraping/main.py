import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
from datetime import datetime, timezone
import pandas as pd
import time


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://www.bbc.com"

TARGET_ARTICLES = 100

SECTION_URLS = [
    "https://www.bbc.com/news",
    "https://www.bbc.com/news/world",
    "https://www.bbc.com/news/world/asia",
    "https://www.bbc.com/news/world/asia/india",
    "https://www.bbc.com/news/world/asia/pakistan",
    "https://www.bbc.com/news/world/europe",
    "https://www.bbc.com/news/world/us_and_canada",
    "https://www.bbc.com/news/politics",
    "https://www.bbc.com/news/business",
    "https://www.bbc.com/news/technology",
    "https://www.bbc.com/news/science_and_environment",
    "https://www.bbc.com/news/health",
    "https://www.bbc.com/news/entertainment_and_arts",
    "https://www.bbc.com/sport",
    "https://www.bbc.com/travel",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    )
}


# ============================================================
# GET BBC ARTICLE LINKS
# ============================================================

def get_article_links():

    article_links = set()

    print("\nStarting BBC link discovery...\n")

    for section_url in SECTION_URLS:

        try:

            response = requests.get(
                section_url,
                headers=HEADERS,
                timeout=20
            )

            print(
                f"{section_url} -> Status: {response.status_code}"
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "lxml"
            )

            for link in soup.find_all("a", href=True):

                href = link["href"]

                full_url = urljoin(
                    BASE_URL,
                    href
                )

                # ------------------------------------------------
                # Only BBC news article URLs
                # ------------------------------------------------

                if (
                    "bbc.com" in full_url
                    and "/news/" in full_url
                    and "/articles/" in full_url
                ):

                    # Remove query parameters/fragments
                    clean_url = full_url.split("?")[0]
                    clean_url = clean_url.split("#")[0]

                    article_links.add(clean_url)

        except requests.RequestException as error:

            print(
                f"Failed to retrieve: {section_url}"
            )

            print(
                "Error:",
                error
            )

        except Exception as error:

            print(
                f"Extraction failed: {section_url}"
            )

            print(
                "Error:",
                error
            )


        # Small delay between section requests
        time.sleep(1)


    return sorted(article_links)


# ============================================================
# EXTRACT ARTICLE DATA
# ============================================================

def extract_article(url):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "lxml"
    )


    article = {
        "source": "BBC",
        "article_id": url.rstrip("/").split("/")[-1],
        "title": None,
        "description": None,
        "published_time": None,
        "updated_time": None,
        "author": None,
        "content": None,
        "url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat()
    }


    # ========================================================
    # DESCRIPTION FROM META TAG
    # ========================================================

    description_tag = soup.find(
        "meta",
        attrs={"name": "description"}
    )

    if description_tag:

        article["description"] = description_tag.get(
            "content"
        )


    # ========================================================
    # JSON-LD STRUCTURED DATA
    # ========================================================

    json_ld_blocks = soup.find_all(
        "script",
        type="application/ld+json"
    )


    for block in json_ld_blocks:

        try:

            if not block.string:
                continue

            data = json.loads(
                block.string
            )

            if isinstance(data, dict):

                items = [data]

            elif isinstance(data, list):

                items = data

            else:

                continue


            for item in items:

                if not isinstance(item, dict):
                    continue


                # BBC can use different article types
                article_type = item.get("@type")

                if article_type in [
                    "ReportageNewsArticle",
                    "NewsArticle",
                    "Article"
                ]:


                    # ------------------------------------------------
                    # TITLE
                    # ------------------------------------------------

                    article["title"] = (
                        item.get("headline")
                        or article["title"]
                    )


                    # ------------------------------------------------
                    # DESCRIPTION
                    # ------------------------------------------------

                    article["description"] = (
                        item.get("description")
                        or article["description"]
                    )


                    # ------------------------------------------------
                    # PUBLISHED TIME
                    # ------------------------------------------------

                    article["published_time"] = (
                        item.get("datePublished")
                        or article["published_time"]
                    )


                    # ------------------------------------------------
                    # UPDATED TIME
                    # ------------------------------------------------

                    article["updated_time"] = (
                        item.get("dateModified")
                        or article["updated_time"]
                    )


                    # ------------------------------------------------
                    # AUTHOR
                    # ------------------------------------------------

                    author = item.get("author")


                    if isinstance(author, dict):

                        article["author"] = author.get(
                            "name"
                        )


                    elif isinstance(author, list):

                        names = []

                        for person in author:

                            if isinstance(person, dict):

                                name = person.get(
                                    "name"
                                )

                                if name:
                                    names.append(name)

                            elif isinstance(person, str):

                                names.append(person)


                        if names:

                            article["author"] = ", ".join(
                                names
                            )


                    elif isinstance(author, str):

                        article["author"] = author


                    # ------------------------------------------------
                    # ARTICLE BODY
                    # ------------------------------------------------

                    article["content"] = (
                        item.get("articleBody")
                        or article["content"]
                    )


        except (
            json.JSONDecodeError,
            TypeError
        ):

            continue


    # ========================================================
    # TITLE FALLBACK
    # ========================================================

    if not article["title"] and soup.title:

        article["title"] = soup.title.get_text(
            strip=True
        )


    # ========================================================
    # CONTENT FALLBACK
    # ========================================================

    if not article["content"]:

        article_tag = soup.find(
            "article"
        )


        if article_tag:

            paragraphs = article_tag.find_all(
                "p"
            )

            content_parts = []


            for paragraph in paragraphs:

                text = paragraph.get_text(
                    " ",
                    strip=True
                )

                if text:

                    content_parts.append(
                        text
                    )


            if content_parts:

                article["content"] = "\n".join(
                    content_parts
                )


    return article


# ============================================================
# MAIN SCRAPER
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("STARTING BBC WEB SCRAPER")
    print("=" * 50)


    # ========================================================
    # STEP 1: GET ARTICLE LINKS
    # ========================================================

    try:

        article_links = get_article_links()

        print(
            "\nTotal unique article links found:",
            len(article_links)
        )


    except Exception as error:

        print(
            "\nFailed to retrieve BBC links."
        )

        print(
            "Error:",
            error
        )

        raise SystemExit


    # ========================================================
    # STEP 2: SELECT FIRST 100 ARTICLES
    # ========================================================

    article_links = article_links[
        :TARGET_ARTICLES
    ]


    print(
        "Articles selected for scraping:",
        len(article_links)
    )


    # ========================================================
    # STEP 3: SCRAPE ARTICLES
    # ========================================================

    articles = []


    for index, url in enumerate(
        article_links,
        start=1
    ):

        print(
            f"\nScraping article {index}/{len(article_links)}"
        )


        try:

            article = extract_article(
                url
            )

            articles.append(
                article
            )


            print(
                "✓ Success:",
                article["title"]
            )


        except requests.RequestException as error:

            print(
                "✗ Request failed:",
                url
            )

            print(
                "Error:",
                error
            )


        except Exception as error:

            print(
                "✗ Extraction failed:",
                url
            )

            print(
                "Error:",
                error
            )


        # ----------------------------------------------------
        # Respectful delay
        # ----------------------------------------------------

        time.sleep(1)


    # ========================================================
    # STEP 4: CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        articles
    )


    # ========================================================
    # STEP 5: ADD WORD COUNT
    # ========================================================

    df["word_count"] = (
        df["content"]
        .fillna("")
        .str.split()
        .str.len()
    )


    # ========================================================
    # STEP 6: DATA QUALITY REPORT
    # ========================================================

    print(
        "\n" + "=" * 50
    )

    print(
        "DATA QUALITY REPORT"
    )

    print(
        "=" * 50
    )


    print(
        "\nTotal records:",
        len(df)
    )


    print(
        "\nMissing values:"
    )

    print(
        df.isnull().sum()
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
            .fillna("")
            .str.strip()
            == ""
        ).sum()
    )


    # ========================================================
    # WORD COUNT STATISTICS
    # ========================================================

    print(
        "\nWord count statistics:"
    )

    print(
        df["word_count"].describe()
    )


    # ========================================================
    # STEP 7: SAVE CSV
    # ========================================================

    output_file = (
        "data/bbc_articles.csv"
    )


    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # STEP 8: FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 50
    )

    print(
        "BBC SCRAPING COMPLETED"
    )

    print(
        "=" * 50
    )


    print(
        "Articles collected:",
        len(df)
    )


    print(
        "CSV file:",
        output_file
    )


    # ========================================================
    # STEP 9: DATASET PREVIEW
    # ========================================================

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