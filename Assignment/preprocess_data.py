import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


# ============================================================
# ALL-URL NEWS DATA PREPROCESSING
# ============================================================

print("=" * 70)
print("ALL-URL NEWS DATA PREPROCESSING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "scraped_all_articles.csv"
OUTPUT_FILE = DATA_DIR / "preprocessed_all_articles.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading scraped data...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8",
    low_memory=False
)

original_count = len(df)

print(f"Original records: {original_count}")
print(f"Original columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 1. REMOVE DUPLICATE URLs
# ============================================================

before = len(df)

df["url"] = (
    df["url"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df.drop_duplicates(
    subset=["url"],
    keep="first"
)

print(
    f"\nDuplicate URLs removed: "
    f"{before - len(df)}"
)


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")

text_columns = [
    "source",
    "title",
    "description",
    "content",
    "published_date",
    "author",
    "status_code",
    "response_time",
    "scraped_at",
    "status",
    "error"
]

for column in text_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )


# ============================================================
# 3. CLEAN TEXT
# ============================================================

print("Cleaning text...")

text_columns = [
    "title",
    "description",
    "content",
    "author"
]

for column in text_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )


# ============================================================
# 4. FIX COMMON ENCODING PROBLEMS
# ============================================================

def fix_encoding(text):

    if not isinstance(text, str):
        return text

    try:
        return text.encode(
            "latin1"
        ).decode(
            "utf-8"
        )
    except:
        return text


for column in [
    "title",
    "description",
    "content",
    "author"
]:

    if column in df.columns:

        df[column] = df[column].apply(
            fix_encoding
        )


# ============================================================
# 5. STANDARDIZE SOURCE
# ============================================================

print("Standardizing sources...")

df["source"] = (
    df["source"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# 6. CLEAN URLs
# ============================================================

print("Cleaning URLs...")


def clean_url(url):

    if not isinstance(url, str):
        return ""

    url = url.strip()

    if not url:
        return ""

    try:

        parts = urlsplit(url)

        cleaned = urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            "",
            ""
        ))

        return cleaned

    except:

        return url


df["url"] = df["url"].apply(
    clean_url
)


# ============================================================
# 7. REMOVE INVALID / EMPTY URLs
# ============================================================

before = len(df)

df = df[
    df["url"].str.strip() != ""
]

print(
    f"Empty URLs removed: "
    f"{before - len(df)}"
)


# ============================================================
# 8. PROCESS PUBLISHED DATE
# ============================================================

print("Processing publication dates...")

if "published_date" in df.columns:

    df["published_date"] = pd.to_datetime(
        df["published_date"],
        errors="coerce",
        utc=True
    )


# ============================================================
# 9. PROCESS SCRAPED DATE
# ============================================================

if "scraped_at" in df.columns:

    df["scraped_at"] = pd.to_datetime(
        df["scraped_at"],
        errors="coerce",
        utc=True
    )


# ============================================================
# 10. REMOVE EMPTY CONTENT
# ============================================================

before = len(df)

df = df[
    df["content"].str.strip() != ""
]

print(
    f"\nEmpty content articles removed: "
    f"{before - len(df)}"
)


# ============================================================
# 11. REMOVE EMPTY TITLES
# ============================================================

before = len(df)

df = df[
    df["title"].str.strip() != ""
]

print(
    f"Empty title articles removed: "
    f"{before - len(df)}"
)


# ============================================================
# 12. CALCULATE WORD COUNT
# ============================================================

print("Calculating word count...")

df["word_count"] = (
    df["content"]
    .str.split()
    .str.len()
)


# ============================================================
# 13. CALCULATE CHARACTER COUNT
# ============================================================

print("Calculating character count...")

df["character_count"] = (
    df["content"]
    .str.len()
)


# ============================================================
# 14. REMOVE VERY SHORT ARTICLES
# ============================================================

MIN_WORDS = 20

before = len(df)

df = df[
    df["word_count"] >= MIN_WORDS
]

print(
    f"Articles with less than "
    f"{MIN_WORDS} words removed: "
    f"{before - len(df)}"
)


# ============================================================
# 15. REMOVE DUPLICATE TITLE + CONTENT
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["title", "content"],
    keep="first"
)

print(
    f"Duplicate title + content removed: "
    f"{before - len(df)}"
)


# ============================================================
# 16. RESET INDEX
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# 17. FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PREPROCESSING REPORT")
print("=" * 70)

print(
    f"\nOriginal records       : "
    f"{original_count}"
)

print(
    f"Final records          : "
    f"{len(df)}"
)

print(
    f"Records removed        : "
    f"{original_count - len(df)}"
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    df.isna().sum()
)


# ============================================================
# DUPLICATES
# ============================================================

print("\nDuplicate URLs:")

print(
    df["url"].duplicated().sum()
)


print("\nDuplicate title + content:")

print(
    df.duplicated(
        subset=["title", "content"]
    ).sum()
)


# ============================================================
# EMPTY CONTENT
# ============================================================

print("\nEmpty content:")

print(
    (
        df["content"]
        .str.strip() == ""
    ).sum()
)


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

print("\nSource distribution:")

print(
    df["source"]
    .value_counts()
)


# ============================================================
# WORD COUNT STATISTICS
# ============================================================

print("\nWord count statistics:")

print(
    df["word_count"]
    .describe()
)


# ============================================================
# SAVE PREPROCESSED DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PREVIEW
# ============================================================

print("\n")
print("=" * 70)
print("DATA PREPROCESSING COMPLETED")
print("=" * 70)

print(
    f"\nPreprocessed file saved at:"
)

print(
    OUTPUT_FILE
)

print(
    f"\nFinal dataset shape:"
)

print(
    df.shape
)

print("\nDataset Preview:")

preview_columns = [
    "source",
    "title",
    "published_date",
    "author",
    "word_count",
    "character_count"
]

print(
    df[
        preview_columns
    ]
    .head(10)
    .to_string(index=False)
)

print("\nDone! 🎉")