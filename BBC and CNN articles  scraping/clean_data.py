import pandas as pd
from pathlib import Path


# ============================================================
# DATA PREPROCESSING
# ============================================================

print("=" * 70)
print("WEB SCRAPED DATA PREPROCESSING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "scraped_articles.csv"
OUTPUT_FILE = DATA_DIR / "preprocessed_articles.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading scraped data...")

df = pd.read_csv(INPUT_FILE)

print(f"Original records: {len(df)}")
print(f"Original columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. REMOVE DUPLICATE URLs
# ============================================================

before = len(df)

df = df.drop_duplicates(subset=["url"])

after = len(df)

print(f"\nDuplicate URLs removed: {before - after}")


# ============================================================
# 3. REMOVE DUPLICATE ARTICLES
# ============================================================

before = len(df)

# Duplicate titles + content
df = df.drop_duplicates(
    subset=["title", "content"]
)

after = len(df)

print(f"Duplicate articles removed: {before - after}")


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")

text_columns = [
    "title",
    "description",
    "content",
    "author",
    "published_date"
]

for column in text_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
        )


# ============================================================
# 5. CLEAN TEXT
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
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )


# ============================================================
# 6. CLEAN SOURCE
# ============================================================

print("Standardizing sources...")

if "source" in df.columns:

    df["source"] = (
        df["source"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.upper()
    )


# ============================================================
# 7. CLEAN URLS
# ============================================================

print("Cleaning URLs...")

df["url"] = (
    df["url"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 8. CLEAN DATE
# ============================================================

print("Processing publication dates...")

if "published_date" in df.columns:

    df["published_date"] = pd.to_datetime(
        df["published_date"],
        errors="coerce",
        utc=True
    )


# ============================================================
# 9. CALCULATE WORD COUNT
# ============================================================

print("Calculating word count...")

df["word_count"] = (
    df["content"]
    .str.split()
    .str.len()
)


# ============================================================
# 10. CALCULATE CHARACTER COUNT
# ============================================================

print("Calculating character count...")

df["character_count"] = (
    df["content"]
    .str.len()
)


# ============================================================
# 11. REMOVE EMPTY CONTENT
# ============================================================

before = len(df)

df = df[
    df["content"].str.strip() != ""
]

after = len(df)

print(f"\nEmpty content articles removed: {before - after}")


# ============================================================
# 12. REMOVE EMPTY TITLES
# ============================================================

before = len(df)

df = df[
    df["title"].str.strip() != ""
]

after = len(df)

print(f"Empty title articles removed: {before - after}")


# ============================================================
# 13. REMOVE VERY SHORT CONTENT
# ============================================================

before = len(df)

df = df[
    df["word_count"] >= 20
]

after = len(df)

print(
    f"Articles with less than 20 words removed: "
    f"{before - after}"
)


# ============================================================
# 14. RESET INDEX
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# 15. DATA QUALITY REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PREPROCESSING REPORT")
print("=" * 70)

print(f"\nOriginal records       : {len(pd.read_csv(INPUT_FILE))}")
print(f"Final records         : {len(df)}")
print(f"Records removed       : {len(pd.read_csv(INPUT_FILE)) - len(df)}")


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    df.isna().sum()
)


# ============================================================
# DUPLICATE CHECK
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
# EMPTY CONTENT CHECK
# ============================================================

print("\nEmpty content:")

print(
    (df["content"].str.strip() == "").sum()
)


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

print("\nSource distribution:")

print(
    df["source"].value_counts()
)


# ============================================================
# WORD COUNT STATISTICS
# ============================================================

print("\nWord count statistics:")

print(
    df["word_count"].describe()
)


# ============================================================
# 16. SAVE PREPROCESSED DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. PREVIEW
# ============================================================

print("\n")
print("=" * 70)
print("DATA PREPROCESSING COMPLETED")
print("=" * 70)

print(f"\nPreprocessed file saved at:")

print(OUTPUT_FILE)

print(f"\nFinal dataset shape:")

print(df.shape)


print("\nDataset Preview:")

preview_columns = [
    "source",
    "title",
    "published_date",
    "author",
    "word_count",
    "character_count"
]

available_columns = [
    column
    for column in preview_columns
    if column in df.columns
]

print(
    df[available_columns]
    .head(10)
    .to_string(index=False)
)


print("\nDone! 🎉")