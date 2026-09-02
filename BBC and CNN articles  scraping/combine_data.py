import pandas as pd

# Load BBC dataset
bbc_df = pd.read_csv("data/bbc_articles.csv")

# Load CNN dataset
cnn_df = pd.read_csv("data/cnn_articles.csv")

# Combine both datasets
combined_df = pd.concat(
    [bbc_df, cnn_df],
    ignore_index=True
)

# Save combined dataset
output_file = "data/combined_news.csv"

combined_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 50)
print("DATASETS COMBINED SUCCESSFULLY")
print("=" * 50)

print(f"\nBBC articles: {len(bbc_df)}")
print(f"CNN articles: {len(cnn_df)}")
print(f"Total articles: {len(combined_df)}")

print(f"\nCombined CSV saved as: {output_file}")

print("\nDataset shape:")
print(combined_df.shape)

print("\nSource distribution:")
print(combined_df["source"].value_counts())

print("\nDataset preview:")
print(
    combined_df[
        [
            "source",
            "title",
            "published_time",
            "author",
            "word_count"
        ]
    ].head(10).to_string(index=False)
)