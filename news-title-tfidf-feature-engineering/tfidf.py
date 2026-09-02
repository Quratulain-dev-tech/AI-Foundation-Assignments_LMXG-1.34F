import os
import joblib
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Untitled spreadsheet - Sheet1.csv"

OUTPUT_CSV = "news_titles_tfidf.csv"
SPARSE_MATRIX_FILE = "tfidf_matrix.npz"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"

NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95
SUBLINEAR_TF = True


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 65)
print("TF-IDF FEATURE ENGINEERING")
print("=" * 65)

print("\n[1/9] Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print(f"Rows loaded: {len(df)}")


# ============================================================
# 2. VALIDATE INPUT
# ============================================================

print("\n[2/9] Validating input data...")

if "Title" not in df.columns:
    raise ValueError(
        "ERROR: Input CSV must contain a 'Title' column."
    )

df["Title"] = df["Title"].fillna("").astype(str)

empty_titles = (
    df["Title"].str.strip() == ""
).sum()

if empty_titles != 0:
    raise ValueError(
        f"ERROR: {empty_titles} empty titles found."
    )

input_rows = len(df)

print(f"Empty titles: {empty_titles}")
print("Input validation: PASSED")


# ============================================================
# 3. CREATE TF-IDF VECTORIZER
# ============================================================

print("\n[3/9] Creating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF,
    sublinear_tf=SUBLINEAR_TF
)


# ============================================================
# 4. APPLY TF-IDF
# ============================================================

print("\n[4/9] Applying TF-IDF to article titles...")

tfidf_matrix = vectorizer.fit_transform(
    df["Title"]
)

feature_names = vectorizer.get_feature_names_out()

n_rows, n_features = tfidf_matrix.shape
non_zero_values = tfidf_matrix.nnz

print("TF-IDF transformation: COMPLETED")
print(f"TF-IDF rows: {n_rows}")
print(f"TF-IDF features: {n_features}")
print(f"Non-zero TF-IDF values: {non_zero_values}")


# ============================================================
# 5. VALIDATE TF-IDF MATRIX
# ============================================================

print("\n[5/9] Validating TF-IDF matrix...")

if n_rows != input_rows:
    raise ValueError(
        "ERROR: TF-IDF row count does not match input row count."
    )

if n_features == 0:
    raise ValueError(
        "ERROR: No TF-IDF features were generated."
    )

print("TF-IDF matrix validation: PASSED")


# ============================================================
# 6. IDENTIFY ALL-ZERO DOCUMENTS
# ============================================================

print("\n[6/9] Checking document coverage...")

non_zero_per_row = tfidf_matrix.getnnz(axis=1)

zero_rows = [
    index
    for index, count in enumerate(non_zero_per_row)
    if count == 0
]

print(f"Total input titles: {input_rows}")
print(f"Titles with TF-IDF values: {input_rows - len(zero_rows)}")
print(f"All-zero TF-IDF titles: {len(zero_rows)}")

if zero_rows:
    print(f"All-zero row IDs: {zero_rows}")

print("Document coverage check: COMPLETED")


# ============================================================
# 7. CREATE COMPACT SPARSE CSV
# ============================================================

print("\n[7/9] Creating compact sparse TF-IDF CSV...")

# Get all non-zero TF-IDF positions.
row_indices, column_indices = tfidf_matrix.nonzero()

values = tfidf_matrix.data

# Map feature indexes to actual vocabulary terms.
features = feature_names[column_indices]

# Preserve original titles.
titles = df.iloc[row_indices]["Title"].to_numpy()

# Non-zero TF-IDF records.
sparse_records = pd.DataFrame({
    "row_id": row_indices,
    "Title": titles,
    "feature": features,
    "tfidf": values
})


# ------------------------------------------------------------
# Add explicit records for all-zero documents.
#
# These titles genuinely have no remaining TF-IDF features
# after the configured min_df/max_df filtering.
# We represent them with:
#
#     feature = "__NO_TFIDF_FEATURE__"
#     tfidf   = 0.0
#
# This does NOT create a fake TF-IDF feature. It is only a
# clear document-level marker so all 5,000 titles are present.
# ------------------------------------------------------------

if zero_rows:

    zero_records = pd.DataFrame({
        "row_id": zero_rows,
        "Title": df.iloc[zero_rows]["Title"].to_numpy(),
        "feature": "__NO_TFIDF_FEATURE__",
        "tfidf": 0.0
    })

    sparse_records = pd.concat(
        [
            sparse_records,
            zero_records
        ],
        ignore_index=True
    )


# Keep output ordered by original document.
sparse_records = sparse_records.sort_values(
    by=["row_id", "feature"],
    kind="stable"
).reset_index(drop=True)


# Remove previous CSV if present.
if os.path.exists(OUTPUT_CSV):

    try:
        os.remove(OUTPUT_CSV)

    except PermissionError:

        raise PermissionError(
            f"\nERROR: {OUTPUT_CSV} is currently open.\n"
            "Close it in WPS/Excel/another program and run again."
        )


# Save compact CSV.
sparse_records.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(f"CSV records written: {len(sparse_records)}")
print(f"CSV output: {OUTPUT_CSV}")


# ============================================================
# 8. SAVE REUSABLE TF-IDF ARTIFACTS
# ============================================================

print("\n[8/9] Saving reusable TF-IDF artifacts...")

save_npz(
    SPARSE_MATRIX_FILE,
    tfidf_matrix,
    compressed=True
)

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)

print(f"Sparse matrix: {SPARSE_MATRIX_FILE}")
print(f"Vectorizer: {VECTORIZER_FILE}")


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

print("\n[9/9] Final validation...")

saved_csv = pd.read_csv(
    OUTPUT_CSV,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# Validate columns
# ------------------------------------------------------------

expected_columns = [
    "row_id",
    "Title",
    "feature",
    "tfidf"
]

if list(saved_csv.columns) != expected_columns:

    raise ValueError(
        "ERROR: Output CSV columns are incorrect."
    )


# ------------------------------------------------------------
# Validate all 5,000 documents are represented
# ------------------------------------------------------------

unique_output_ids = saved_csv["row_id"].nunique()

if unique_output_ids != input_rows:

    raise ValueError(
        f"ERROR: Expected {input_rows} unique row IDs, "
        f"but found {unique_output_ids}."
    )


# ------------------------------------------------------------
# Validate row ID range
# ------------------------------------------------------------

if saved_csv["row_id"].min() != 0:

    raise ValueError(
        "ERROR: Output row IDs do not start at 0."
    )

if saved_csv["row_id"].max() != input_rows - 1:

    raise ValueError(
        "ERROR: Output row IDs do not cover all input rows."
    )


# ------------------------------------------------------------
# Validate titles
# ------------------------------------------------------------

output_titles = (
    saved_csv[["row_id", "Title"]]
    .drop_duplicates("row_id")
    .sort_values("row_id")["Title"]
    .reset_index(drop=True)
)

input_titles = (
    df["Title"]
    .reset_index(drop=True)
)

if not output_titles.equals(input_titles):

    raise ValueError(
        "ERROR: Original titles were not preserved correctly."
    )


# ------------------------------------------------------------
# Validate TF-IDF values
# ------------------------------------------------------------

if saved_csv["tfidf"].isna().any():

    raise ValueError(
        "ERROR: Missing TF-IDF values found."
    )

if (saved_csv["tfidf"] < 0).any():

    raise ValueError(
        "ERROR: Negative TF-IDF values detected."
    )


# ------------------------------------------------------------
# Validate all-zero markers
# ------------------------------------------------------------

marker_rows = saved_csv[
    saved_csv["feature"] == "__NO_TFIDF_FEATURE__"
]

if len(marker_rows) != len(zero_rows):

    raise ValueError(
        "ERROR: All-zero document markers do not match."
    )


if not (marker_rows["tfidf"] == 0.0).all():

    raise ValueError(
        "ERROR: Invalid TF-IDF value for all-zero documents."
    )


# ------------------------------------------------------------
# Validate output files
# ------------------------------------------------------------

required_files = [
    OUTPUT_CSV,
    SPARSE_MATRIX_FILE,
    VECTORIZER_FILE
]

for file_name in required_files:

    if not os.path.exists(file_name):

        raise FileNotFoundError(
            f"ERROR: Required file was not created: {file_name}"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("TF-IDF FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Input titles             : {input_rows}")
print(f"Output document IDs      : {unique_output_ids}")
print(f"TF-IDF features          : {n_features}")
print(f"Non-zero TF-IDF values   : {non_zero_values}")
print(f"All-zero titles          : {len(zero_rows)}")
print(f"CSV records              : {len(saved_csv)}")

print("\nTF-IDF configuration:")
print(f"  ngram_range            : {NGRAM_RANGE}")
print(f"  min_df                 : {MIN_DF}")
print(f"  max_df                 : {MAX_DF}")
print(f"  sublinear_tf           : {SUBLINEAR_TF}")

print("\nOutput files:")
print(f"  CSV                    : {OUTPUT_CSV}")
print(f"  Sparse matrix          : {SPARSE_MATRIX_FILE}")
print(f"  Vectorizer             : {VECTORIZER_FILE}")

print("\nValidation:")
print("  Input titles           : PASSED")
print("  Empty titles           : PASSED")
print("  TF-IDF generation      : PASSED")
print("  Document coverage      : PASSED")
print("  All titles preserved   : PASSED")
print("  CSV columns            : PASSED")
print("  TF-IDF values          : PASSED")
print("  All-zero handling      : PASSED")
print("  Sparse matrix saved    : PASSED")
print("  Vectorizer saved       : PASSED")
print("  Output files           : PASSED")

print("=" * 65)