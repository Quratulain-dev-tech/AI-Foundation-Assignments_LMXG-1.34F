# ================================================================
# NEWS TITLE CLASSIFICATION
# TF-IDF + PCA + 5-FOLD CV + RANDOMIZED SEARCH + RANDOM FOREST
# INDUSTRY-STYLE END-TO-END PIPELINE
# ================================================================

import os
import json
import time
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_STATE = 42

DATA_FILE = "News_Classification_Labeling.xlsx"

OUTPUT_DIR = "News_RF_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# START
# ================================================================

start_time = time.time()

print("=" * 75)
print("NEWS TITLE CLASSIFICATION")
print("TF-IDF + PCA + 5-FOLD CV + RANDOMIZED SEARCH + RANDOM FOREST")
print("=" * 75)

# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1] Loading dataset...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nDataset not found: {DATA_FILE}\n"
        f"Please place the Excel file in the same folder as this Python script."
    )

df = pd.read_excel(DATA_FILE)

print(f"Dataset shape: {df.shape}")

# ================================================================
# 2. CHECK REQUIRED COLUMNS
# ================================================================

required_columns = ["Title", "Category"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' not found in Excel file.\n"
            f"Available columns: {list(df.columns)}"
        )

# ================================================================
# 3. DATA CLEANING
# ================================================================

print("\n[2] Cleaning data...")

df = df[required_columns].copy()

# Remove missing values
before_missing = len(df)

df = df.dropna(subset=["Title", "Category"])

after_missing = len(df)

print(f"Removed missing rows: {before_missing - after_missing}")

# Convert to string
df["Title"] = df["Title"].astype(str).str.strip()
df["Category"] = df["Category"].astype(str).str.strip()

# Remove empty titles
df = df[df["Title"].str.len() > 0]

# Remove duplicate titles
before_duplicates = len(df)

df = df.drop_duplicates(subset=["Title"])

after_duplicates = len(df)

print(f"Removed duplicate titles: {before_duplicates - after_duplicates}")

# Reset index
df = df.reset_index(drop=True)

print(f"Final dataset shape: {df.shape}")

# ================================================================
# 4. TEXT CLEANING FUNCTION
# ================================================================

print("\n[3] Preparing text...")

def clean_text(text):

    text = str(text).lower()

    # Replace URLs
    import re

    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep letters and numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["Clean_Title"] = df["Title"].apply(clean_text)

# ================================================================
# 5. FEATURE ENGINEERING
# ================================================================

print("\n[4] Feature engineering...")

df["Title_Length"] = df["Title"].str.len()

df["Word_Count"] = df["Title"].str.split().str.len()

df["Unique_Word_Count"] = (
    df["Clean_Title"]
    .apply(lambda x: len(set(x.split())))
)

df["Unique_Word_Ratio"] = (
    df["Unique_Word_Count"] /
    df["Word_Count"].replace(0, 1)
)

df["Digit_Count"] = (
    df["Title"]
    .str.count(r"\d")
)

print("Feature engineering completed.")

# ================================================================
# 6. CLASS DISTRIBUTION
# ================================================================

print("\n[5] Category distribution...")

class_distribution = (
    df["Category"]
    .value_counts()
    .sort_index()
)

print(class_distribution)

# ================================================================
# 7. LABEL ENCODING
# ================================================================

print("\n[6] Encoding target labels...")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df["Category"])

X_text = df["Clean_Title"]

print("\nCategories:")

for i, category in enumerate(label_encoder.classes_):
    print(f"{i}: {category}")

# ================================================================
# 8. TRAIN / TEST SPLIT
# ================================================================

print("\n[7] Train/Test split...")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {len(X_train_text)}")
print(f"Testing samples : {len(X_test_text)}")

# ================================================================
# 9. TF-IDF
# ================================================================

print("\n[8] Applying TF-IDF...")

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=8000,
    sublinear_tf=True,
    strip_accents="unicode"
)

X_train_tfidf = tfidf.fit_transform(X_train_text)

X_test_tfidf = tfidf.transform(X_test_text)

print(f"TF-IDF training shape: {X_train_tfidf.shape}")
print(f"TF-IDF testing shape : {X_test_tfidf.shape}")
print(f"Vocabulary size      : {len(tfidf.vocabulary_)}")

# ================================================================
# 10. CONVERT TO DENSE FOR PCA
# ================================================================

print("\n[9] Preparing data for PCA...")

X_train_dense = X_train_tfidf.toarray().astype(np.float32)

X_test_dense = X_test_tfidf.toarray().astype(np.float32)

print(f"Dense training shape: {X_train_dense.shape}")
print(f"Dense testing shape : {X_test_dense.shape}")

# ================================================================
# 11. PCA
# ================================================================

print("\n[10] Applying PCA...")

PCA_COMPONENTS = 500

pca = PCA(
    n_components=PCA_COMPONENTS,
    svd_solver="randomized",
    random_state=RANDOM_STATE
)

X_train_pca = pca.fit_transform(X_train_dense)

X_test_pca = pca.transform(X_test_dense)

explained_variance = pca.explained_variance_ratio_.sum()

print(f"PCA components       : {PCA_COMPONENTS}")
print(f"Variance explained   : {explained_variance * 100:.2f}%")
print(f"PCA training shape   : {X_train_pca.shape}")
print(f"PCA testing shape    : {X_test_pca.shape}")

# Free unnecessary dense matrices
del X_train_dense
del X_test_dense
del X_train_tfidf
del X_test_tfidf

# ================================================================
# 12. RANDOM FOREST
# ================================================================

print("\n[11] Creating Random Forest...")

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ================================================================
# 13. 5-FOLD STRATIFIED CROSS VALIDATION
# ================================================================

print("\n[12] Setting up 5-Fold Cross Validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# ================================================================
# 14. RANDOMIZED SEARCH
# ================================================================

print("\n[13] Setting up Randomized Search...")

# IMPORTANT:
# Much faster than 144-combination Grid Search.

param_distributions = {

    "n_estimators": [
        100,
        150,
        200,
        300,
        400
    ],

    "max_depth": [
        None,
        10,
        15,
        20,
        30,
        40
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2"
    ],

    "class_weight": [
        None,
        "balanced"
    ]
}

# Number of random parameter combinations
N_ITER = 25

print(f"Random parameter combinations: {N_ITER}")
print("Cross-validation folds       : 5")
print(f"Total RF fits approximately  : {N_ITER * 5}")

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring="f1_weighted",
    cv=cv,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    return_train_score=True
)

# ================================================================
# 15. START SEARCH
# ================================================================

print("\n[14] Starting Randomized Search...")
print("Please wait...\n")

search_start = time.time()

random_search.fit(
    X_train_pca,
    y_train
)

search_time = time.time() - search_start

print("\nRandomized Search completed.")

print(
    f"Search time: "
    f"{search_time / 60:.2f} minutes"
)

# ================================================================
# 16. BEST PARAMETERS
# ================================================================

best_model = random_search.best_estimator_

best_params = random_search.best_params_

best_cv_score = random_search.best_score_

print("\n" + "=" * 75)
print("BEST MODEL")
print("=" * 75)

print("\nBest Parameters:")

for parameter, value in best_params.items():
    print(f"{parameter}: {value}")

print(
    f"\nBest 5-Fold CV Weighted F1: "
    f"{best_cv_score:.4f}"
)

# ================================================================
# 17. TEST PREDICTION
# ================================================================

print("\n[15] Evaluating best model on test set...")

y_pred = best_model.predict(X_test_pca)

# ================================================================
# 18. METRICS
# ================================================================

accuracy = accuracy_score(y_test, y_pred)

precision_weighted = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 75)
print("FINAL TEST RESULTS")
print("=" * 75)

print(f"\nAccuracy           : {accuracy:.4f}")
print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Weighted F1        : {f1_weighted:.4f}")
print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Macro F1           : {f1_macro:.4f}")

# ================================================================
# 19. CLASSIFICATION REPORT
# ================================================================

report_dict = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ================================================================
# 20. CONFUSION MATRIX
# ================================================================

print("\n[16] Creating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title(
    "Random Forest Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

confusion_path = os.path.join(
    OUTPUT_DIR,
    "01_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# 21. CLASS DISTRIBUTION GRAPH
# ================================================================

print("[17] Creating class distribution graph...")

plt.figure(figsize=(12, 7))

class_distribution.plot(
    kind="bar"
)

plt.title(
    "News Category Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Category")
plt.ylabel("Number of Articles")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

class_distribution_path = os.path.join(
    OUTPUT_DIR,
    "02_class_distribution.png"
)

plt.savefig(
    class_distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# 22. MODEL PERFORMANCE GRAPH
# ================================================================

print("[18] Creating model performance graph...")

performance = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],
    "Score": [
        accuracy,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        precision_macro,
        recall_macro,
        f1_macro
    ]
})

plt.figure(figsize=(12, 7))

plt.bar(
    performance["Metric"],
    performance["Score"]
)

plt.ylim(0, 1.05)

plt.title(
    "Random Forest Performance",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel("Score")

plt.xticks(
    rotation=45,
    ha="right"
)

for i, value in enumerate(performance["Score"]):

    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=10
    )

plt.tight_layout()

performance_path = os.path.join(
    OUTPUT_DIR,
    "03_model_performance.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# 23. PER-CATEGORY F1 GRAPH
# ================================================================

print("[19] Creating per-category performance graph...")

category_metrics = report_df.loc[
    label_encoder.classes_,
    ["precision", "recall", "f1-score"]
]

category_metrics.plot(
    kind="bar",
    figsize=(14, 8)
)

plt.title(
    "Per-Category Classification Performance",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Category")
plt.ylabel("Score")

plt.ylim(0, 1.05)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend(
    title="Metric"
)

plt.tight_layout()

category_path = os.path.join(
    OUTPUT_DIR,
    "04_per_category_metrics.png"
)

plt.savefig(
    category_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# 24. PCA EXPLAINED VARIANCE
# ================================================================

print("[20] Creating PCA variance graph...")

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

plt.figure(figsize=(12, 7))

plt.plot(
    range(
        1,
        len(cumulative_variance) + 1
    ),
    cumulative_variance
)

plt.axhline(
    y=0.90,
    linestyle="--",
    label="90% Variance"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% Variance"
)

plt.title(
    "PCA Cumulative Explained Variance",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

pca_variance_path = os.path.join(
    OUTPUT_DIR,
    "05_pca_explained_variance.png"
)

plt.savefig(
    pca_variance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# 25. RANDOM FOREST FEATURE IMPORTANCE
# ================================================================

print("[21] Creating Random Forest feature importance graph...")

importance = best_model.feature_importances_

top_n = min(20, len(importance))

top_indices = np.argsort(importance)[-top_n:][::-1]

top_importance = importance[top_indices]

component_names = [
    f"PC{i + 1}"
    for i in top_indices
]

plt.figure(figsize=(12, 8))

plt.barh(
    component_names[::-1],
    top_importance[::-1]
)

plt.title(
    "Top 20 Important PCA Components",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Random Forest Importance")
plt.ylabel("Principal Component")

plt.tight_layout()

importance_path = os.path.join(
    OUTPUT_DIR,
    "06_feature_importance.png"
)

plt.savefig(
    importance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# 26. RANDOM SEARCH RESULTS
# ================================================================

print("[22] Preparing Random Search results...")

cv_results = pd.DataFrame(
    random_search.cv_results_
)

selected_columns = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "param_n_estimators",
    "param_max_depth",
    "param_min_samples_split",
    "param_min_samples_leaf",
    "param_max_features",
    "param_class_weight"
]

grid_results = cv_results[selected_columns].copy()

grid_results = grid_results.sort_values(
    "rank_test_score"
)

# ================================================================
# 27. PCA COMPONENT INFORMATION
# ================================================================

pca_information = pd.DataFrame({

    "Component": np.arange(
        1,
        PCA_COMPONENTS + 1
    ),

    "Explained_Variance_Ratio":
        pca.explained_variance_ratio_,

    "Cumulative_Explained_Variance":
        cumulative_variance
})

# ================================================================
# 28. SUMMARY TABLE
# ================================================================

summary_df = pd.DataFrame({

    "Metric": [

        "Dataset Samples",
        "Training Samples",
        "Testing Samples",

        "TF-IDF Features",
        "PCA Components",
        "PCA Variance Explained",

        "CV Folds",
        "Random Search Iterations",

        "Best CV Weighted F1",

        "Test Accuracy",
        "Test Weighted Precision",
        "Test Weighted Recall",
        "Test Weighted F1",

        "Test Macro Precision",
        "Test Macro Recall",
        "Test Macro F1",

        "Search Time (Minutes)",
        "Total Runtime (Minutes)"
    ],

    "Value": [

        len(df),
        len(X_train_text),
        len(X_test_text),

        len(tfidf.vocabulary_),
        PCA_COMPONENTS,
        f"{explained_variance * 100:.2f}%",

        5,
        N_ITER,

        best_cv_score,

        accuracy,
        precision_weighted,
        recall_weighted,
        f1_weighted,

        precision_macro,
        recall_macro,
        f1_macro,

        search_time / 60,
        (time.time() - start_time) / 60
    ]
})

# ================================================================
# 29. BEST PARAMETERS TABLE
# ================================================================

best_params_df = pd.DataFrame(
    list(best_params.items()),
    columns=["Parameter", "Best Value"]
)

# ================================================================
# 30. SAVE EXCEL REPORT
# ================================================================

print("\n[23] Creating Excel report...")

excel_path = os.path.join(
    OUTPUT_DIR,
    "News_RF_PCA_Report.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Project Summary",
        index=False
    )

    best_params_df.to_excel(
        writer,
        sheet_name="Best Parameters",
        index=False
    )

    class_distribution.reset_index(
        name="Article_Count"
    ).rename(
        columns={"index": "Category"}
    ).to_excel(
        writer,
        sheet_name="Class Distribution",
        index=False
    )

    report_df.to_excel(
        writer,
        sheet_name="Classification Report"
    )

    category_metrics.to_excel(
        writer,
        sheet_name="Category Metrics"
    )

    grid_results.to_excel(
        writer,
        sheet_name="Random Search Results",
        index=False
    )

    pca_information.to_excel(
        writer,
        sheet_name="PCA Information",
        index=False
    )

# ================================================================
# 31. SAVE MODEL
# ================================================================

print("[24] Saving trained model...")

model_package = {

    "tfidf": tfidf,

    "pca": pca,

    "model": best_model,

    "label_encoder": label_encoder,

    "random_state": RANDOM_STATE,

    "pca_components": PCA_COMPONENTS,

    "tfidf_features": len(tfidf.vocabulary_)
}

model_path = os.path.join(
    OUTPUT_DIR,
    "news_random_forest_model.joblib"
)

joblib.dump(
    model_package,
    model_path
)

# ================================================================
# 32. SAVE CONFIGURATION
# ================================================================

config = {

    "project": "News Title Classification",

    "algorithm": "Random Forest",

    "feature_engineering": [
        "Title Length",
        "Word Count",
        "Unique Word Count",
        "Unique Word Ratio",
        "Digit Count"
    ],

    "tfidf": {
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 8000,
        "sublinear_tf": True
    },

    "pca": {
        "components": PCA_COMPONENTS,
        "solver": "randomized",
        "variance_explained":
            float(explained_variance)
    },

    "cross_validation": {
        "method": "StratifiedKFold",
        "folds": 5,
        "shuffle": True,
        "random_state": RANDOM_STATE
    },

    "optimization": {
        "method": "RandomizedSearchCV",
        "iterations": N_ITER,
        "scoring": "f1_weighted",
        "random_state": RANDOM_STATE
    },

    "best_parameters": {
        key: str(value)
        for key, value in best_params.items()
    },

    "results": {
        "accuracy": float(accuracy),
        "weighted_precision":
            float(precision_weighted),
        "weighted_recall":
            float(recall_weighted),
        "weighted_f1":
            float(f1_weighted),
        "macro_precision":
            float(precision_macro),
        "macro_recall":
            float(recall_macro),
        "macro_f1":
            float(f1_macro)
    }
}

config_path = os.path.join(
    OUTPUT_DIR,
    "project_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )

# ================================================================
# 33. FINAL SUMMARY
# ================================================================

total_time = time.time() - start_time

print("\n")
print("=" * 75)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 75)

print(f"\nFinal Dataset       : {len(df)} articles")
print(f"Training Samples    : {len(X_train_text)}")
print(f"Testing Samples     : {len(X_test_text)}")

print(f"\nTF-IDF Features     : {len(tfidf.vocabulary_)}")
print(f"PCA Components      : {PCA_COMPONENTS}")
print(
    f"PCA Variance        : "
    f"{explained_variance * 100:.2f}%"
)

print("\nBEST RANDOM FOREST PARAMETERS")

for parameter, value in best_params.items():
    print(f"{parameter}: {value}")

print("\nFINAL TEST PERFORMANCE")

print(f"Accuracy            : {accuracy:.4f}")
print(f"Weighted Precision  : {precision_weighted:.4f}")
print(f"Weighted Recall     : {recall_weighted:.4f}")
print(f"Weighted F1         : {f1_weighted:.4f}")
print(f"Macro F1            : {f1_macro:.4f}")

print(
    f"\nTotal Runtime       : "
    f"{total_time / 60:.2f} minutes"
)

print("\nOutput files saved in:")

print(
    os.path.abspath(OUTPUT_DIR)
)

print("\nGenerated files:")

print("1. News_RF_PCA_Report.xlsx")
print("2. news_random_forest_model.joblib")
print("3. project_config.json")
print("4. 01_confusion_matrix.png")
print("5. 02_class_distribution.png")
print("6. 03_model_performance.png")
print("7. 04_per_category_metrics.png")
print("8. 05_pca_explained_variance.png")
print("9. 06_feature_importance.png")

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)