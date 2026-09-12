"""
IPO LISTING SUCCESS CLASSIFICATION
FINAL MACHINE LEARNING PIPELINE

Pipeline:
Data Loading
    ↓
EDA
    ↓
Data Cleaning
    ↓
Target Creation
    ↓
Feature Engineering
    ↓
Data Leakage Removal
    ↓
Train/Test Split
    ↓
Hyperparameter Tuning
    ↓
Threshold Optimization using Training Cross-Validation
    ↓
Final Model Training
    ↓
Unseen Test Evaluation
    ↓
Feature Importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "Initial Public Offering (Updated).xlsx"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)

    else:
        df = pd.read_csv(path)

    print("\n[1] Dataset Loaded")
    print("Shape:", df.shape)

    print("\nColumns found:")
    print(df.columns.tolist())

    return df


# ============================================================
# 2. EDA
# ============================================================

def run_eda(df):

    print("\n[2] Exploratory Data Analysis")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Records ---")
    print("Duplicates:", df.duplicated().sum())

    print("\n--- Summary Statistics ---")
    print(df.describe(include="all").T)

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Histograms

    if len(numeric_cols) > 0:

        df[numeric_cols].hist(
            figsize=(15, 10),
            bins=30
        )

        plt.suptitle(
            "Distribution of Numeric Features"
        )

        plt.tight_layout()

        plt.savefig(
            "eda_histograms.png"
        )

        plt.close()

        print(
            "Saved: eda_histograms.png"
        )

    # Correlation Heatmap

    if len(numeric_cols) > 1:

        plt.figure(
            figsize=(12, 8)
        )

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title(
            "Correlation Heatmap"
        )

        plt.tight_layout()

        plt.savefig(
            "eda_correlation_heatmap.png"
        )

        plt.close()

        print(
            "Saved: eda_correlation_heatmap.png"
        )


# ============================================================
# 3. DATA CLEANING
# ============================================================

def clean_data(df):

    print("\n[3] Cleaning Data")

    # Standardize column names

    df.columns = [

        str(col)
        .strip()
        .lower()
        .replace(" ", "_")

        for col in df.columns
    ]

    # Remove duplicates

    before_rows = len(df)

    df = df.drop_duplicates()

    removed = before_rows - len(df)

    print(
        f"Removed {removed} duplicate rows"
    )

    # Numeric missing values

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        if df[col].isnull().sum() > 0:

            df[col] = df[col].fillna(
                df[col].median()
            )

    # Categorical missing values

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:

        if df[col].isnull().sum() > 0:

            mode_value = df[col].mode()

            if len(mode_value) > 0:

                df[col] = df[col].fillna(
                    mode_value[0]
                )

    print(
        "\nMissing values after cleaning:"
    )

    print(
        df.isnull().sum().sum()
    )

    return df


# ============================================================
# 4. TARGET CREATION
# ============================================================

def create_target(df):

    print("\n[4] Creating Target Variable")

    if "listing_gain" not in df.columns:

        raise ValueError(
            "'listing_gain' column not found."
        )

    print(
        "Using existing listing gain column:"
        " 'listing_gain'"
    )

    # 1 = Positive Listing Gain
    # 0 = Zero / Negative Listing Gain

    df["target"] = np.where(

        df["listing_gain"] > 0,

        1,

        0
    )

    print("\nClass Distribution:")

    print(
        df["target"]
        .value_counts()
        .sort_index()
    )

    plt.figure(
        figsize=(6, 4)
    )

    sns.countplot(
        x="target",
        data=df
    )

    plt.xticks(
        [0, 1],
        [
            "Unsuccessful",
            "Successful"
        ]
    )

    plt.title(
        "IPO Listing Success Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "target_class_distribution.png"
    )

    plt.close()

    print(
        "Saved: target_class_distribution.png"
    )

    return df


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    print("\n[5] Feature Engineering")

    # --------------------------------------------------------
    # AVG SUBSCRIPTION
    # --------------------------------------------------------

    subscription_cols = [

        "qib",
        "hni",
        "rii"
    ]

    available_cols = [

        col

        for col in subscription_cols

        if col in df.columns
    ]

    if len(available_cols) == 3:

        df["avg_subscription"] = (

            df[
                available_cols
            ].mean(axis=1)

        )

        print(
            "Created feature:"
            " avg_subscription"
        )

    else:

        raise ValueError(
            "QIB, HNI, and RII columns "
            "are required."
        )

    # --------------------------------------------------------
    # LEAKAGE REMOVAL
    # --------------------------------------------------------

    leakage_cols = [

        "list_price",
        "listing_gain",
        "cmp(bse)",
        "cmp(nse)",
        "current_gains"
    ]

    existing_leakage = [

        col

        for col in leakage_cols

        if col in df.columns
    ]

    print(
        "\nDropping leakage columns:"
    )

    print(
        existing_leakage
    )

    # --------------------------------------------------------
    # IDENTIFIER / DATE REMOVAL
    # --------------------------------------------------------

    identifier_cols = [

        "date",
        "ipo_name"
    ]

    existing_identifier = [

        col

        for col in identifier_cols

        if col in df.columns
    ]

    print(
        "\nDropping identifier/date columns:"
    )

    print(
        existing_identifier
    )

    # --------------------------------------------------------
    # FINAL FEATURES
    # --------------------------------------------------------

    final_features = [

        "issue_size(crores)",
        "qib",
        "hni",
        "rii",
        "total",
        "offer_price",
        "avg_subscription"
    ]

    missing_features = [

        col

        for col in final_features

        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: "
            f"{missing_features}"
        )

    features_df = df[
        final_features + ["target"]
    ].copy()

    print(
        "\nFinal Feature Columns:"
    )

    print(
        final_features
    )

    return features_df


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

def split_data(features_df):

    print("\n[6] Splitting Data")

    X = features_df.drop(
        columns=["target"]
    )

    y = features_df["target"]

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,
            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y
        )
    )

    print(
        "Train shape:",
        X_train.shape
    )

    print(
        "Test shape:",
        X_test.shape
    )

    print(
        "\nTraining Class Distribution:"
    )

    print(
        y_train
        .value_counts()
        .sort_index()
    )

    print(
        "\nTesting Class Distribution:"
    )

    print(
        y_test
        .value_counts()
        .sort_index()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

def tune_model(
    X_train,
    y_train
):

    print(
        "\n[7] Hyperparameter Tuning"
    )

    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    param_grid = {

        "n_estimators": [
            100,
            200,
            300
        ],

        "learning_rate": [
            0.01,
            0.05,
            0.1
        ],

        "max_depth": [
            2,
            3,
            4
        ],

        "min_samples_split": [
            2,
            5,
            10
        ]
    }

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        scoring="f1_macro",

        cv=cv,

        n_jobs=-1,

        verbose=1
    )

    # Class imbalance handling using sample weights

    sample_weights = compute_sample_weight(

        class_weight="balanced",

        y=y_train
    )

    grid_search.fit(

        X_train,

        y_train,

        sample_weight=sample_weights
    )

    print(
        "\nBest Parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest CV Macro F1:"
    )

    print(
        grid_search.best_score_
    )

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        cv
    )


# ============================================================
# 8. THRESHOLD OPTIMIZATION
# ============================================================

def find_best_threshold(
    model,
    X_train,
    y_train,
    cv
):

    print(
        "\n[8] Optimizing Decision Threshold"
    )

    # Balanced weights

    sample_weights = compute_sample_weight(

        class_weight="balanced",

        y=y_train
    )

    # Out-of-fold probabilities

    probabilities = cross_val_predict(

        model,

        X_train,

        y_train,

        cv=cv,

        method="predict_proba",

        n_jobs=-1,

        params={
            "sample_weight":
            sample_weights
        }

    )[:, 1]

    thresholds = np.arange(

        0.30,

        0.71,

        0.01
    )

    results = []

    best_threshold = 0.50
    best_macro_f1 = -1

    for threshold in thresholds:

        predictions = (

            probabilities >= threshold

        ).astype(int)

        macro_f1 = f1_score(

            y_train,

            predictions,

            average="macro",

            zero_division=0
        )

        accuracy = accuracy_score(

            y_train,

            predictions
        )

        results.append({

            "threshold":
            threshold,

            "accuracy":
            accuracy,

            "macro_f1":
            macro_f1

        })

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1

            best_threshold = threshold

    threshold_df = pd.DataFrame(
        results
    )

    threshold_df.to_csv(

        "threshold_optimization.csv",

        index=False
    )

    print(
        f"\nBest Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best Training CV Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    # Plot

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(

        threshold_df["threshold"],

        threshold_df["macro_f1"]
    )

    plt.xlabel(
        "Classification Threshold"
    )

    plt.ylabel(
        "Macro F1 Score"
    )

    plt.title(
        "Threshold Optimization"
    )

    plt.tight_layout()

    plt.savefig(
        "threshold_optimization.png"
    )

    plt.close()

    return best_threshold


# ============================================================
# 9. FINAL MODEL TRAINING
# ============================================================

def train_final_model(
    best_model,
    X_train,
    y_train
):

    print(
        "\n[9] Training Final Model"
    )

    sample_weights = compute_sample_weight(

        class_weight="balanced",

        y=y_train
    )

    best_model.fit(

        X_train,

        y_train,

        sample_weight=sample_weights
    )

    return best_model


# ============================================================
# 10. FINAL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold
):

    print(
        "\n[10] Evaluating Final Model"
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (

        probabilities >= threshold

    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(

        y_test,

        predictions,

        average="macro",

        zero_division=0
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(

        classification_report(

            y_test,

            predictions,

            zero_division=0
        )
    )

    # Confusion Matrix

    cm = confusion_matrix(

        y_test,

        predictions
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=[
            "Unsuccessful",
            "Successful"
        ],

        yticklabels=[
            "Unsuccessful",
            "Successful"
        ]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "Final Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png"
    )

    plt.close()

    print(
        "Saved: confusion_matrix.png"
    )

    return {

        "accuracy":
        accuracy,

        "precision":
        precision,

        "recall":
        recall,

        "f1":
        f1,

        "macro_f1":
        macro_f1
    }


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

def feature_importance(
    model,
    X_columns
):

    print(
        "\n[11] Feature Importance"
    )

    importance = pd.Series(

        model.feature_importances_,

        index=X_columns

    ).sort_values(
        ascending=False
    )

    print(
        importance
    )

    importance.to_csv(
        "feature_importance.csv"
    )

    plt.figure(
        figsize=(8, 5)
    )

    importance.plot(
        kind="bar"
    )

    plt.title(
        "Feature Importance"
    )

    plt.ylabel(
        "Importance Score"
    )

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png"
    )

    plt.close()

    print(
        "Saved: feature_importance.png"
    )

    return importance


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # Step 1
    df = load_data(
        DATA_PATH
    )

    # Step 2
    run_eda(
        df
    )

    # Step 3
    df = clean_data(
        df
    )

    # Step 4
    df = create_target(
        df
    )

    # Step 5
    features_df = engineer_features(
        df
    )

    # Step 6
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        features_df
    )

    # Step 7
    (
        best_model,
        best_params,
        cv
    ) = tune_model(
        X_train,
        y_train
    )

    # Step 8
    best_threshold = find_best_threshold(

        best_model,

        X_train,

        y_train,

        cv
    )

    # Step 9
    final_model = train_final_model(

        best_model,

        X_train,

        y_train
    )

    # Step 10
    results = evaluate_model(

        final_model,

        X_test,

        y_test,

        best_threshold
    )

    # Step 11
    importance = feature_importance(

        final_model,

        X_train.columns
    )

    # Final Summary

    print(
        "\n"
        "=================================================="
    )

    print(
        "FINAL PROJECT RESULTS"
    )

    print(
        "=================================================="
    )

    print(
        "\nBest Parameters:"
    )

    print(
        best_params
    )

    print(
        f"\nFinal Threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"\nTest Accuracy: "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Test Precision: "
        f"{results['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{results['recall']:.4f}"
    )

    print(
        f"Test F1 Score: "
        f"{results['f1']:.4f}"
    )

    print(
        f"Test Macro F1 Score: "
        f"{results['macro_f1']:.4f}"
    )

    print(
        "\nPipeline completed successfully."
    )


if __name__ == "__main__":

    main()