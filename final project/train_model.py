"""
Step 4 (now also step 10): Train a model to predict spike / crash / neutral.

Uses a Random Forest classifier (robust default for tabular financial features,
handles non-linearity, gives feature importances for your report/viva).

If data/features/labeled_features_with_news.csv exists (built by
build_dataset.py), this trains on PRICE features + real scraped NEWS
sentiment features together. Otherwise it falls back to price-only features
(the original behavior).

Usage:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import config

PRICE_FEATURE_COLUMNS = [
    *[f"return_{w}d" for w in config.RETURN_WINDOWS],
    "volatility",
    "hl_range_pct",
    "volume_ratio",
    "gap_pct",
]
NEWS_FEATURE_COLUMNS = ["news_sentiment_mean", "news_count"]

COMBINED_PATH = os.path.join(config.FEATURES_DATA_DIR, "labeled_features_with_news.csv")
PRICE_ONLY_PATH = os.path.join(config.FEATURES_DATA_DIR, "labeled_features.csv")
FEATURES_META_PATH = os.path.join(config.MODEL_DIR, "features_used.joblib")


def load_dataset():
    """Prefer the combined (price + news) dataset if build_dataset.py has been run."""
    if os.path.exists(COMBINED_PATH):
        print(f"Using combined price+news dataset -> {COMBINED_PATH}")
        feature_columns = PRICE_FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS
        df = pd.read_csv(COMBINED_PATH, parse_dates=["Date"])
    else:
        print(f"No combined dataset found ({COMBINED_PATH}) -- "
              f"run build_dataset.py to include news features. Using price-only features for now.")
        feature_columns = PRICE_FEATURE_COLUMNS
        df = pd.read_csv(PRICE_ONLY_PATH, parse_dates=["Date"])

    df = df.dropna(subset=feature_columns + ["target"])
    return df, feature_columns


def train():
    df, feature_columns = load_dataset()
    X = df[feature_columns]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance (neutral days usually dominate)
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print("\n=== Classification report (test set) ===")
    print(classification_report(y_test, preds))
    print("Macro F1:", f1_score(y_test, preds, average="macro"))
    print("\nConfusion matrix (rows=true, cols=pred), labels =", sorted(classes))
    print(confusion_matrix(y_test, preds, labels=sorted(classes)))

    print("\n=== Feature importances ===")
    for name, imp in sorted(zip(feature_columns, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s} {imp:.4f}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(feature_columns, FEATURES_META_PATH)
    print(f"\nSaved model -> {config.MODEL_PATH}")
    print(f"Saved scaler -> {config.SCALER_PATH}")
    print(f"Saved feature list -> {FEATURES_META_PATH}")


if __name__ == "__main__":
    train()
