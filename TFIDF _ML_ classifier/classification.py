"""
News Category Classification
=============================
Column used for prediction (X): Title
Column to predict (y):          Category

Pipeline:
1. Load data
2. Clean text
3. (Demo) Word frequency dictionary -- taake samajh aaye TF-IDF ke andar kya hota hai
4. TF-IDF vectorization (Title ko numbers mein convert karna)
5. Train-test split
6. Train 3 models: Logistic Regression, Decision Tree, Random Forest
7. Evaluate: Accuracy, Precision, Recall, F1-score (compare all 3)
"""

import os
import pandas as pd
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # image files ke liye, koi pop-up window nahi khulegi
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# Saare graphs/images isi folder mein save hongi
os.makedirs("outputs", exist_ok=True)


# -------------------------------------------------------
# STEP 1: Load the Excel file
# -------------------------------------------------------
# Apni file ka path yahan daalo (agar .xlsx hai to path change kar dena)
df = pd.read_excel("News_Classification_Labeling.xlsx")

print("Total rows:", len(df))
print(df.head())
print("\nCategory counts:\n", df["Category"].value_counts())


# -------------------------------------------------------
# STEP 2: Clean the Title text
# -------------------------------------------------------
def clean_text(text):
    text = str(text).lower()                     # lowercase
    text = re.sub(r"[^a-z\s]", " ", text)         # numbers/punctuation hatao
    text = re.sub(r"\s+", " ", text).strip()      # extra spaces hatao
    return text

df["clean_title"] = df["Title"].apply(clean_text)

# -------------------------------------------------------
# STEP 2b: Drop rows with missing Title or Category
# -------------------------------------------------------
# train_test_split aur models NaN (khaali) values ke sath kaam nahi karte.
before = len(df)
df = df.dropna(subset=["Title", "Category"])
df = df[df["Category"].astype(str).str.strip() != ""]
after = len(df)
print(f"\nDropped {before - after} row(s) with missing Title/Category. Remaining: {after}")


# -------------------------------------------------------
# STEP 3 (DEMO ONLY): Word Frequency Dictionary
# -------------------------------------------------------
# Ye sirf samajhne ke liye hai -- TF-IDF ke peeche yehi concept hai.
# Har word kitni dafa aaya, uska dictionary bana rahe hain.
all_words = " ".join(df["clean_title"]).split()
word_freq = Counter(all_words)

print("\nTop 15 most common words in all titles:")
for word, count in word_freq.most_common(15):
    print(f"{word}: {count}")

# Note: Ye Counter/dictionary sirf analysis ke liye hai.
# Actual model training ke liye hum neeche TfidfVectorizer use karenge,
# jo automatically ye counting + weighting (TF-IDF formula) khud kar deta hai.


# -------------------------------------------------------
# STEP 4: TF-IDF Vectorization
# -------------------------------------------------------
# Models text nahi samajhte -- unhe numbers chahiye.
# TF-IDF har title ko ek numeric vector mein convert kar deta hai.
tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X = tfidf.fit_transform(df["clean_title"])
y = df["Category"]


# -------------------------------------------------------
# STEP 5: Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


# -------------------------------------------------------
# STEP 6: Train 3 different models & compare
# -------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    })

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- Confusion Matrix image save karna ---
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(f"outputs/confusion_matrix_{safe_name}.png")
    plt.close(fig)


# -------------------------------------------------------
# STEP 7: Final comparison table
# -------------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1 Score", ascending=False)
print("\n\nFINAL COMPARISON:")
print(results_df)

# CSV bhi save kar do -- GitHub / README ke liye kaam aayega
results_df.to_csv("outputs/model_comparison.csv", index=False)

# Sabse zyada F1 Score wala model = best model for this data


# -------------------------------------------------------
# STEP 8: Comparison Bar Chart (Accuracy/Precision/Recall/F1)
# -------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = range(len(results_df))
width = 0.2

fig, ax = plt.subplots(figsize=(9, 5))
for i, metric in enumerate(metrics):
    ax.bar([p + i * width for p in x], results_df[metric], width=width, label=metric)

ax.set_xticks([p + width * 1.5 for p in x])
ax.set_xticklabels(results_df["Model"])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison_chart.png")
plt.close(fig)

print("\nSaved: outputs/confusion_matrix_*.png, outputs/model_comparison_chart.png, outputs/model_comparison.csv")