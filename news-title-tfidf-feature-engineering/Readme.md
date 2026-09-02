# 📊 News Title TF-IDF Feature Engineering

A clean and reproducible Natural Language Processing (NLP) feature-engineering pipeline for converting **5,000 news article titles** into numerical **TF-IDF (Term Frequency–Inverse Document Frequency)** features.

---

## 🚀 Project Overview

This project applies **TF-IDF feature engineering** to a dataset containing 5,000 news article titles.

The pipeline takes raw news titles as text and converts them into numerical feature vectors. These numerical features can then be used as input for **Machine Learning, NLP, text classification, clustering, similarity analysis, and other downstream tasks**.

The pipeline is designed to be:

* Clean
* Reproducible
* Validated
* Memory-efficient
* Easy to understand
* Suitable for further Machine Learning workflows

---

## 🎯 Objective

The main objective is to transform news article titles into meaningful numerical representations while:

* Preserving the original article titles
* Maintaining complete document coverage
* Assigning importance scores to words and phrases
* Generating a reusable TF-IDF vectorizer
* Saving the resulting feature matrix for future ML/NLP tasks

---

## 🧠 What is TF-IDF?

**TF-IDF** stands for **Term Frequency–Inverse Document Frequency**.

It is a common technique in NLP used to measure how important a word is within a collection of documents.

TF-IDF considers two main factors:

### Term Frequency (TF)

Measures how frequently a word appears in a particular document.

A word that appears more frequently in a title can receive a higher TF value.

### Inverse Document Frequency (IDF)

Measures how rare or common a word is across all documents.

* Common words receive lower importance.
* Rare and more distinctive words receive higher importance.

### Final TF-IDF Score

The TF and IDF values are combined to produce a numerical score for each term.

This allows the text data to be represented as numerical features that Machine Learning algorithms can process.

---

## 📥 Input Data

The pipeline uses a dataset containing:

**5,000 news article titles**

Each title is treated as an individual document.

For example:

```text
"AI technology is transforming healthcare"
"Global markets rise after economic announcement"
"New technology improves machine learning systems"
```

Each title is converted into a numerical TF-IDF vector.

---

## ⚙️ Processing Pipeline

The project follows a simple and reproducible workflow:

```text
News Article Titles
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Numerical Feature Matrix
        ↓
Validation
        ↓
Saved Output Files
```

The resulting matrix contains numerical TF-IDF values representing the importance of terms across the 5,000 news titles.

---

## 📤 Output

The pipeline generates the following files:

### `news_titles_tfidf.csv`

Contains the TF-IDF feature representation in CSV format.

This format makes the generated features easy to inspect, analyze, and use in other workflows.

### `tfidf_matrix.npz`

Stores the TF-IDF matrix in **sparse matrix format**.

Sparse storage is useful because text feature matrices usually contain many zero values, allowing the data to be stored more efficiently.

### `tfidf_vectorizer.joblib`

Contains the fitted TF-IDF vectorizer.

Saving the vectorizer makes it possible to reuse the same vocabulary and transformation settings on new article titles.

---

## 📂 Project Structure

```text
feature-engineering/
│
├── tfidf.py
├── news_titles_tfidf.csv
├── tfidf_matrix.npz
├── tfidf_vectorizer.joblib
└── README.md
```

### File Description

| File                      | Purpose                                |
| ------------------------- | -------------------------------------- |
| `tfidf.py`                | Main TF-IDF feature-engineering script |
| `news_titles_tfidf.csv`   | TF-IDF features in CSV format          |
| `tfidf_matrix.npz`        | Sparse TF-IDF feature matrix           |
| `tfidf_vectorizer.joblib` | Saved fitted TF-IDF vectorizer         |
| `README.md`               | Project documentation                  |

---

## 🔍 Validation

The pipeline includes validation to ensure that the generated features are consistent with the input data.

The validation focuses on:

* Number of input titles
* Number of generated documents
* Feature matrix dimensions
* Preservation of document coverage
* Successful vectorizer creation
* Successful output file generation

The expected document count is:

**5,000 input titles → 5,000 TF-IDF document vectors**

---

## 💾 Memory Efficiency

TF-IDF matrices can become large because every document may contain many possible terms.

To improve memory efficiency, the project stores the TF-IDF matrix using a **sparse matrix format**.

Instead of storing large numbers of zero values, sparse storage keeps only the meaningful non-zero values.

This makes the feature representation more efficient and suitable for further NLP and Machine Learning processing.

---

## 🔄 Reproducibility

The fitted TF-IDF vectorizer is saved as:

```text
tfidf_vectorizer.joblib
```

This allows the same vocabulary and TF-IDF transformation to be reused later.

For example, new news titles can be transformed using the previously fitted vectorizer without creating a completely new feature space.

---

## 🤖 Possible Machine Learning Applications

The generated TF-IDF features can be used for various NLP and Machine Learning tasks, including:

* News topic classification
* Text classification
* News similarity
* Duplicate article detection
* Clustering
* Search and information retrieval
* Recommendation systems
* Sentiment analysis
* Machine Learning model training

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **SciPy**
* **Joblib**
* **TF-IDF Vectorization**

---

## ▶️ Running the Project

Run the main Python script:

```bash
python tfidf.py
```

After successful execution, the generated feature files will be available in the project directory.

---

## 📌 Summary

This project provides a clean and reproducible approach for converting **5,000 news article titles into numerical TF-IDF features**.

The resulting features are stored in both CSV and sparse matrix formats, while the fitted vectorizer is saved for future reuse.

The generated TF-IDF representation can serve as a foundation for subsequent **NLP and Machine Learning workflows**.
