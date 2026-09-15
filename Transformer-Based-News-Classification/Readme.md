# Transformer-Based News Classification

A Transformer-based Natural Language Processing (NLP) project for automatically classifying news article titles into different news categories using **RoBERTa**.

The project focuses on multiclass news classification using a real-world labeled news dataset and evaluates the model using accuracy, precision, recall, F1-score, and a confusion matrix.

---

## 📌 Project Overview

News websites contain a large amount of textual information that needs to be organized into meaningful categories.

This project develops a Transformer-based text classification system that takes a **news article title** as input and predicts its corresponding category.

The implemented model is:

- **RoBERTa (`roberta-base`)**

A previously obtained **DistilBERT baseline** is also included for performance comparison, although its training cells are not part of the current notebook.

---

## 🎯 Objectives

The main objectives of this project are:

- Build a Transformer-based news classification system.
- Classify news titles into multiple categories.
- Apply appropriate text preprocessing and label encoding.
- Fine-tune RoBERTa for multiclass classification.
- Evaluate the model using multiple performance metrics.
- Analyze class-wise performance.
- Compare RoBERTa with a previously obtained DistilBERT baseline.
- Provide a foundation for future NLP tasks such as Named Entity Recognition (NER).

---

## 📊 Dataset

The dataset contains **5,000 news records** with the following columns:

| Column | Description |
|---|---|
| URL | URL of the news article |
| Title | News article title |
| Category | News category |

The classification task uses:

- **Input:** `Title`
- **Target:** `Category`

The `URL` column is not used for model training.

### Categories

The dataset contains six categories:

- Business
- Energy
- Health
- Markets
- Politics
- Technology

One record had a missing category value and was removed before training.

After cleaning:

**4,999 records** remained.

### Class Distribution

| Category | Records |
|---|---:|
| Business | 1,896 |
| Markets | 1,518 |
| Politics | 862 |
| Technology | 452 |
| Health | 144 |
| Energy | 127 |

The dataset is imbalanced, with Business and Markets containing substantially more samples than Energy and Health.

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

1. Removed records with missing category values.
2. Selected only `Title` and `Category`.
3. Converted text and category values to string format.
4. Applied label encoding to convert categories into numerical labels.
5. Performed a stratified train-validation-test split.

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training | 3,999 |
| Validation | 500 |
| Testing | 500 |
| **Total** | **4,999** |

Stratification was used to maintain similar class distributions across the splits.

---

## 🤖 Model

### RoBERTa

The main model used in this project is:

**`roberta-base`**

RoBERTa is a Transformer-based language model designed to produce contextual representations of text.

For this project, RoBERTa was fine-tuned for a **6-class news classification task**.

### Configuration

| Parameter | Value |
|---|---|
| Model | `roberta-base` |
| Number of classes | 6 |
| Maximum sequence length | 128 |
| Epochs | 5 |
| Training batch size | 16 |
| Evaluation batch size | 16 |
| Learning rate | 2e-5 |
| Evaluation | After each epoch |
| Best model selection | Weighted F1 |

---

## 📈 Results

The final RoBERTa model achieved the following performance on the held-out test set:

| Metric | RoBERTa |
|---|---:|
| Accuracy | **71.40%** |
| Precision | **71.42%** |
| Recall | **71.40%** |
| Weighted F1 | **71.28%** |

### RoBERTa Class-wise Performance

| Category | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Business | 0.69 | 0.68 | 0.68 |
| Energy | 0.58 | 0.58 | 0.58 |
| Health | 0.70 | 0.47 | 0.56 |
| Markets | 0.73 | 0.72 | 0.73 |
| Politics | 0.77 | 0.78 | 0.77 |
| Technology | 0.69 | 0.82 | 0.75 |

The strongest performance was observed for **Politics, Technology, and Markets**, while **Health and Energy** remain more challenging due partly to their smaller number of training examples.

---

## 🔄 Comparison with DistilBERT Baseline

A DistilBERT experiment was previously completed and used as a baseline for comparison.

> **Note:** The DistilBERT training cells were removed from the current notebook. Therefore, the current notebook contains the RoBERTa implementation, while the DistilBERT numbers below represent the previously obtained baseline results.

| Model | Accuracy | Precision | Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| DistilBERT | 70.20% | 70.15% | 70.20% | 70.03% |
| **RoBERTa** | **71.40%** | **71.42%** | **71.40%** | **71.28%** |

### Improvement

Compared with the previously obtained DistilBERT baseline:

- Accuracy improved by **1.20 percentage points**
- Weighted F1 improved by **1.25 percentage points**

This indicates that RoBERTa provided a modest improvement over the previous DistilBERT baseline on this dataset.

---

## 📊 Evaluation

The final model was evaluated using:

- Accuracy
- Precision
- Recall
- Weighted F1-score
- Classification report
- Confusion matrix

Both overall and class-wise performance were analyzed to understand how well the model performs across different news categories.

---

## 🔬 Training Process

The overall workflow is:

```text
Raw News Dataset
       ↓
Data Cleaning
       ↓
Remove Missing Labels
       ↓
Select Title + Category
       ↓
Label Encoding
       ↓
Stratified Train / Validation / Test Split
       ↓
RoBERTa Tokenization
       ↓
RoBERTa Fine-Tuning
       ↓
Validation After Each Epoch
       ↓
Best Model Selection
       ↓
Test Set Evaluation
       ↓
Classification Report
       ↓
Confusion Matrix
