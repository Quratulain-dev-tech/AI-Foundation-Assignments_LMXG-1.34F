# 📈 IPO Listing Success Classification using Machine Learning

> An end-to-end Supervised Machine Learning project that predicts whether an Initial Public Offering (IPO) will achieve a positive listing gain using pre-listing subscription and financial information.

---

## 🏆 Model Performance

The final model was evaluated on **131 unseen IPO records** that were not used during model training.

| Metric | Score |
|---|---:|
| 🎯 Accuracy | **71.76%** |
| 🎯 Precision | **86.30%** |
| 🎯 Recall | **70.00%** |
| 🏆 F1 Score | **77.30%** |
| ⚖️ Macro F1 Score | **69.96%** |

### Class-Wise Performance

| IPO Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| 🔴 Unsuccessful IPO | 0.53 | 0.76 | 0.63 |
| 🟢 Successful IPO | 0.86 | 0.70 | 0.77 |

**Final Classification Threshold:** `0.49`

> 📌 The model was evaluated using a balanced approach to avoid relying only on accuracy. The Macro F1-score of **69.96%** shows the model's ability to identify both successful and unsuccessful IPOs.

---

## 🚀 Project Overview

Initial Public Offerings (IPOs) involve uncertainty regarding their listing-day performance. Investor demand, subscription levels, issue size, and offer price may influence whether an IPO achieves a positive listing gain.

This project uses **Supervised Machine Learning** to classify IPOs into two categories:

- 🟢 **Successful IPO (Class 1):** Positive Listing Gain
- 🔴 **Unsuccessful IPO (Class 0):** Zero or Negative Listing Gain

The project follows a complete Machine Learning pipeline from data loading and preprocessing to model training, hyperparameter tuning, threshold optimization, and final evaluation.

---

## 🎯 Project Objective

The objective of this project is to develop a Machine Learning model that predicts IPO listing success using information available before the IPO listing outcome.

The project aims to:

- Analyze historical IPO data.
- Clean and preprocess the dataset.
- Perform Exploratory Data Analysis (EDA).
- Create a binary classification target.
- Engineer relevant features.
- Prevent data leakage.
- Train a Gradient Boosting classifier.
- Optimize model hyperparameters.
- Handle class imbalance.
- Optimize the classification threshold.
- Evaluate the model on unseen test data.
- Identify important features influencing IPO listing success.

---

## 📊 Dataset

The dataset contains historical IPO observations.

### Dataset Size

| Dataset Split | Records |
|---|---:|
| Total IPO Records | **652** |
| Training Records | **521** |
| Testing Records | **131** |

The data was divided using an **80/20 stratified train-test split**.

### Original Dataset Features

```text
Date
IPO_Name
Issue_Size(crores)
QIB
HNI
RII
Total
Offer Price
List Price
Listing Gain
CMP(BSE)
CMP(NSE)
Current Gains
