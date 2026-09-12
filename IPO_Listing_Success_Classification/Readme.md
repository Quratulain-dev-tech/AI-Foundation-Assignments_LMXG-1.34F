# 📈 IPO Listing Success Classification using Machine Learning

> A complete end-to-end Machine Learning project that predicts whether an Initial Public Offering (IPO) is likely to achieve a positive listing gain using pre-listing subscription and financial features.

---

## 🚀 Project Overview

Initial Public Offerings (IPOs) are often associated with uncertainty regarding their listing-day performance. Investor demand, subscription levels, issue size, and offer price can influence whether an IPO experiences a positive or non-positive listing gain.

This project develops a **Supervised Machine Learning classification model** to predict IPO listing success.

An IPO is classified as:

- 🟢 **Successful (Class 1):** Positive Listing Gain
- 🔴 **Unsuccessful (Class 0):** Zero or Negative Listing Gain

The project follows a complete Machine Learning pipeline, including data cleaning, exploratory data analysis, feature engineering, data leakage prevention, hyperparameter tuning, class imbalance handling, threshold optimization, and final evaluation on unseen test data.

---

# 🎯 Project Objective

The main objective of this project is to build a Machine Learning model that predicts IPO listing success using information available before the IPO listing outcome.

The project specifically aims to:

- Analyze historical IPO data.
- Identify important pre-listing features.
- Create a binary classification target.
- Prevent data leakage.
- Train and optimize a Gradient Boosting classifier.
- Handle class imbalance.
- Optimize the classification threshold.
- Evaluate the model using unseen test data.
- Identify the most important factors influencing IPO listing success.

---

# 📊 Dataset

The dataset contains historical IPO observations.

### Dataset Size

| Description | Value |
|---|---:|
| Total IPO Records | **652** |
| Training Records | **521** |
| Testing Records | **131** |

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
