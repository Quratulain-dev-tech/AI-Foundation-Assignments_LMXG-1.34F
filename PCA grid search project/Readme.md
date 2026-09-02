# 📰 News Article Classification using PCA & Grid Search

An end-to-end **Machine Learning pipeline for News Article Classification** using **TF-IDF Feature Extraction, Principal Component Analysis (PCA), K-Fold Cross-Validation, and Grid Search Hyperparameter Optimization with Random Forest**.

The project focuses on reducing the dimensionality of high-dimensional text data while systematically finding the best Random Forest configuration through cross-validation.

---

## 📌 Project Overview

News classification is a text classification problem where each news article/title is assigned to a predefined category.

Text data usually contains thousands of unique words, resulting in a **very high-dimensional feature space**. This project addresses that problem using:

* **TF-IDF** for converting text into numerical features
* **PCA** for dimensionality reduction
* **K-Fold Cross-Validation** for reliable model evaluation
* **Grid Search** for hyperparameter optimization
* **Random Forest** for classification
* **Accuracy, Precision, Recall and F1-Score** for performance evaluation
* **Confusion Matrix** for detailed classification analysis

---

## 🚀 Machine Learning Pipeline

```text
                    NEWS DATASET
                         │
                         ▼
                ┌─────────────────┐
                │  Data Loading   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Data Cleaning  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     TF-IDF      │
                │ Feature Matrix  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │       PCA       │
                │ Dimensionality  │
                │    Reduction    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Train / Test   │
                │      Split      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Grid Search +  │
                │  K-Fold CV      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Random Forest   │
                │ Best Parameters │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Evaluation    │
                └────────┬────────┘
                         │
                         ▼
              Accuracy / Precision
              Recall / F1 / Matrix
```

---

## 🧠 Technologies & Libraries

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Core programming language    |
| Pandas       | Data processing              |
| NumPy        | Numerical operations         |
| Scikit-learn | Machine Learning             |
| SciPy        | Sparse matrix handling       |
| OpenPyXL     | Excel report generation      |
| Joblib       | Model/Pipeline serialization |
| Matplotlib   | Visualization                |

---

## 🔹 1. TF-IDF Feature Engineering

The raw news text cannot be directly provided to a machine learning algorithm.

TF-IDF (**Term Frequency–Inverse Document Frequency**) converts textual data into numerical feature vectors.

The basic idea is:

```text
Raw News Text
      ↓
Tokenization / Processing
      ↓
TF-IDF
      ↓
Numerical Feature Matrix
```

The resulting matrix can contain thousands of dimensions because every unique word can become a feature.

For example:

```text
Training samples : 3650
Testing samples  : 913
TF-IDF features  : 8000
```

This creates a high-dimensional feature space, which motivates the use of PCA.

---

## 🔹 2. Principal Component Analysis (PCA)

**Principal Component Analysis** is used to reduce the dimensionality of the TF-IDF feature space.

Instead of using all original features, PCA transforms them into a smaller number of principal components that capture the most important variance in the data.

```text
8000 TF-IDF Features
          ↓
         PCA
          ↓
Reduced Feature Space
```

### Why PCA?

PCA can help:

* Reduce dimensionality
* Reduce computational cost
* Remove redundant information
* Make model training more manageable
* Potentially improve generalization

The project uses **actual PCA** rather than simply selecting a fixed number of original features.

---

## 🔹 3. Train-Test Split

The dataset is divided into training and testing sets.

```text
Dataset
   │
   ├──────────────► Training Set
   │
   └──────────────► Testing Set
```

The training set is used for:

* Model training
* Cross-validation
* Hyperparameter optimization

The testing set is kept separate and is used for the final unbiased evaluation.

---

## 🔹 4. K-Fold Cross-Validation

Instead of depending on a single validation split, **K-Fold Cross-Validation** divides the training data into multiple folds.

Example:

```text
Fold 1 → Validation
Fold 2 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training

Then the process is repeated so every fold becomes validation data.
```

This provides a more reliable estimate of model performance.

---

## 🔹 5. Grid Search

Random Forest has multiple hyperparameters that affect its performance.

Instead of manually selecting these parameters, **GridSearchCV** evaluates different combinations.

Typical parameters include:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

The process is:

```text
Parameter Combinations
          ↓
     Grid Search
          ↓
     K-Fold CV
          ↓
Evaluate Each Combination
          ↓
   Best Parameters
```

The combination producing the best cross-validation performance is selected.

---

## 🔹 6. Random Forest Classifier

The optimized classifier used in this project is **Random Forest**.

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

```text
              Random Forest
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      Tree 1      Tree 2      Tree 3
        │           │           │
        └───────────┼───────────┘
                    ▼
             Final Prediction
```

Using multiple trees helps the model learn complex patterns in the reduced feature space.

---

## 📊 Model Evaluation

The final model is evaluated using multiple classification metrics.

### Accuracy

Measures the percentage of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted samples for a class were actually correct.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall)
     --------------------------
       Precision + Recall
```

---

## 📈 Evaluation Outputs

The project generates detailed evaluation results including:

* Overall Accuracy
* Macro Precision
* Macro Recall
* Macro F1-Score
* Weighted Precision
* Weighted Recall
* Weighted F1-Score
* Per-category classification metrics
* Confusion Matrix
* Best Grid Search parameters
* Cross-validation score

---

## 🔥 Confusion Matrix

The confusion matrix provides a detailed view of how the classifier performs across different news categories.

```text
                 Predicted
              ┌────┬────┬────┐
Actual Class  │ C1 │ C2 │ C3 │
              ├────┼────┼────┤
       C1     │ ✓  │    │    │
       C2     │    │ ✓  │    │
       C3     │    │    │ ✓  │
              └────┴────┴────┘
```

Diagonal values represent correctly classified samples, while off-diagonal values represent classification errors.

---

## 💾 Model Serialization with Joblib

The trained model can be saved using **Joblib** so that it does not need to be trained from scratch every time.

For example:

```python
joblib.dump(model, "random_forest_model.joblib")
```

The saved model can later be loaded:

```python
model = joblib.load("random_forest_model.joblib")
```

This is useful when deploying the classifier for prediction.

If the project saves the **TF-IDF vectorizer and PCA object separately**, those should also be preserved because a new input must go through the same feature transformation pipeline before prediction.

---

## 📁 Project Structure

```text
PCA_Grid_Search/
│
├── news_pca_random_forest.py
│
├── dataset/
│   ├── news_data.csv
│   └── categories.xlsx
│
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── pca_model.joblib
│   └── random_forest_model.joblib
│
├── results/
│   └── PCA_GridSearch_Results.xlsx
│
├── README.md
└── requirements.txt
```

> Adjust the filenames above according to the actual files in your repository.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd PCA_Grid_Search
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install the main libraries directly:

```bash
pip install pandas numpy scikit-learn scipy openpyxl joblib matplotlib
```

---

## ▶️ How to Run

Run the main Python script:

```bash
python news_pca_random_forest.py
```

The pipeline will:

1. Load the dataset
2. Prepare the text and labels
3. Generate TF-IDF features
4. Apply PCA
5. Split the data
6. Perform K-Fold Cross-Validation
7. Run Grid Search
8. Select the best Random Forest parameters
9. Train the optimized model
10. Evaluate the model
11. Generate the confusion matrix
12. Save the results/model files

---

## 📋 Example Pipeline Output

```text
========================================================
NEWS ARTICLE CLASSIFICATION
TF-IDF + ACTUAL PCA + K-FOLD + GRID SEARCH
========================================================

Training samples: 3650
Testing samples : 913

Applying TF-IDF...
TF-IDF training shape: (3650, 8000)
TF-IDF testing shape : (913, 8000)

Applying PCA...
PCA transformation completed.

Running Grid Search...
Running K-Fold Cross-Validation...

Best Parameters:
...

Best CV Score:
...

Test Accuracy:
...

Classification Report:
...

Confusion Matrix:
...
```

---

## 📊 Results

The final results are organized into a structured report containing:

### Model Performance

| Metric        | Result                           |
| ------------- | -------------------------------- |
| Best CV Score | Generated by Grid Search         |
| Test Accuracy | Generated after final evaluation |
| Precision     | Generated after evaluation       |
| Recall        | Generated after evaluation       |
| F1-Score      | Generated after evaluation       |

### Per-Category Performance

The classification report provides precision, recall and F1-score for every news category.

### Confusion Matrix

A confusion matrix is generated for the final optimized Random Forest model.

---

## 🎯 Key Features

✅ TF-IDF based text feature engineering
✅ High-dimensional text processing
✅ Actual PCA dimensionality reduction
✅ Train/Test separation
✅ K-Fold Cross-Validation
✅ Automated Grid Search
✅ Random Forest classification
✅ Best hyperparameter selection
✅ Multiple evaluation metrics
✅ Confusion Matrix
✅ Excel-based result reporting
✅ Joblib model serialization
✅ Reproducible ML workflow

---

## 🔬 Why This Approach?

This project combines several important Machine Learning techniques into a single end-to-end pipeline.

```text
TF-IDF
  ↓
High-Dimensional Text Features
  ↓
PCA
  ↓
Reduced Feature Space
  ↓
K-Fold Cross-Validation
  ↓
Grid Search
  ↓
Optimized Random Forest
  ↓
Final Evaluation
```

This approach provides a systematic way to build and optimize a text classification model instead of relying on manually selected parameters.

---

## 📌 Future Improvements

Possible future improvements include:

* Comparing Random Forest with SVM, Logistic Regression and Gradient Boosting
* Testing different PCA component configurations
* Applying advanced text preprocessing
* Using word and character n-grams
* Comparing TF-IDF with Word2Vec or transformer embeddings
* Hyperparameter optimization using Randomized Search or Bayesian Optimization
* Deploying the trained model through Flask/FastAPI
* Building a web interface for real-time news classification

---

## 👨‍💻 Project Type

**Machine Learning | Natural Language Processing | Text Classification | Feature Engineering | Dimensionality Reduction | Hyperparameter Optimization**

---

## ⭐ Conclusion

This project demonstrates a complete Machine Learning workflow for news classification, starting from raw text and ending with an optimized Random Forest classifier.

The combination of **TF-IDF, PCA, K-Fold Cross-Validation, and Grid Search** provides a structured and reproducible approach to handling high-dimensional text classification problems.

---

## 📜 License

This project is intended for educational and research purposes.

