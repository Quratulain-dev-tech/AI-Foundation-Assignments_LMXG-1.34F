# 👗 Fashion-MNIST — Deep Learning Model Comparison

A Deep Learning image classification project using the **Fashion-MNIST dataset**, where different neural network architectures were trained and compared.

## 📌 What I Did

* Loaded and preprocessed **Fashion-MNIST**
* Normalized pixel values from **0–255 → 0–1**
* Trained and compared **4 Deep Learning models**
* Evaluated models using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix
* Tested the trained models on individual images

## 🧠 Models

| Model    | Architecture                            |
| -------- | --------------------------------------- |
| Dense 7  | 7-layer Fully Connected Neural Network  |
| Dense 10 | 10-layer Fully Connected Neural Network |
| CNN 3    | CNN with 3 convolutional layers         |
| CNN 5    | Deeper CNN with 5 convolutional layers  |

## 🏆 Best Result

**CNN 5 — Best Performing Model**

The deeper CNN architecture performed best among the four models, showing the advantage of convolutional networks for image classification.

## 📊 Model Comparison

The project compares all models based on their:

**Accuracy → Precision → Recall → F1-Score**

Training/validation curves and a confusion matrix were also used to analyze model performance.

## 🛠️ Tech Stack

`Python` · `TensorFlow/Keras` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `Scikit-learn`

## 📂 Dataset

**Fashion-MNIST** — 60,000 training images and 10,000 test images, with 10 clothing categories.

## 🚀 Project Focus

**Image Classification | Deep Learning | CNN vs Dense Networks | Model Comparison**

---

### 📓 Notebook

The complete implementation and results are available in the Kaggle notebook included in this repository.
