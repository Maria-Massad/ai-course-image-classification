# AI Course Image Classification

## Project Overview
This project is an Artificial Intelligence course project that focuses on comparing different machine learning models for image classification.

The implemented models include:
- Naive Bayes Classifier
- Decision Tree Classifier (with hyperparameter tuning)
- Feedforward Neural Network (MLP)

The goal is to analyze and compare the performance of classical machine learning approaches and neural networks on a multi-class image dataset.

---

## Techniques Used
- Image preprocessing (resizing, normalization, flattening)
- Dimensionality reduction using Principal Component Analysis (PCA)
- Supervised learning for image classification
- Model optimization using GridSearchCV
- Performance evaluation using accuracy, precision, recall, F1-score, and confusion matrices

---

## Dataset
- Images organized into class-based folders
- Example classes: Apple, Banana, Orange
- Images resized to 64×64 pixels
- Pixel values normalized to the range [0, 1]

---

## Tools & Libraries
- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow (Keras utilities)
- Matplotlib
- Seaborn

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
