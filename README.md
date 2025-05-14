# 🧠 KNN Digit Classifier

This project implements a digit classification system using a subset of the MNIST dataset. The goal is to accurately classify grayscale images of handwritten digits (0–9) using a k-Nearest Neighbour (k-NN) classifier with dimensionality reduction techniques.

## 📂 Project Structure

- `system.py` — Core implementation of the feature extraction and classification logic (main contribution)
- `train.py` — Runs the training process 
- `evaluate.py` — Evaluates the model on noisy and masked test data 
- `utils.py` — Handles data loading and saving 
- `image_labels.csv` — Labels for training images

## 🧠 Key Techniques

### 1. Feature Extraction (PCA)
- Used Principal Component Analysis (PCA) to reduce the original 784-dimensional image vectors to 48 components.
- Chosen based on cumulative variance analysis, balancing computational efficiency and retained information.
- Outlier clipping was applied during preprocessing to improve stability and consistency across datasets.

### 2. Classifier Design (k-NN)
- Implemented a k-NN classifier with `k=10`, using Euclidean distance for prediction.
- Added a tie-breaking mechanism to handle equal distances.
- The reduced feature space from PCA helped improve generalisation and reduced overfitting.

## 📊 Performance

| Test Set       | Accuracy |
|----------------|----------|
| Noisy Images   | 92.5%    |
| Masked Images  | 73.5%    |

- The system performs robustly on noisy data due to strong preprocessing and feature selection.
- Masked data remains more challenging due to occlusion, limiting key visual cues.

## 🛠️ Implementation Focus

The main development work is in `system.py`, which includes:
- Dimensionality reduction logic (PCA)
- Classifier implementation (k-NN)
- Preprocessing techniques for real-world robustness

## 📎 Notes

- Dataset images (including noisy/masked test sets) were excluded from this repository due to size.
- This project was completed for the Data Driven Computing module at the University of Sheffield.
