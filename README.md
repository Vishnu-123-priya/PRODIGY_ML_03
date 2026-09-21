# Cat vs Dog Image Classification using SVM

## Project Overview

This project implements a **Support Vector Machine (SVM)** based image classification system to classify images into two categories:

* Cat
* Dog

The project uses image preprocessing and **Histogram of Oriented Gradients (HOG)** feature extraction before training an SVM classifier.

The objective is to build a traditional machine-learning computer vision pipeline for binary image classification.

---

## Problem Statement

Implement a Support Vector Machine (SVM) to classify images of cats and dogs.

The system takes an image as input, extracts relevant visual features, and predicts whether the image belongs to the **Cat** or **Dog** class.

---

## Dataset

The project uses the **Kaggle Cats and Dogs** image dataset.

The dataset contains two classes:

```text
PetImages/
├── Cat/
└── Dog/
```

Approximately 25,000 images are available in the dataset.

For the initial model development and experimentation, a subset of **8,000 images** is used.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Scikit-image
* Pillow
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

---

## Machine Learning Pipeline

```text
Input Images
     ↓
Image Loading
     ↓
Grayscale Conversion
     ↓
Image Resizing
     ↓
HOG Feature Extraction
     ↓
Train-Test Split
     ↓
SVM Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Cat / Dog Prediction
```

---

## Image Preprocessing

Each image is processed using the following steps:

### 1. Grayscale Conversion

Images are converted from RGB to grayscale.

### 2. Image Resizing

Images are resized to:

```text
64 × 64 pixels
```

This provides a consistent input size.

### 3. Pixel Normalization

Pixel values are normalized to the range:

```text
0 to 1
```

### 4. HOG Feature Extraction

Histogram of Oriented Gradients (HOG) features are extracted to represent image edges and local shape information.

The HOG configuration used is:

```text
Orientations: 9
Pixels per cell: 8 × 8
Cells per block: 2 × 2
Block normalization: L2-Hys
```

---

## SVM Model

The classifier uses the Support Vector Machine implementation from Scikit-learn.

Model configuration:

```text
Algorithm: Support Vector Machine
Kernel: RBF
C: 10
Gamma: scale
```

The value of `C` can be updated based on the result of the hyperparameter tuning experiment.

---

## Dataset Split

For the current development experiment:

```text
Total selected images: 8,000
Training images: 6,400
Testing images: 1,600
```

The dataset is split using:

```text
80% Training
20% Testing
```

Stratified splitting is used to preserve the Cat/Dog class distribution.

---

## Hyperparameter Tuning

The project experiments with different SVM `C` values:

```text
C = 1
C = 10
C = 100
```

The configuration producing the highest measured validation accuracy is selected for the final model.

---

## Model Evaluation

The classifier is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The evaluation results are stored in:

```text
results/metrics.txt
results/confusion_matrix.png
```

### Final Performance

Update this section with the actual results from your final evaluation.

```text
Accuracy: XX.XX%
Precision: XX.XX%
Recall: XX.XX%
F1-score: XX.XX%
```

> Performance values should be updated after the final experiment and should reflect the actual measured results.

---

## Project Structure

```text
svm_classifier/
│
├── data/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
│
├── models/
│   ├── svm_cat_dog_model.pkl
│   └── test_data.pkl
│
├── notebooks/
│
├── results/
│   ├── confusion_matrix.png
│   └── metrics.txt
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── tune_svm.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run:

```bash
python src/train.py
```

The trained SVM model will be saved as:

```text
models/svm_cat_dog_model.pkl
```

---

## Evaluating the Model

Run:

```bash
python src/evaluate.py
```

This generates:

```text
results/metrics.txt
results/confusion_matrix.png
```

---

## Hyperparameter Tuning

Run:

```bash
python src/tune_svm.py
```

The script compares different values of the SVM `C` parameter.

---

## Making Predictions

The trained model can be used to classify a new image.

Example:

```bash
python src/predict.py "path/to/image.jpg"
```

Example output:

```text
==============================
PREDICTION
==============================

Predicted Class: Cat
```

---

## Key Learning Outcomes

Through this project, the following concepts were implemented:

* Image preprocessing
* Grayscale image conversion
* Image resizing
* HOG feature extraction
* Feature representation for machine learning
* Support Vector Machines
* RBF kernel
* Hyperparameter tuning
* Train-test splitting
* Binary image classification
* Model evaluation
* Confusion matrix visualization
* Model serialization using Joblib
* Command-line image prediction

---

## Future Improvements

Possible improvements include:

* Training on a larger portion of the dataset
* Comparing Linear SVM and RBF SVM
* Experimenting with different image sizes
* Additional HOG configurations
* Cross-validation
* Feature scaling and dimensionality reduction
* Comparing SVM with CNN-based approaches
* Building a simple web interface for image prediction



