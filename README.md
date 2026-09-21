# Cat vs Dog Image Classification using SVM

## 📌 Project Overview

This project implements a **Support Vector Machine (SVM)** to classify images into two categories:

* 🐱 Cat
* 🐶 Dog

The project was developed as part of the **Machine Learning Internship at Prodigy Infotech**.

Instead of using raw image pixels directly, the project uses **Histogram of Oriented Gradients (HOG)** for feature extraction. These features are then used to train an **RBF-kernel Support Vector Machine**.

---

## 🎯 Problem Statement

Build a machine learning model using Support Vector Machines to classify images of cats and dogs.

The system performs the following steps:

1. Load cat and dog images.
2. Convert images to grayscale.
3. Resize images to 64 × 64 pixels.
4. Normalize pixel values.
5. Extract HOG features.
6. Split the data into training and testing sets.
7. Train an SVM classifier.
8. Tune the SVM hyperparameter `C`.
9. Evaluate the trained model.
10. Predict the class of new images.

---

## 📂 Dataset

The project uses the **Microsoft Kaggle Cats and Dogs dataset**, which contains images of cats and dogs.

Dataset source:

**Microsoft — Kaggle Cats and Dogs Dataset**

The extracted dataset is organized as:

```text
data/
└── PetImages/
    ├── Cat/
    └── Dog/
```

The dataset contains approximately **25,000 images**.

During preprocessing, invalid or corrupted images are skipped automatically.

The preprocessing pipeline successfully loaded:

```text
Total valid images: 24,998
```

---

## 🛠️ Technologies Used

* Python 3.11
* NumPy
* Pandas
* Scikit-learn
* Scikit-image
* Pillow
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook
* Git & GitHub

---

## 🧠 Machine Learning Approach

The project uses the following pipeline:

```text
Input Image
     ↓
Grayscale Conversion
     ↓
Resize to 64 × 64
     ↓
Pixel Normalization
     ↓
HOG Feature Extraction
     ↓
SVM Classifier
     ↓
Cat / Dog Prediction
```

---

## 🔍 Image Preprocessing

Each image is processed using the following steps.

### 1. Grayscale Conversion

Color images are converted into grayscale to reduce the dimensionality of the input.

### 2. Image Resizing

Every image is resized to:

```text
64 × 64 pixels
```

### 3. Pixel Normalization

Pixel values are normalized to the range:

```text
0 to 1
```

### 4. HOG Feature Extraction

Histogram of Oriented Gradients is used to capture important image structure and edge information.

The HOG configuration is:

```text
Orientations       : 9
Pixels per Cell    : (8, 8)
Cells per Block    : (2, 2)
Block Normalization: L2-Hys
```

Each processed image produces:

```text
1,764 HOG features
```

---

## 🤖 SVM Model

The classifier used in this project is:

**Support Vector Classifier (SVC)** from Scikit-learn.

The final configuration is:

```text
Kernel : RBF
C      : 10
Gamma  : scale
```

The RBF kernel allows the SVM to model nonlinear relationships between the extracted image features.

---

## 📊 Dataset Split

To keep the training process computationally manageable, **8,000 images** were selected from the available valid images.

The data was split using an 80/20 train-test split.

```text
Total images used : 8,000

Training samples  : 6,400
Testing samples   : 1,600
```

Stratified splitting was used to maintain the class distribution between training and testing sets.

---

## ⚙️ Hyperparameter Tuning

The SVM parameter `C` was evaluated using three different values:

```text
C = 1
C = 10
C = 100
```

The results were:

| C Value |   Accuracy |
| ------: | ---------: |
|       1 |     73.75% |
|  **10** | **74.75%** |
|     100 |     74.75% |

The final configuration uses:

```text
C = 10
```

Both `C=10` and `C=100` achieved the same measured accuracy on the evaluation split, so `C=10` was selected as the final configuration.

---

## 📈 Model Performance

The final SVM model achieved:

### Accuracy

```text
74.75%
```

Evaluation was performed on:

```text
1,600 test images
```

### Classification Report

```text
              precision    recall  f1-score   support

Cat              0.73      0.78      0.75       800
Dog              0.76      0.72      0.74       800

accuracy                             0.75      1600
macro avg         0.75      0.75      0.75      1600
weighted avg      0.75      0.75      0.75      1600
```

---

## 🔢 Confusion Matrix

The resulting confusion matrix is:

```text
[[621 179]
 [225 575]]
```

Interpreted as:

| Actual | Predicted Cat | Predicted Dog |
| ------ | ------------: | ------------: |
| Cat    |           621 |           179 |
| Dog    |           225 |           575 |

Therefore:

* **621 cats** were correctly classified.
* **179 cats** were classified as dogs.
* **575 dogs** were correctly classified.
* **225 dogs** were classified as cats.

The confusion matrix visualization is saved as:

```text
results/confusion_matrix.png
```

---

## 📁 Project Structure

```text
svm_classifier/
│
├── data/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
│
├── models/
│   ├── .gitkeep
│   ├── svm_cat_dog_model.pkl
│   └── test_data.pkl
│
├── notebooks/
│   ├── .gitkeep
│   └── 01_exploratory_data_analysis.ipynb
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
├── README.md
└── requirements.txt
```

> The dataset and trained `.pkl` model files are excluded from GitHub using `.gitignore` to avoid uploading large files.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/PRODIGY_ML_03.git
```

Move into the project directory:

```bash
cd PRODIGY_ML_03
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

Download and extract the Cats and Dogs dataset.

Place the extracted images in:

```text
data/PetImages/
```

The directory should contain:

```text
data/PetImages/Cat/
data/PetImages/Dog/
```

---

## ▶️ Training the Model

From the project root:

```powershell
python src\train.py
```

The training script:

1. Loads the images.
2. Extracts HOG features.
3. Selects the required number of samples.
4. Splits the data into training and testing sets.
5. Trains the SVM.
6. Calculates validation accuracy.
7. Saves the trained model.
8. Saves the test data.

The trained model is saved as:

```text
models/svm_cat_dog_model.pkl
```

---

## 🔧 Hyperparameter Tuning

To compare different values of `C`:

```powershell
python src\tune_svm.py
```

The current experiment compares:

```text
C = 1
C = 10
C = 100
```

---

## 📊 Model Evaluation

Run:

```powershell
python src\evaluate.py
```

This generates:

```text
results/confusion_matrix.png
results/metrics.txt
```

The evaluation script calculates:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

## 🔮 Predict a New Image

The project also includes a prediction script.

Run:

```powershell
python src\predict.py ".\data\PetImages\Cat\0.jpg"
```

or:

```powershell
python src\predict.py ".\data\PetImages\Dog\0.jpg"
```

The image goes through the same preprocessing and HOG feature extraction pipeline before being passed to the trained SVM model.

Example output:

```text
Predicted class: Cat
```

or:

```text
Predicted class: Dog
```

---

## 📓 Exploratory Data Analysis

The project includes a Jupyter notebook:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

The notebook contains:

* Dataset exploration
* Class distribution
* Sample image visualization
* Image preprocessing demonstration
* HOG feature extraction
* Feature shape analysis
* Model evaluation overview

Launch Jupyter Notebook using:

```powershell
jupyter notebook
```

---

## 📋 Results Summary

| Configuration       | Value         |
| ------------------- | ------------- |
| Dataset             | Cats and Dogs |
| Valid images loaded | 24,998        |
| Images used         | 8,000         |
| Training samples    | 6,400         |
| Testing samples     | 1,600         |
| Image size          | 64 × 64       |
| Image type          | Grayscale     |
| Feature extraction  | HOG           |
| HOG feature count   | 1,764         |
| SVM kernel          | RBF           |
| C                   | 10            |
| Gamma               | scale         |
| Test accuracy       | **74.75%**    |

---

## ⚠️ Limitations

The current implementation achieves **74.75% accuracy** on the selected evaluation split.

Some limitations include:

* Only 8,000 images were used for training to keep computation manageable.
* HOG features primarily capture shape and edge information.
* HOG does not capture high-level semantic information as effectively as modern deep learning models.
* The model may have difficulty with images containing unusual poses, backgrounds, lighting conditions, or occlusions.
* The current model uses a traditional machine learning approach rather than a convolutional neural network.

---

## 🔮 Future Improvements

Possible improvements include:

* Train on a larger portion of the dataset.
* Experiment with additional SVM kernels.
* Perform more extensive hyperparameter tuning.
* Experiment with different HOG configurations.
* Apply data augmentation.
* Compare HOG + SVM with CNN-based models.
* Use transfer learning with pretrained models such as ResNet or MobileNet.
* Build a simple web interface for real-time image prediction.
* Deploy the model using Flask, FastAPI, or Streamlit.

---

## 💡 Key Learning Outcomes

Through this project, I gained practical experience in:

* Image preprocessing
* Grayscale image conversion
* Feature extraction using HOG
* Support Vector Machines
* RBF kernels
* Hyperparameter tuning
* Train-test splitting
* Classification metrics
* Confusion matrix analysis
* Model serialization using Joblib
* Building a reusable ML project structure
* Git and GitHub version control

---

## 🏆 Final Result

The final HOG-based SVM classifier achieved:

```text
74.75% test accuracy
```

using:

```text
64 × 64 grayscale images
        ↓
HOG feature extraction
        ↓
1,764 features
        ↓
RBF SVM
        ↓
C = 10
        ↓
74.75% accuracy
```

This project demonstrates a complete traditional computer-vision and machine-learning pipeline for binary image classification.

---

