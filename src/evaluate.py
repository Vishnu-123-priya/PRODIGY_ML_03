import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix
)
MODEL_PATH = "models/svm_cat_dog_model.pkl"
TEST_DATA_PATH = "models/test_data.pkl"
RESULTS_DIR = "results"
print("Loading trained SVM model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")
print("\nLoading test data...")
test_data = joblib.load(TEST_DATA_PATH)
X_test = test_data["X_test"]
y_test = test_data["y_test"]
print(f"Testing samples:{len(X_test)}")
print("\nGenerating predictions...")
y_pred = model.predict(X_test)
print("Predicctions completed.")
accuracy = accuracy_score(y_test,y_pred)
print("\n========================")
print("SVM MODEL EVALUATION")
print("=========================")
print(f"\nAccuracy:{accuracy:.4f}")
print(f"Accuracy:{accuracy*100:.2f}%")
report = classification_report(
    y_test,
    y_pred,
    target_names=["Cat","Dog"]
)
print("\nclassification Report:")
print(report)
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
os.makedirs(RESULTS_DIR,exist_ok=True)
plt.figure(figsize=(7,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Cat","Dog"],
    yticklabels=["Cat","Dog"]

)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("SVM Cat vs Dog confusion Matrix")
plt.tight_layout()
plot_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)
plt.savefig(plot_path,dpi=300)
plt.close()
print(f"\nconfusion matrix saved to:")
print(plot_path)
metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.txt"
)
with open(metrics_path,"w") as file:
    file.write("SVM Cat vs Dog Image Classification\n")
    file.write("=========================\n\n")
    file.write(f"Testing Samples:{len(X_test)}\n")
    file.write("Image Size: 64x64\n")
    file.write("Image Type: Grayscale\n")
    file.write("Features: HOG\n")
    file.write("HOG Orientations:9\n")
    file.write("Pixels Per cell:(8,8)\n")
    file.write("Cells Per Block: (2,2)\n")
    file.write("Block Normalization: L2-Hys\n")
    file.write(f"Feature Count:{X_test.shape[1]}\n")
    file.write("Kernel: RBF\n")
    file.write("C: 10\n")
    file.write("Gamma: scale\n\n")
    file.write(f"Accuracy: {accuracy:.4f}")
    file.write(f"Accuracy Percentage: {accuracy*100:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))
print("Metrics saved to:")
print(metrics_path)
print("\nEvaluation completed successfully.")