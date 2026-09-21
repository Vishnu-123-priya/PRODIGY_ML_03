
import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Allow importing preprocess.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_images




MODEL_PATH = "./models/svm_cat_dog_model.pkl"
TEST_DATA_PATH = "./models/test_data.pkl"

MAX_SAMPLES = 8000


BEST_C = 10




print("Loading images...")

X, y = load_images()

print(f"\nTotal images loaded: {len(X)}")




if len(X) > MAX_SAMPLES:

    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=MAX_SAMPLES,
        stratify=y,
        random_state=42
    )

print(f"Images used for training: {len(X)}")




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))





print("\nTraining final SVM...")

model = SVC(
    kernel="rbf",
    C=BEST_C,
    gamma="scale"
)




model.fit(X_train, y_train)

print("Final SVM training completed.")




y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("FINAL MODEL RESULT")
print("================================")

print(f"Kernel: RBF")
print(f"C: {BEST_C}")
print(f"Gamma: scale")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")




os.makedirs("./models", exist_ok=True)




joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)




test_data = {
    "X_test": X_test,
    "y_test": y_test
}

joblib.dump(test_data, TEST_DATA_PATH)

print("\nTest data saved successfully:")
print(TEST_DATA_PATH)

