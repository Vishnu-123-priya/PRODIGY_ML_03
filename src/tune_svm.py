import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_images


MAX_SAMPLES = 8000


print("Loading HOG features...")

X, y = load_images()

print(f"Total images: {len(X)}")



if len(X) > MAX_SAMPLES:

    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=MAX_SAMPLES,
        stratify=y,
        random_state=42
    )



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# SVM configurations
C_values = [1, 10, 100]

results = []


for C_value in C_values:

    print("\n-----------------------------")
    print(f"Training SVM with C={C_value}")
    print("-----------------------------")

    model = SVC(
        kernel="rbf",
        C=C_value,
        gamma="scale"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append(
        (C_value, accuracy)
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")



print("\n================================")
print("SVM HYPERPARAMETER COMPARISON")
print("================================")

for C_value, accuracy in results:

    print(
        f"C={C_value:<3} "
        f"Accuracy={accuracy * 100:.2f}%"
    )



best_C, best_accuracy = max(
    results,
    key=lambda item: item[1]
)


print("\nBest measured configuration:")
print(f"C = {best_C}")
print(f"Accuracy = {best_accuracy * 100:.2f}%")