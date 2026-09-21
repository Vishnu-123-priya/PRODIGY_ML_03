import os
import numpy as np

from PIL import Image
from skimage.feature import hog


DATA_DIR = "./data/PetImages"
IMG_SIZE = 64


def load_images():

    images = []
    labels = []

    categories = {
        "Cat": 0,
        "Dog": 1
    }

    for category, label in categories.items():

        folder = os.path.join(DATA_DIR, category)

        print(f"Loading {category} images ...")

        for filename in os.listdir(folder):

            filepath = os.path.join(folder, filename)

            try:
                image = Image.open(filepath)

                # Convert to grayscale
                image = image.convert("L")

                # Resize
                image = image.resize((IMG_SIZE, IMG_SIZE))

                # Convert to NumPy array
                image_array = np.array(
                    image,
                    dtype=np.float32
                ) / 255.0

                # Extract HOG features
                features = hog(
                    image_array,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys"
                )

                images.append(features)
                labels.append(label)

            except Exception:
                continue

    X = np.array(images)
    y = np.array(labels)

    print("\nImage loading completed")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)

    return X, y


if __name__ == "__main__":

    X, y = load_images()

    print("\nFirst image feature values:")
    print(X[0][:10])

    print("\nFirst image label:")
    print(y[0])