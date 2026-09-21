
import os
import sys
import joblib

import numpy as np
from PIL import Image
from skimage.feature import hog




MODEL_PATH = "./models/svm_cat_dog_model.pkl"

IMG_SIZE = 64

CATEGORIES = {
    0: "Cat",
    1: "Dog"
}




print("Loading trained SVM model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")




if len(sys.argv) < 2:

    print("\nUsage:")
    print("python src/predict.py <image_path>")

    sys.exit()


image_path = sys.argv[1]




if not os.path.exists(image_path):

    print(f"\nImage not found: {image_path}")

    sys.exit()



try:

    image = Image.open(image_path)

    print(f"\nImage: {image_path}")

   
    image = image.convert("L")

    
    image = image.resize((IMG_SIZE, IMG_SIZE))

    
    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    
    features = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    
    features = features.reshape(1, -1)

    

    prediction = model.predict(features)[0]

    predicted_class = CATEGORIES[prediction]

    print("\n==============================")
    print("PREDICTION")
    print("==============================")

    print(f"Predicted Class: {predicted_class}")

except Exception as e:

    print(f"\nError processing image: {e}")

