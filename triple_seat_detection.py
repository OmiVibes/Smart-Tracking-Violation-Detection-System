import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import argparse

# Load trained model
model = load_model("models/triple_seat_model.h5")

# Define categories
categories = ["no_violation", "triple_seat"]
img_size = 128

def predict_image(image_path):
    image_path = os.path.abspath(image_path)
    print(f"Loading image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print("🚨 Error: Unable to load the image. Check the file path!")
        return None, None  # ✅ Return a valid tuple to avoid unpacking error

    # Resize and preprocess image
    img_resized = cv2.resize(img, (img_size, img_size))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    label = categories[np.argmax(prediction)]
    probability = np.max(prediction)

    print(f"🚦 Detected: {label} (Confidence: {probability:.2f})")

    return label, probability  # ✅ Ensure function returns valid values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image")
    args = parser.parse_args()
    predict_image(args.image)
