import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Dataset path
data_dir = "D:\Notes\Degree\MITWPU HACKATHON\Smart Tracking Violation Detection System\Dataset\Triple Seat Detection Dataset"
categories = ["no_violation", "triple_seat"]
img_size = 128

# Load dataset
def load_data():
    data, labels = [], []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_index = categories.index(category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                data.append(img)
                labels.append(class_index)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    return np.array(data), np.array(labels)

# Preprocess dataset
data, labels = load_data()
data = data / 255.0  # Normalize images
labels = to_categorical(labels, num_classes=len(categories))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save model
model.save("models/triple_seat_model.h5")

print("Model training complete and saved as triple_seat_model.h5")
