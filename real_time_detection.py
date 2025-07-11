import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/wrong_way_model.h5")
categories = ["correct_way", "wrong_way"]
img_size = 128

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess
    img = cv2.resize(frame, (img_size, img_size))
    img = np.expand_dims(img / 255.0, axis=0)

    # Predict
    prediction = model.predict(img)
    label = categories[np.argmax(prediction)]

    # Display result
    cv2.putText(frame, f"Violation: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Wrong Way Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
