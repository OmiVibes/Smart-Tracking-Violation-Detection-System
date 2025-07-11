import cv2
import numpy as np
from ultralytics import YOLO

# ✅ Load trained YOLO model (same for both helmet & people detection)
model = YOLO("yolo_models/helmet_yolov8_best.pt")  # ✅ Ensure correct path

# ✅ Define class labels
class_names = ["With Helmet", "Without Helmet", "Person"]

def detect_helmet(image_path):
    """Detects helmets and classifies violations."""
    image = cv2.imread(image_path)
    results = model(image, conf=0.5, iou=0.5)

    helmet_detections = []
    
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(cls.item())

            # ✅ Only detect "With Helmet" or "Without Helmet"
            if class_id in [0, 1]:
                label = f"{class_names[class_id]} {conf:.2f}"
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for helmet, Red for no helmet

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                helmet_detections.append((class_names[class_id], conf))

    # ✅ Save processed image
    processed_image_path = image_path.replace("uploads", "results")
    cv2.imwrite(processed_image_path, image)

    return processed_image_path, helmet_detections

def detect_people(image_path):  # ✅ Correct function name
    """Detects people in an image, draws bounding boxes, and checks for violations."""
    image = cv2.imread(image_path)
    results = model(image, conf=0.5, iou=0.5)

    person_count = 0  # Count number of people detected

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(cls.item())

            # ✅ Only detect "Person" class
            if class_id == 2:
                person_count += 1
                color = (255, 0, 0)  # Blue for people
                label = f"Person {person_count}"
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 🚦 Classify Number of Riders
    if person_count == 2:
        message = "✅ Two people detected."
    elif person_count == 3:
        message = "⚠️ Three people detected! Violation!"
    elif person_count > 3:
        message = "🚨 More than three people detected! Major Violation!"
    else:
        message = "❌ No valid detection."

    # ✅ Save processed image
    processed_image_path = image_path.replace("uploads", "results")
    cv2.imwrite(processed_image_path, image)

    return processed_image_path, message
