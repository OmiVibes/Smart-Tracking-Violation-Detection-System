import cv2
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("helmet_yolov8_best.pt")

# Define class labels
class_names = ["With Helmet", "Without Helmet"]  # Ensure class 0 = Helmet, 1 = No Helmet

def detect_helmet_in_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(" Error: Could not load image.")
        return None

    # Run YOLO detection
    results = model(image, conf=0.5, iou=0.5)  # Reduced conf threshold to detect all objects

    # Process results
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(cls.item())
            confidence = conf.item()

            # Check if class_id is valid
            if class_id >= len(class_names):
                print(f"⚠ Unknown class ID {class_id}, skipping...")
                continue  # Skip invalid detections

            # Set label and color
            label = f"{class_names[class_id]} {confidence:.2f}"
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for Helmet, Red for No Helmet

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save and return processed image path
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    return output_path
