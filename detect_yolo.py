from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Load a test image
image_path = "D:/Notes/Degree/MITWPU HACKATHON/Smart Tracking Violation Detection System/Dataset/Helmet Detection Dataset/images/test/BikesHelmets100.png"
img = cv2.imread(image_path)

# Run detection
results = model.predict(source=image_path, show=True)  # Show results
