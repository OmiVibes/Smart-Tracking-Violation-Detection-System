import os
import xml.etree.ElementTree as ET

# Paths
ANNOTATIONS_DIR = "D:/Notes/Degree/MITWPU HACKATHON/Smart Tracking Violation Detection System/Dataset/Helmet Detection Dataset/annotations"
YOLO_LABELS_DIR = "D:/Notes/Degree/MITWPU HACKATHON/Smart Tracking Violation Detection System/Dataset/Helmet Detection Dataset/yolo_labels"

# Create output directory if not exists
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Class mapping
CLASS_NAMES = ["With Helmet", "Without Helmet"]

def convert_xml_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    image_name = root.find("filename").text
    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)
    
    output_path = os.path.join(YOLO_LABELS_DIR, image_name.replace(".jpg", ".txt"))
    
    with open(output_path, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in CLASS_NAMES:
                continue
            
            class_id = CLASS_NAMES.index(class_name)
            bbox = obj.find("bndbox")
            xmin, ymin, xmax, ymax = map(int, [bbox.find(tag).text for tag in ["xmin", "ymin", "xmax", "ymax"]])
            
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            w = (xmax - xmin) / img_width
            h = (ymax - ymin) / img_height
            
            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# Convert all annotation files
for file in os.listdir(ANNOTATIONS_DIR):
    if file.endswith(".xml"):
        convert_xml_to_yolo(os.path.join(ANNOTATIONS_DIR, file))

print("Annotations converted successfully!")
