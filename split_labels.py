import os
import shutil

# Paths
dataset_path = "D:/Notes/Degree/MITWPU HACKATHON/Smart Tracking Violation Detection System/Dataset/Helmet Detection Dataset/images"
label_path = "D:/Notes/Degree/MITWPU HACKATHON/Smart Tracking Violation Detection System/Dataset/Helmet Detection Dataset/labels"

train_img_path = os.path.join(dataset_path, "train")
val_img_path = os.path.join(dataset_path, "val")
test_img_path = os.path.join(dataset_path, "test")

train_label_path = os.path.join(label_path, "train")
val_label_path = os.path.join(label_path, "val")
test_label_path = os.path.join(label_path, "test")

# Ensure label folders exist
for folder in [train_label_path, val_label_path, test_label_path]:
    os.makedirs(folder, exist_ok=True)

# Move labels based on corresponding images
for split, img_folder, label_folder in zip(["train", "val", "test"], 
                                            [train_img_path, val_img_path, test_img_path], 
                                            [train_label_path, val_label_path, test_label_path]):
    for img_file in os.listdir(img_folder):
        label_file = img_file.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
        src_label = os.path.join(label_path, label_file)

        if os.path.exists(src_label):
            shutil.move(src_label, os.path.join(label_folder, label_file))

print(" Labels successfully moved to train, val, and test folders!")
