import os

labels_path = "D:/Notes/Degree/MITWPU HACKATHON/Smart Tracking Violation Detection System/Dataset/Helmet Detection Dataset/labels/train"
images_path = "D:/Notes/Degree/MITWPU HACKATHON/Smart Tracking Violation Detection System/Dataset/Helmet Detection Dataset/images/train"

corrupt_files = []

for label_file in os.listdir(labels_path):
    label_path = os.path.join(labels_path, label_file)
    image_name = label_file.replace(".txt", ".png")  # Change to .jpg if using JPG files
    image_path = os.path.join(images_path, image_name)

    if not os.path.exists(image_path):
        corrupt_files.append(label_path)
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    valid_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            corrupt_files.append(label_path)
            break
        class_id, x, y, w, h = map(float, parts)
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            corrupt_files.append(label_path)
            break
        valid_lines.append(line)

    if len(valid_lines) < len(lines):
        with open(label_path, "w") as f:
            f.writelines(valid_lines)

print(" Checked labels. Corrupt labels:", len(corrupt_files))
for file in corrupt_files:
    print(f"Corrupt file: {file}")
