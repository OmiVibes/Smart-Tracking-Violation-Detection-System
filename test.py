import os

folder_path = "static/challans"
if not os.path.exists(folder_path):
    print(f"Folder Missing: {folder_path}")
else:
    print(f" Folder Exists: {folder_path}")
