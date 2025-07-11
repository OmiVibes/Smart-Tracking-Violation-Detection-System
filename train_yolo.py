from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)  # Fix for Windows

    model = YOLO("yolov8n.pt")  

    model.train(
        data="Dataset/Helmet Detection Dataset/data.yaml",
        epochs=50,
        imgsz=512,  # Reduce image size to save memory
        batch=8,  # Reduce batch size for better GPU performance
        workers=0,  # Fix for Windows multiprocessing
        device="cuda:0"  # Ensure it's running on GPU
    )

    model.save("helmet_yolov8_best.pt")
    print("Model training completed and saved as helmet_yolov8_best.pt")
