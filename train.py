from ultralytics import YOLO

if __name__ == "__main__":
    # Load a YOLOv8 model 
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="Part_1_Glove_Detection/Gloves and bare hands detection.v1i.yolov8/data.yaml",  # dataset YAML
        epochs=50,                 # depending on your dataset size
        imgsz=640,                 # image size
        batch=16,                  # depending on GPU memory, I used an RTX 3070
        workers=4                  # data loading workers
    )

    # save trained model path for inference later
    model_path = "Part_1_Glove_Detection/runs/detect/train/weights/best.pt"
