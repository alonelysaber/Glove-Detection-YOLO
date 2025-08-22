from ultralytics import YOLO
import os
import json
import argparse

def main():
    # setting up CLI arguments
    parser = argparse.ArgumentParser(description='Detect gloved and bare hands in images')
    parser.add_argument('--input', 
                       default='Part_1_Glove_Detection/Gloves and bare hands detection.v1i.yolov8/test/images',
                       help='Path to folder containing input images')
    parser.add_argument('--output', 
                       default='output',
                       help='Path to folder for saving annotated images')
    parser.add_argument('--logs',
                       default='logs', 
                       help='Path to folder for saving JSON logs')
    parser.add_argument('--model',
                       default='runs/detect/train/weights/best.pt',
                       help='Path to trained YOLO model')
    parser.add_argument('--confidence',
                       type=float,
                       default=0.25,
                       help='Confidence threshold for detections (0.0-1.0)')
    
    # parsing the arguments
    args = parser.parse_args()
    
    # loading model
    model = YOLO(args.model)
    
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    
    
    for img_file in os.listdir(args.input):
        if not img_file.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(args.input, img_file)
        
        # running inference
        results = model(img_path, conf=args.confidence)
        
        # save annotated image
        results[0].plot()
        results[0].save(os.path.join(args.output, img_file))
        
        # save JSON log
        detections = []
        for box in results[0].boxes:
            cls_idx = int(box.cls[0])             # class index
            label = model.names[cls_idx]          # map index to class name
            detections.append({
                "label": label,
                "confidence": float(box.conf[0]),
                "bbox": [int(x) for x in box.xyxy[0]]  # x1, y1, x2, y2
            })
        
        log_path = os.path.join(args.logs, f"{os.path.splitext(img_file)[0]}.json")
        with open(log_path, "w") as f:
            json.dump({"filename": img_file, "detections": detections}, f, indent=4)
    
    print(f"Processing complete! Check '{args.output}' for images and '{args.logs}' for JSON logs.")

if __name__ == "__main__":
    main()
