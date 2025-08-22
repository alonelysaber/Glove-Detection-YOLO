# Gloved vs Ungloved Hand Detection

A safety compliance system that detects whether workers are wearing gloves using YOLOv8. This system can be deployed on video streams or snapshots from factory cameras to ensure workplace safety compliance.

## Project Overview

This project implements a computer vision solution to detect two classes:
- `gloved_hand` - Hands wearing protective gloves
- `bare_hand` - Unprotected hands without gloves

The system processes images and outputs both annotated images and structured JSON logs with detection results, making it suitable for automated safety monitoring systems.

## Dataset

**Source**: Roboflow Universe - "Gloves and bare hands detection.v1i.yolov8" 
**Link** : https://universe.roboflow.com/moksha-me3nv/gloves-and-bare-hands-detection-pxk9g/dataset/1
**Format**: YOLO v8 format  
**Classes**: 2 (gloved_hand, bare_hand)  
**Split**: 80% Train / 10% Validation / 10% Test  

The dataset is organized in standard YOLO format with separate folders for images and labels:
```
Gloves and bare hands detection.v1i.yolov8/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Model Architecture

**Model**: YOLOv8 Nano (yolov8n.pt)  
**Framework**: Ultralytics YOLOv8  
**Input Size**: 640x640 pixels  
**Base Weights**: Pre-trained COCO weights for faster convergence  

### Training Configuration
- **Epochs**: 50
- **Batch Size**: 16
- **Image Size**: 640x640
- **Workers**: 4
- **Optimizer**: Default YOLOv8 (AdamW)

## Installation

### Requirements
```bash
pip install ultralytics
pip install opencv-python
```

### Dependencies
- Python 3.8+
- PyTorch (automatically installed with ultralytics)
- OpenCV
- Ultralytics YOLOv8

## Usage

### Training the Model

1. **Prepare your dataset** in YOLO format with the correct directory structure
2. **Update the data path** in `train.py` to point to your `data.yaml` file
3. **Run training**:
```bash
python train.py
```

The trained model will be saved to `runs/detect/train/weights/best.pt` 

### Running Detection

#### Basic Usage (Default Settings)
```bash
python detection_script.py
```
This uses the default settings:
- Input: `Gloves and bare hands detection.v1i.yolov8/test/images/`
- Output: `output/` folder
- Logs: `logs/` folder  
- Model: `runs/detect/train/weights/best.pt`
- Confidence: 0.25

#### Advanced Usage with CLI Arguments
```bash
# Custom input and output folders
python detection_script.py --input my_test_images --output my_results

# Set confidence threshold (filter low-confidence detections)
python detection_script.py --confidence 0.8

# Use different trained model
python detection_script.py --model runs/detect/train5/weights/best.pt

# Custom logs folder
python detection_script.py --logs custom_logs

# Combine multiple options
python detection_script.py --input new_images --output results --confidence 0.7 --logs detection_logs
```

#### Available CLI Arguments
- `--input`: Path to folder containing input images (default: test images folder)
- `--output`: Path for saving annotated images (default folder : `output`)
- `--logs`: Path for saving JSON logs (default: `logs`)  
- `--model`: Path to trained YOLO model (default: `runs/detect/train/weights/best.pt`)
- `--confidence`: Confidence threshold 0.0-1.0 (default: 0.25)

### Output Format

The script generates two types of outputs:

#### 1. Annotated Images
- Saved to `output/` folder
- Original images with bounding boxes and labels drawn
- Same filename as input images

#### 2. JSON Detection Logs
- Saved to `logs/` folder
- One JSON file per processed image
- Format:
```json
{
  "filename": "image1.jpg",
  "detections": [
    {
      "label": "gloved_hand",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    },
    {
      "label": "bare_hand",
      "confidence": 0.85,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

## Project Structure

```
Part_1_Glove_Detection/
├── detection_script.py          # Main inference script
├── train.py                     # Model training script  
├── output/                      # Annotated output images
├── logs/                        # JSON detection logs
├── runs/                        # YOLOv8 training outputs
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt      # Trained model weights
├── Gloves and bare hands detection.v1i.yolov8/  # Dataset
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
└── README.md
```

## Model Performance

The model was trained for 50 epochs using YOLOv8 nano architecture. Training results and metrics can be found in the `runs/detect/train/` directory, including:
- Training/validation loss curves
- Precision/Recall metrics
- Confusion matrix
- Sample predictions

## Technical Implementation

### Key Features
- **CLI Arguments**: Flexible command-line interface for different input/output paths and settings
- **Confidence Filtering**: Adjustable confidence threshold to filter low-quality detections
- **Batch Processing**: Processes all images in the specified directory automatically
- **Flexible Output**: Generates both visual and structured data outputs
- **YOLO Integration**: Uses Ultralytics YOLOv8 for state-of-the-art detection performance
- **JSON Logging**: Structured output format suitable for integration with other systems

### Detection Process
1. Parse command-line arguments (or use defaults)
2. Load trained YOLO model from specified path
3. Create output and logs directories if they don't exist
4. Iterate through all `.jpg` images in input directory
5. Run inference with specified confidence threshold
6. Extract bounding boxes, confidence scores, and class labels
7. Save annotated image with visualizations to output folder
8. Generate JSON log with structured detection data

## What Worked

 **CLI Arguments**: Implemented command-line interface for flexible usage  
 **Default Data Augmentation**: YOLO already has augmentation built in which alters the input images to improve performance during epochs
 **YOLOv8 Performance**: The model successfully distinguishes between gloved and bare hands  
 **Pre-trained Weights**: Starting with COCO weights improved convergence speed  
 **Data Format**: YOLO format provided clean, consistent annotations  
 **Output Structure**: JSON logging format meets specification requirements perfectly  
 **Automation**: Script processes entire directory without manual intervention  
 **Confidence Filtering**: Adjustable threshold helps filter out false positives  

## What Didn't Work / Challenges

 **Error Handling**: Limited error handling for corrupted images or missing files  
 **Batch Inference**: Processes images one-by-one rather than in batches for better GPU utilization  
 **File Format Support**: Currently only supports .jpg files
 **Class Remapping**: The original dataset contains images divided into three classes, which I had to remap into two and had to make edits to the YAML file, for reference compare original and latest YAML.
 **Rotation Issue**: Some images in the dataset were rotated while labelling, which resulted in wrong annotation when picking up data from the labels files, manual rotation one by one was the tedious but only option.  

## Future Improvements

- Implement batch inference for better GPU performance
- Add support for additional image formats (PNG, BMP, etc.)
- Add comprehensive error handling and logging
- Add multiprocessing support for faster processing
- Implement video stream processing capability
- Add email/SMS alerts for safety violations
- Create web dashboard for monitoring results

## Troubleshooting

**Model not found**: Check the model path using `--model` argument or ensure default path exists  
**Empty detections**: Try lowering confidence threshold with `--confidence 0.1`  
**Input folder not found**: Verify the input path exists or use `--input` to specify correct path  
**Permission errors**: Ensure you have write permissions for output and logs directories  
**Memory issues**: Reduce batch size in training or ensure sufficient RAM for image processing  
**No images processed**: Ensure input folder contains .jpg files (case-sensitive)



