# Infrastructure Defect Detection using YOLOv8

## About this project

I built an end-to-end machine learning pipeline for detecting defects in infrastructure like cracks, corrosion, and structural damage. The goal was to automate visual inspection using deep learning instead of relying on manual human inspection which is slow and error prone.

I collected the dataset, annotated it manually, designed the training pipeline, trained the model from scratch, evaluated the results, and finally deployed it as a web application.

## Problem statement

Manual inspection of infrastructure like roads, bridges, and buildings is time consuming and inconsistent. I wanted to build a computer vision solution that can automatically detect and localize defects from images and video feeds in real time.

## How I built it

### Step 1 - Data collection and annotation

I gathered images of infrastructure showing various types of defects. Each image was manually annotated using LabelImg tool where I drew bounding boxes around every defect and saved the labels in YOLO format (txt files with class id and normalized coordinates).

The dataset was split into three parts:
- Training set (70%) for the model to learn from
- Validation set (20%) to monitor performance during training
- Test set (10%) to evaluate final model accuracy

### Step 2 - Model selection

I chose YOLOv8 (You Only Look Once version 8) by Ultralytics because:
- It is anchor-free which makes it faster and more accurate than older YOLO versions
- It supports real-time inference which is critical for practical deployment
- It provides pretrained weights that can be fine-tuned on custom datasets
- The architecture balances speed and accuracy well for edge deployment

I went with the YOLOv8s (small) variant as it gives a good tradeoff between detection accuracy and inference speed.

### Step 3 - Training

I configured the training pipeline with the following parameters:

Model: YOLOv8s (small)
Pretrained weights: yolov8s.pt (COCO pretrained)
Epochs: 50
Image size: 640x640
Batch size: 16
Optimizer: SGD with momentum
Learning rate: auto scheduled by Ultralytics
Augmentations: mosaic, flip, scale, hsv shifts (built into Ultralytics)
Hardware: Local GPU setup

During training I monitored box loss, classification loss, and dfl loss to make sure the model was converging properly. I also tracked mAP on the validation set after each epoch.

### Step 4 - Evaluation

After training I evaluated the model on the held-out test set. I looked at:
- mAP@50 (mean average precision at 50% IoU threshold)
- mAP@50-95 (averaged across multiple IoU thresholds)
- Precision and Recall per class
- Confusion matrix to understand misclassifications

### Step 5 - Deployment

I built an interactive web application using Streamlit that allows users to:
- Upload images and see detections with bounding boxes and confidence scores
- Upload videos and process them frame by frame
- Use their webcam for live real-time detection
- Adjust the confidence threshold to filter predictions

The inference runs on the trained model weights (best.pt) using the Ultralytics predict API with OpenCV handling image and video processing.

## Tech stack

- Python 3.12 - core programming language
- PyTorch - deep learning framework used under the hood by YOLO
- Ultralytics YOLOv8 - object detection model and training framework
- OpenCV - image and video processing
- Streamlit - web application framework for the demo
- Pillow - image handling
- NumPy - numerical operations
- LabelImg - annotation tool for labeling the dataset

## Machine learning concepts used

- Transfer learning (fine-tuning pretrained COCO weights on custom data)
- Object detection (localization + classification in a single forward pass)
- Data augmentation (mosaic, flipping, scaling to improve generalization)
- Loss functions (box regression loss, classification loss, distribution focal loss)
- Non-max suppression (filtering overlapping detections)
- Intersection over Union (IoU) for evaluating detection quality
- Train/val/test split to prevent overfitting and ensure fair evaluation

## Project structure

app.py - main Streamlit web application
best.pt - trained YOLOv8 model weights
requirements.txt - Python dependencies
demo.png - screenshot of the app
README.md - project documentation

## How to run

Clone the repo:
git clone https://github.com/tradologi/infrastructure-defect-detection.git
cd infrastructure-defect-detection

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

Open http://localhost:8501 in your browser.

## What I learned

- How to structure a dataset for YOLO training including folder hierarchy, label formatting, and data.yaml configuration
- Differences between anchor-based (YOLOv5) and anchor-free (YOLOv8) detection architectures
- How to monitor training metrics like box loss, cls loss, and mAP to diagnose underfitting or overfitting
- Importance of data quality over data quantity - clean annotations matter more than having thousands of noisy labels
- Building an end-to-end ML pipeline from data collection to deployment
- Handling real-time video inference efficiently using OpenCV frame buffering

## Future improvements

- Deploy the app on Streamlit Cloud or AWS for public access
- Train with YOLOv8m or YOLOv8l for better accuracy on small defects
- Add more defect categories and expand the dataset with harder examples
- Implement model explainability using GradCAM to visualize what the model focuses on
- Export model to ONNX or TensorRT for faster inference on edge devices
- Add a dashboard showing detection analytics like defect count per class and severity distribution

## Author

Pradeep Singh

Open to feedback and suggestions. Feel free to reach out.
