# 🪖 Real-Time Smart Helmet Detection System (YOLOv8)

## 📌 Overview
This project implements an **advanced real-time helmet detection system** using **YOLOv8** and OpenCV.

Unlike basic detection systems, this version uses **smart pairing logic** to ensure:
> ✅ Each person (head) is actually wearing a helmet  
> ❌ Not just detecting a helmet somewhere in the frame  

It uses a webcam to monitor safety compliance, triggers an **alarm for violations**, and **captures screenshots** for record-keeping.

---

## 🚀 Features

### 🔍 Smart Detection (Improved Logic)
- Detects:
  - Helmet
  - Head (no helmet)
  - Person (optional)
- Uses **bounding box overlap (IoU)** to match:
  - Helmet ↔ Head
- Ensures **helmet is worn on the head**, not just present in frame

---

### 🚨 Violation Alert System
- Triggers alarm when:
  - A head is detected **without a helmet**
- Smart control:
  - ✅ Alarm plays only once per violation
  - ✅ Stops automatically when compliance is restored

---

### 📸 Screenshot Capture
- Automatically saves violation images:
  - Folder format: `screenshots/YYYY-MM-DD/`
  - File format: `no_helmet_HH-MM-SS.jpg`
- Useful for:
  - Safety audits
  - Monitoring logs

---

### 🎯 High Accuracy Detection
- Uses **class-based detection (not confidence-based mistakes)**  
- Correct class mapping from dataset:
  - `0 → helmet`
  - `1 → vest`
  - `2 → head (no helmet)`
  - `3 → person`
- Filters detections using configurable confidence threshold

---

### ⚡ Real-Time Performance
- Optimized with YOLOv8
- Works with live webcam feed
- Fast inference (~100–150 ms per frame depending on system)

---

## 🧠 How It Works

1. Capture frame from webcam  
2. Run YOLOv8 object detection  
3. Separate detections:
   - Helmets
   - Heads  
4. For each **head**:
   - Check if any **helmet overlaps (IoU > 0.3)**  
   - If yes → ✅ Safe  
   - If no → 🚨 Violation  
5. Trigger:
   - Alarm
   - Screenshot capture  

---

## 📁 Project Structure

helmet-detection-yolov8/
│
├── data.yaml
├── train.py
├── detect_webcam.py # Smart detection script
├── alarm.wav
├── screenshots/
│ └── YYYY-MM-DD/
│
└── runs/
└── detect/


---

## ⚙️ Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- imutils
- numpy

Install dependencies:

```bash
- pip install ultralytics opencv-python imutils numpy


 ▶️ Usage

- 1. Train Model
- python train.py

- 2. Run Real-Time Detection
- python detect_webcam.py