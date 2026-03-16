# Real-Time Helmet Detection with Alarm

## Overview
This project implements a **real-time helmet detection system** using **YOLOv8** and OpenCV.  
It detects whether a person is wearing a helmet or not using a webcam, triggers an **alarm** if the helmet is not worn, and **saves screenshots** of violations in a date-based folder.

### Features
- Real-time helmet detection with YOLOv8.
- Alarm system for violations:
  - Plays a sound when someone is not wearing a helmet.
  - Stops immediately when the helmet is worn.
- Screenshot capture:
  - Saves images of violations inside a folder named with the current date.
  - Helps keep a log for audits or safety compliance.
- Confidence-based detection:
  - Confidence ≥ 0.7 → helmet worn (green box).
  - Confidence < 0.7 → no helmet (red box, alarm triggered).

---