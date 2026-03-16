from ultralytics import YOLO
from imutils.video import VideoStream
import argparse
import time
import cv2
import imutils
import winsound
import os
from datetime import datetime

# Function to play alarm asynchronously
def sound_alarm():
    winsound.PlaySound("alarm.wav", winsound.SND_ASYNC)

# Function to save screenshot in date folder
def save_screenshot(frame):
    today = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join("screenshots", today)
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%H-%M-%S")
    file_path = os.path.join(folder_path, f"no_helmet_{timestamp}.jpg")
    cv2.imwrite(file_path, frame)
    print(f"[INFO] Screenshot saved: {file_path}")

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
                default="runs/detect/train/weights/best.pt",
                help="Path to trained helmet detection model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load YOLO model
print("[INFO] Loading helmet detection model...")
helmetNet = YOLO(args["model"])

# Start webcam
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

violation_active = False  # Tracks if alarm/screenshot is active

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    results = helmetNet(rgb_frame, conf=args["confidence"])[0]

    helmet_detected = False

    # Loop through all detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        if confidence >= 0.8:
            label = f"Helmet: Allowed {confidence*100:.2f}%"
            color = (0, 255, 0)
            helmet_detected = True
        else:
            label = f"No Helmet: Not Allowed {confidence*100:.2f}%"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Violation logic: trigger only once per violation
    if not helmet_detected:
        if not violation_active:
            sound_alarm()         # Play alarm
            save_screenshot(frame)  # Save screenshot
            violation_active = True
    else:
        if violation_active:
            winsound.PlaySound(None, winsound.SND_PURGE)  # Stop alarm
            violation_active = False

    cv2.imshow("Helmet Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()