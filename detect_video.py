# USAGE
# python detect_video.py --video construction_video.mp4

from ultralytics import YOLO
import argparse
import cv2
import os

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video")
ap.add_argument("-m", "--model", type=str, default="runs/detect/train/weights/best.pt",
                help="path to helmet detection model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum detection confidence")
args = vars(ap.parse_args())

# load model
print("[INFO] loading helmet detection model...")
model = YOLO(args["model"])

# create outputs folder if it doesn't exist
output_folder = "outputs_video"
os.makedirs(output_folder, exist_ok=True)

# open video file
cap = cv2.VideoCapture(args["video"])
video_name = os.path.basename(args["video"])
output_path = os.path.join(output_folder, f"output_{video_name}")

# get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("[INFO] running helmet detection on video...")

# optional: set a window size multiplier for better visibility
scale = 0.3  # increase to make video bigger
window_width = int(width * scale)
window_height = int(height * scale)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=args["confidence"])

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            if class_name == "helmet":
                label = f"Helmet: Allowed {confidence*100:.2f}%"
                color = (0, 255, 0)
            elif class_name == "head":
                label = f"No Helmet: Not Allowed {confidence*100:.2f}%"
                color = (0, 0, 255)
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # write the annotated frame to output video
    out.write(frame)

    # resize frame for display
    resized_frame = cv2.resize(frame, (window_width, window_height))
    cv2.imshow("Helmet Detection", resized_frame)

    # quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Output video saved to {output_path}")