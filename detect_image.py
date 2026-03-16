# USAGE
# python detect_image.py --image test.jpg

from ultralytics import YOLO
import argparse
import cv2

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", type=str, default="runs/detect/train/weights/best.pt",
                help="path to helmet detection model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum detection confidence")
args = vars(ap.parse_args())

# load model
print("[INFO] loading helmet detection model...")
model = YOLO(args["model"])

# load image
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

# detect objects
print("[INFO] running helmet detection...")
results = model(image)

results = model(image, conf=args["confidence"])

for r in results:
    for box in r.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        if class_name == "helmet":
            label = f"Helmet: Allowed {confidence*100:.2f}%"
            color = (0,255,0)

        elif class_name == "no_helmet":
            label = f"No Helmet: Not Allowed {confidence*100:.2f}%"
            color = (0,0,255)

        else:
            continue

        cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        cv2.putText(image,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

# show output
cv2.imshow("Helmet Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()