import cv2
import math
import winsound  # Import the sound library
from ultralytics import YOLO

# Load the model
model = YOLO("runs/detect/train/weights/best.pt")

classNames = ['With Helmet', 'Without Helmet']

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    if not success:
        break

    # Run detection
    results = model(img, stream=True)
    
    # Reset unsafe flag for this frame
    is_unsafe = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Filter: Only confident detections
            if conf > 0.5:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                if cls < len(classNames):
                    currentClass = classNames[cls]
                else:
                    currentClass = "Unknown"

                if currentClass == 'With Helmet':
                    myColor = (0, 255, 0) # Green
                    label = "SAFE"
                else:
                    myColor = (0, 0, 255) # Red
                    label = "UNSAFE"
                    is_unsafe = True  # <--- Mark as unsafe

                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                label_text = f'{currentClass} {int(conf*100)}%'
                cv2.putText(img, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, myColor, 2)

    # --- THE TIIIIK TIIIIK SOUND ---
    if is_unsafe:
        # Frequency = 2500 (High pitch "Tiik")
        # Duration = 50 (Very short, 50 milliseconds)
        winsound.Beep(2500, 50) 

    cv2.imshow("Smart Helmet Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()