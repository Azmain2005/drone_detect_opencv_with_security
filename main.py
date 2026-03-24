import cv2
import torch
from PIL import Image
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

# Set video source (webcam or video file)
cap = cv2.VideoCapture(0)

# Define the classes you want to detect
classes = ['Drone']

# Flag to ensure sound plays only once per detection
alert_played = False

# Function to play sound in a separate thread


while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera found, switching to drone.mp4")
        cap = cv2.VideoCapture('drone.mp4')
        ret, frame = cap.read()
        if not ret:
            print("Failed to read drone.mp4")
            break

    # Convert frame to PIL Image
    img = Image.fromarray(frame[..., ::-1])

    # Run inference
    results = model(img, size=720)

    drone_detected = False

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > 0.5 and classes[int(cls)] in classes:
            drone_detected = True

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Show confidence
            text_conf = "{:.2f}%".format(conf * 100)
            cv2.putText(frame, text_conf, (int(x1), int(y1) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Show coordinates
            text_coords = "({}, {})".format(int((x1 + x2) / 2), int(y2))
            cv2.putText(frame, text_coords, (int(x1), int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Label the object
            cv2.putText(frame, "Drone", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Play sound only once when a drone is detected
    if drone_detected and not alert_played:
        alert_played = True

    # Reset the alert flag if no drone is detected
    if not drone_detected:
        alert_played = False

    # Display frame
    cv2.imshow('Drone Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()