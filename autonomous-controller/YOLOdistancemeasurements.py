from ultralytics import YOLO
import cv2
import math
import numpy as np
import socket
import time

host = "192.168.4.1" # Set to ESP32 Access Point IP Address
port = 80

# Function to convert a bounding box to a point at its center
def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def delay(seconds):
    start_time = time.time()
    while time.time() - start_time < seconds:
        pass

# Start webcam
video_path = "REPLACE_WITH_YOUR_VIDEO_PATH"
cap = cv2.VideoCapture(video_path)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("REPLACE_WITH_YOUR_MODEL_PATH")

classNames = ["A234Alley", "Alley", "DTKOM", "SEKDEP"]

# Create a window for displaying the webcam feed
cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam', 800, 600)

# Define the real height of the object in meters
object_real_height = 0.2  # Replace with the actual real height of the object in meters

# Define the focal length and pixel size based on your camera calibration
focal_length = 1000  # Replace with the actual focal length in pixels
pixel_size = 0.0002  # Replace with the actual pixel size in meters

textTemp = None

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:

            # Draw bounding box on the webcam feed
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1-30, y1-30]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            # Calculate and display estimated distance
            object_image_height = y2 - y1  # Height of the bounding box in pixels
            pixel_size = 1.0  # Assuming you have calibrated your camera properly
            distance = (object_real_height * focal_length) / (object_image_height * pixel_size)
            cv2.putText(img, f"Distance: {distance:.2f} meters", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if classNames[cls] == 'A234Alley' and distance < 0.53:
                new_text = 'kanan'
            else :
                new_text = 'maju'
                
            if new_text != textTemp:     
               with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((host, port))
                
                    delay(0.5)  

                    arah = 'C\n'
                    s.send(arah.encode('utf-8'))
                    delay(1)

                    if new_text == 'kanan':
                        arah = 'E\n'
                    elif new_text == 'maju':
                        arah = 'B\n'

                    s.send(arah.encode('utf-8'))
                    textTemp = new_text
                        
                
            text_size, _ = cv2.getTextSize(new_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)  
            text_x = (img.shape[1] - text_size[0]) // 2  
            text_y = (img.shape[0] - text_size[1]) // 2  
            cv2.putText(img, new_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4) 
                
            
    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()

