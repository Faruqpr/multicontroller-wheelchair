from ultralytics import YOLO
import cv2
import numpy as np
import socket
import time
from rplidar import RPLidar

def delay(seconds):
    start_time = time.time()
    while time.time() - start_time < seconds:
        pass

# Setup RPLidar
PORT_NAME = 'COM6'
lidar = RPLidar(PORT_NAME)
lidar.start_motor()

# ESP32 Settings
host = "192.168.4.1"
port = 80

# Initialize YOLO and Camera
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
model = YOLO("C:/Users/User/Downloads/trt.pt")
classNames = ["A235Alley", "DTMAlley", "DTKOM", "Sekdep"]

# Initialize LIDAR Visualization
height, width = 800, 800
lidar_img = np.zeros((height, width, 3), dtype=np.uint8)

def get_lidar_distance(lidar, visualize=False):
    global lidar_img
    for scan in lidar.iter_scans():
        if visualize:
            lidar_img.fill(0)  # Clear the image for a new scan
        for quality, angle, distance in scan:
            if 265 <= angle <= 275:  # Angle range for the target direction
                if visualize:
                    # Visualization logic
                    angle_rad = np.deg2rad(angle)
                    x = int(distance * np.cos(angle_rad) * 0.1) + width // 2
                    y = int(distance * np.sin(angle_rad) * 0.1) + height // 2
                    cv2.line(lidar_img, (width // 2, height // 2), (x, y), (255, 255, 255), 1)
                    cv2.imshow("RPLIDAR", lidar_img)
                return distance  # Return distance for the first target direction match
    return None

textTemp = None

try:
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        detected_class = None
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if classNames[cls] in ['A235Alley', 'DTKOM']:
                    detected_class = classNames[cls]
                    break

        lidar_distance = get_lidar_distance(lidar, visualize=True)  # Enable visualization

        if detected_class:
            if lidar_distance is not None:
                if detected_class == 'A235Alley' and lidar_distance < 2000:
                    new_text = 'kanan'
                elif detected_class == 'DTKOM' and lidar_distance < 3.0:
                    new_text = 'kiri'
                else:
                    new_text = 'maju'

                if new_text != textTemp:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((host, port))
                        delay(0.5)
                        s.send(b'C\n')  # Stop Command 
                        time.sleep(1)
                        if new_text == 'kanan':
                            s.send(b'E\n')  # Right Command
                        elif new_text == 'kiri': 
                            s.send(b'A\n')  # Left Command
                        else:
                            s.send(b'B\n')  # Forward Command
                        textTemp = new_text

        # Display Image with YOLO detections
        cv2.imshow('Webcam', img)
        # Display LIDAR Visualization
        #cv2.imshow("RPLIDAR", lidar_img)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print('Stopping.')

# Cleanup
lidar.stop()
lidar.stop_motor()
lidar.disconnect()
cap.release()
cv2.destroyAllWindows()
