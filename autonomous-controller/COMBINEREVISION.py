from ultralytics import YOLO
import cv2
import numpy as np
import socket
import time
from rplidar import RPLidar
import datetime

def delay(seconds):
    start_time = time.time()
    while time.time() - start_time < seconds:
        pass

# Configuration and initialization
PORT_NAME = 'COM6'
VIDEO_PATH = "C:/Users/User/Pictures/Camera Roll/DATARevise.mp4"
MODEL_PATH = "C:/Users/User/Downloads/ModelRevisi.pt"
CLASS_NAMES = ["A235", "DTM", "Lab", "Sekdep"]
HOST = "192.168.4.1"
PORT = 80

# Setup lidar and camera
lidar = RPLidar(PORT_NAME)
lidar.start_motor()
cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL_PATH)

def control_wheelchair(command, socket):
    commands = {'stop': b'C\n', 'right': b'E\n', 'left': b'A\n', 'forward': b'B\n'}
    
    # Send the 'stop' command before any directional command
    if command != 'forward':
        print("Sending stop command to prepare for direction change...")
        socket.send(commands['stop'])
        delay(0.5)  # Wait for 0.5 seconds to ensure the command is executed
    
    socket.send(commands[command])
    print(f"Command sent: {command}")
    delay(0.5)  # Wait for 0.5 seconds to ensure the command is executed

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        
        # Send a stop command at the start
        control_wheelchair('stop', s)
        
        # Main loop
        for scan in lidar.iter_scans():
            success, img = cap.read()
            if not success:
                print("Failed to read from camera")
                break
        
            results = model(img, stream=True)
            detected_class = None
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if CLASS_NAMES[cls] in ['A235', 'Sekdep']:
                        detected_class = CLASS_NAMES[cls]
                        print(f"Detected class: {detected_class}")
                        break
                if detected_class:
                    break
        
            # The following code should be at this indentation level, not inside the 'for r in results:' loop.
            distances = np.zeros((361))
            for quality, angle, distance in scan:
                nangle = int(np.fix(angle))
                if 265 <= nangle <= 275:
                    print(f"Angle: {nangle}, Distance: {distance}")  # Debug output for each angle
                distances[nangle] = distance
            
            # Filter out zero distances
            valid_distances = distances[265:276][distances[265:276] > 0]
            print(f"Valid distances within range: {valid_distances}")  # Debug output for the range of interest
            target_distance = np.min(valid_distances) if valid_distances.size > 0 else float('inf')
            
            print(f"Target distance: {target_distance}")
            
            if detected_class == 'A235' and target_distance < 200:
                control_wheelchair('right', s)
            elif detected_class == 'Sekdep' and target_distance < 300:
                control_wheelchair('left', s)
            else:
                control_wheelchair('forward', s)
        
            if cv2.waitKey(1) == ord('q'):
                break
            # ... (rest of your loop)

except KeyboardInterrupt:
    print('Stopping.')

finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    cap.release()
    cv2.destroyAllWindows()
