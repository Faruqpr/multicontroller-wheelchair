import socket
import time
import json
import random

import datetime

host = "192.168.4.1" # Set to ESP32 Access Point IP Address
port = 80

# Create a socket connection
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    # Connect to the ESP32 server
    s.connect((host, port))
    
    i = 0
    
    while True:
        
        if i < 31:
            # Get keyboard input for two values
            arah = random.choice('ABCDE')
            kecepatan = random.randint(0,255)
        
            # Create a JSON object
            json_data = {"arah": arah, "kecepatan": kecepatan}
            date = datetime.datetime.now()
        

            # Serialize the JSON data
            json_string = json.dumps(json_data)
        
            s.send(json_string.encode('utf-8'))
            print(f"{date} -> {json_data}")
        
            i = i + 1
            time.sleep(1.5)
        
            if i == 31:
                break
        
s.close()
