import json
import bluetooth
import datetime
import random
import time

# ESP32 Bluetooth address
# esp32_address = "EC:62:60:9B:E4:92"  # Replace with your ESP32's Bluetooth address

esp32_address = "C0:49:EF:E7:BD:EA"

# Create a Bluetooth socket
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

# Connect to the ESP32
sock.connect((esp32_address, 1))

i = 0

while True:
    if i < 6:
        # Get keyboard input for two values
        arah = random.choice('ABCDE')
        kecepatan = random.randint(0,255)
    
        # Create a JSON object
        json_data = {"arah": arah, "kecepatan": kecepatan}
        print(json_data)

        # Serialize the JSON data
        json_string = json.dumps(json_data)
        
        sock.send(json_string)
        
        i = i + 1
        time.sleep(1.1)
        
        if i == 6:
            break

# Close the Bluetooth socket
sock.close()
