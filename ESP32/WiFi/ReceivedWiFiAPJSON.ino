#include <WiFi.h>
#include <Arduino.h>
#include <ArduinoJson.h>

int maxspeed;

const char* ssid = "B300-Access-Point";
const char* password = "123456789";

WiFiServer server(80);

void setup(){
  Serial.begin(115200);


  // Connect to Wi-Fi network with SSID and password
  Serial.print("Setting AP (Access Point)…");
  // Remove the password parameter, if you want the AP (Access Point) to be open
  WiFi.softAP(ssid, password);
 
  IPAddress IP = WiFi.softAPIP();
  Serial.print("ESP32 AP IP Address : ");
  Serial.println(IP);

  // Start the server
  server.begin();
}

void loop(){
    // Check if a client has connected
  WiFiClient client = server.available();

  if (client) {
    //Serial.println("New client connected");
    
    // Read the data from the client
    while (client.connected()) {
      if (client.available()) {
    
        String json_data = client.readStringUntil('\n');

        DynamicJsonDocument doc(1024);
        DeserializationError error = deserializeJson(doc, json_data);

        if(error){
          Serial.println("Failed to parse JSON");
        }
        else{
          const char* arah = doc["arah"];
          int kecepatan = doc["kecepatan"];

          Serial.print("Received Arah : ");
          Serial.print(arah);
          Serial.print("   Received Kecepatan : ");
          Serial.println(kecepatan);
        }
      }
    }

    client.stop();

  }
}
