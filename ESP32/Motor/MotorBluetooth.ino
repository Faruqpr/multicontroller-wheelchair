#include <BluetoothSerial.h>

#include <Arduino.h>

//Motor Kiri
#define pwmpin1 5
#define dir1 18
#define dir2 19

//Motor kanan
#define pwmpin2 25
#define dir3 32
#define dir4 33

//STATE Motor
int stdir[4];

#define pwmChannel1 0
#define pwmChannel2 1
#define freq 15000
#define res 8

int PWM1_DutyCycle = 0;
int maxspeed = 0;
int turnspeed = 0;

BluetoothSerial SerialBT;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32_B300");

  pinMode(dir1, OUTPUT);
  pinMode(dir2, OUTPUT);
  pinMode(dir3, OUTPUT);
  pinMode(dir4, OUTPUT);

  ledcSetup(pwmChannel1, freq, res);
  ledcSetup(pwmChannel2, freq, res);

  ledcAttachPin(pwmpin1, pwmChannel1);
  ledcAttachPin(pwmpin2, pwmChannel2);

  Serial.begin(115200);

}

void loop() {
  if (SerialBT.available()) {
    String receivedData = SerialBT.readStringUntil('\n');

    String arah = receivedData.substring(0, receivedData.indexOf(','));
    String kecepatan = receivedData.substring(receivedData.indexOf(',') + 1);

    maxspeed = kecepatan.toInt();
    turnspeed = maxspeed / 2;

    Serial.print("Arah : ");
    Serial.println(arah);
    Serial.print("Kecepatan : ");
    Serial.println(kecepatan);

    if (arah == "A") {
      while (PWM1_DutyCycle <= turnspeed) {
        stdir[0] = LOW;
        stdir[1] = LOW;
        stdir[2] = HIGH;
        stdir[3] = LOW;

        digitalWrite(dir1, stdir[0]);
        digitalWrite(dir2, stdir[1]);
        digitalWrite(dir3, stdir[2]);
        digitalWrite(dir4, stdir[3]);
        ledcWrite(pwmChannel1, PWM1_DutyCycle++);
        ledcWrite(pwmChannel2, PWM1_DutyCycle++);
        delay(10);
      }

    } else if (arah == "B") {
      while (PWM1_DutyCycle <= maxspeed) {
        stdir[0] = HIGH;
        stdir[1] = LOW;
        stdir[2] = HIGH;
        stdir[3] = LOW;

        digitalWrite(dir1, stdir[0]);
        digitalWrite(dir2, stdir[1]);
        digitalWrite(dir3, stdir[2]);
        digitalWrite(dir4, stdir[3]);
        ledcWrite(pwmChannel1, PWM1_DutyCycle++);
        ledcWrite(pwmChannel2, PWM1_DutyCycle++);
        delay(10);
      }

    } else if (arah == "C") {
      while (PWM1_DutyCycle >= 0) {

        digitalWrite(dir1, stdir[0]);
        digitalWrite(dir2, stdir[1]);
        digitalWrite(dir3, stdir[2]);
        digitalWrite(dir4, stdir[3]);
        ledcWrite(pwmChannel1, PWM1_DutyCycle--);
        ledcWrite(pwmChannel2, PWM1_DutyCycle--);
        delay(10);
      }
    } else if (arah == "D") {
      while (PWM1_DutyCycle <= turnspeed) {
        stdir[0] = LOW;
        stdir[1] = HIGH;
        stdir[2] = LOW;
        stdir[3] = HIGH;

        digitalWrite(dir1, stdir[0]);
        digitalWrite(dir2, stdir[1]);
        digitalWrite(dir3, stdir[2]);
        digitalWrite(dir4, stdir[3]);
        ledcWrite(pwmChannel1, PWM1_DutyCycle++);
        ledcWrite(pwmChannel2, PWM1_DutyCycle++);
        delay(10);
      }

    } else if (arah == "E") {
      while (PWM1_DutyCycle <= turnspeed) {
        stdir[0] = HIGH;
        stdir[1] = LOW;
        stdir[2] = LOW;
        stdir[3] = LOW;

        digitalWrite(dir1, stdir[0]);
        digitalWrite(dir2, stdir[1]);
        digitalWrite(dir3, stdir[2]);
        digitalWrite(dir4, stdir[3]);
        ledcWrite(pwmChannel1, PWM1_DutyCycle++);
        ledcWrite(pwmChannel2, PWM1_DutyCycle++);
        delay(10);
      }

    }
  }
}
