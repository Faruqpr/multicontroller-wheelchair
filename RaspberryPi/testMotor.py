import RPi.GPIO as GPIO
from gpiozero import PWMOutputDevice
from time import sleep

# Motor Kiri
dir1 = 18
dir2 = 23
pwmpin1 = 24

# Motor Kanan
dir3 = 25
dir4 = 12
pwmpin2 = 16

maxspeed = 100  # Skala 0-100 untuk PWM gpiozero
PWM1_DutyCycle = 0

# Setup untuk GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup([dir1, dir2, dir3, dir4], GPIO.OUT, initial=GPIO.LOW)

# Setup untuk PWM
motor1_pwm = PWMOutputDevice(pwmpin1, True, 0, 1000)
motor2_pwm = PWMOutputDevice(pwmpin2, True, 0, 1000)

def soft_start():
    """ Fungsi untuk memulai gerakan motor secara bertahap """
    global PWM1_DutyCycle
    while PWM1_DutyCycle < maxspeed:
        GPIO.output([dir1, dir3], GPIO.HIGH)
        GPIO.output([dir2, dir4], GPIO.LOW)
        motor1_pwm.value = PWM1_DutyCycle / 100
        motor2_pwm.value = PWM1_DutyCycle / 100
        PWM1_DutyCycle += 1
        sleep(0.01)

def soft_stop():
    """ Fungsi untuk menghentikan gerakan motor secara bertahap """
    global PWM1_DutyCycle
    while PWM1_DutyCycle > 0:
        GPIO.output([dir1, dir3], GPIO.HIGH)
        GPIO.output([dir2, dir4], GPIO.LOW)
        motor1_pwm.value = PWM1_DutyCycle / 100
        motor2_pwm.value = PWM1_DutyCycle / 100
        PWM1_DutyCycle -= 1
        sleep(0.01)

def main():
    try:
        while True:
            # Contoh: Soft start dan soft stop
            soft_start()
            sleep(5)  # Motor berjalan dengan kecepatan penuh selama 5 detik
            soft_stop()
            sleep(1)  # Motor berhenti selama 1 detik
            
            # Tambahkan fungsi maju, mundur, belok kanan dan belok kiri sesuai dengan kebutuhan

    except KeyboardInterrupt:
        # Pembersihan GPIO pada interupsi (Ctrl+C)
        GPIO.cleanup()

if __name__ == '__main__':
    main()
