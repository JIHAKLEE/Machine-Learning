import cv2
import pytesseract
import time
import numpy as np
#import RPi.GPIO as GPIO
from PIL import Image
from pytesseract import Output
from gpiozero import Button
from time import sleep

button = Button(25)

cap = cv2.VideoCapture(0)  # 0: default camera

while cap.isOpened():
    # 카메라 프레임 읽기
    success, frame = cap.read()
    #GPIO.setmode(GPIO.BCM)
    #GPIO.setup(25,GPIO.IN,GPIO.PUD_UP)
    if success:
        # 프레임 출력
        cv2.imshow('Camera Window', frame)

        # ESC를 누르면 캡처
        key = cv2.waitKey(1) & 0xFF
        #if (key == 27):
        if button.is_pressed:
            return_value, image = cap.read()
            cv2.imwrite("capture/opencv.jpg", image)
            time.sleep(1)
            img_source = cv2.imread('capture/opencv.jpg') 
            
            text = pytesseract.image_to_string(Image.open("capture/opencv.jpg"), lang="eng+kor")
            print(text.replace(" ", ""))
            break


cap.release()
cv2.destroyAllWindows()
