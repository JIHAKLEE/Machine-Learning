#!/bin/sh

echo "RED: Mic / Blue: Web cam"

gpio -g mode 23 input
gpio -g mode 23 up
gpio -g mode 24 input
gpio -g mode 24 up
gpio -g mode 25 input
gpio -g mode 25 up

while true; do
    sw1=`gpio -g read 23`
    if [ $sw1 -eq 0 ]; then
        cd /home/pi/0.Project/ocr
        python ex2.py >/home/pi/0.Project/text/image_to_text.txt 
        espeak -v ko -f /home/pi/0.Project/text/image_to_text.txt 
        
        cd /home/pi/0.Project/text
        python3 red_bluetooth.py
        
        exit
    fi
    
    sw2=`gpio -g read 24`
    if [ $sw2 -eq 0 ]; then
        python3 -m venv projectenv
        source /home/pi/projectenv/bin/activate
        
        cd /home/pi/projectenv/lib/python3.7/site-packages/googlesamples/assistant/grpc
        googlesamples-assistant-pushtotalk --project-id ai-project-fd3d7 --device-model-id ai-project-fd3d7-raspberry-pi-4-n68t8b  
        
        sw3=`gpio -g read 25`
        if [ $sw3 -eq 0 ]; then
            cd /home/pi/0.Project/text
            vi /home/pi/0.Project/text/speech_to_text.txt
            python3 blue_bluetooth.py
            
            exit
        fi
        exit
    fi
    
done
