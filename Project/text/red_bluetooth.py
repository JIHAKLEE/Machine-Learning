from bluetooth import *

socket = BluetoothSocket( RFCOMM )
socket.connect(("98:DA:60:01:E4:D8", 1))
print("bluetooth connected!")

f = open("speech_to_text.txt",'r')
strings = f.read()
msg = strings
socket.send(msg)

print("finished!")
socket.close()
f.close()