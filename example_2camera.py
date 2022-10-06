import numpy as np
import cv2

video_0 = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
video_1 = cv2.VideoCapture(cv2.CAP_DSHOW + 1)

while True:
    ret0, frame0 = video_0.read()
    ret1, frame1 = video_1.read()    
    
    if (ret0):
        cv2.imshow('cam0', frame0)
        
    if (ret1):
        cv2.imshow('cam1', frame1)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_0.release()
video_1.release()
cv2.destroyAllWindows()