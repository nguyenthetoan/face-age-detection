import cv2
from time import sleep

webcam = cv2.VideoCapture(0)
count = 0

while count < 50:
    if not webcam.isOpened():
        print("No active recording device...")
        sleep(5)
        pass

    ret, frame = webcam.read()
    print("new frame is being recorded?: ", ret)
    cv2.imwrite("data/frame-%d.jpg" % count, frame)
    count += 1

webcam.release()
cv2.destroyAllWindows()