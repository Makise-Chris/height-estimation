import cv2
import numpy as np
capture = cv2.VideoCapture('gmm.mp4')
backSub = cv2.createBackgroundSubtractorMOG2()

last_frame = 112
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    if capture.get(cv2.CAP_PROP_POS_FRAMES) == last_frame:
        cv2.imwrite("last_frame.png", frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break