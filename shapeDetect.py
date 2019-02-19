import numpy as np
import cv2
 
scaleFactor = .25

img = cv2.imread('test/right.jpg')
img = cv2.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor) 

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([70,62,240])
upper_blue = np.array([94,200,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

im2,contours,h = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if len(approx) != 4:
    	continue

    x,y,w,h = cv2.boundingRect(cnt)
    if w < 10 or h < 10:
    	continue

    # print(approx)
    M = cv2.moments(cnt)

    for pnt in approx:
    	cv2.circle(img, tuple(pnt[0]), 1, (0, 255, 255), thickness=14)
    cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    break
