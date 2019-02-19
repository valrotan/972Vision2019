import cv2 as cv
import numpy as np
import time
import math
import sys
# from networktables import NetworkTables

# ip = 'roborio-972-frc.local'
# NetworkTables.initialize(server=ip)
# sd = NetworkTables.getTable("SmartDashboard")
# programStart = time.time()
# print(programStart)

# cap.set(3, 160) #160
# cap.set(4, 120) #120

# cv2.destroyAllWindows()

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

scaleFactor = .25
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.float32([(-5.936, -.501, 0),(-4, 0, 0),(-5.377, 5.325, 0),(-7.313, 4.824, 0),
        (5.936, -.501, 0),(4, 0, 0),(5.377, 5.325, 0),(7.313, 4.824, 0)])
objp = np.float32(sorted(objp, key=lambda x: x[0]))

axis = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)

rvecs = [[0], [0], [0]]
tvecs = [[0], [0], [0]]

# Load previously saved data with np.load('B.npz') as X:
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
    return img

def detectCorners(img):

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([70,62,240])
    upper_blue = np.array([94,200,255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    im2,contours,h = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    pntSet = []

    for cnt in contours:
        epsilon = 0.1*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)
        if len(approx) != 4:
            continue

        x,y,w,h = cv.boundingRect(cnt)
        if w < 10 or h < 10:
            continue

        # print(approx)
        M = cv.moments(cnt)

        for pnt in approx:
            pntSet.append(tuple(pnt[0]))
            cv.circle(img, tuple(pnt[0]), 0, (0, 255, 255), thickness=5)
        # break
    print(pntSet)
    return np.float32(pntSet)
        # cv.circle(img, , 1, (0, 255, 255), thickness=14)
        # cv.drawContours(img,[cnt],0,(0,0,255),-1)

def key(val):
    return val[0]



# project 3D points to image plane
# imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

# corner = tuple(imgpts[0].ravel())
# img = draw(img,corner,imgpts)
# cv.imshow('img',img)
# k = cv.waitKey(-1) & 0xFF
# if k == ord('s'):
#     cv.imwrite(fname[:6]+'.png', img)
# cv.destroyAllWindows()

def solve(img):
    # img = cv.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor)
    corners = detectCorners(img)

    corners = np.float32(sorted(corners, key=lambda x: x[0]))
    if len(corners) != 8:
        print('no can do')
        return

    ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

    print('R: %s' % rvecs)
    print('T: %s' % tvecs)

cap = cv.VideoCapture(0)

while(1):
    print("\n")
    startTime = time.time();

    # Take each frame
    _, frame = cap.read()
    #frame = cv2.imread('test2.jpg')

    solve(frame)

    print(tvecs[0][0])
    print(tvecs[2][0])

    # sd.putNumber("Distance", round(cmDistance, 3))
    # sd.putNumber("visionAngle", angleFromCenter)

    # Display image
    cv.imshow('frame',frame)

    # Output time in milliseconds of processing time
    print("runtime ms: ", round((time.time()-startTime)*1000))

    # esc to kill program
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break