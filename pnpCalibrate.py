from __future__ import print_function
import cv2
import numpy as np

patternSize = (7,7)

img = cv2.imread('calibrate/IMG_3069.JPG', cv2.IMREAD_COLOR)
if img is None:
    print('Could not open or find the images!')
    exit(0)

scaleFactor = .25

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# define range of blue color in HSV
lower_blue = np.array([81,62,250])
upper_blue = np.array([94,145,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

img = cv2.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor) 

# mask = cv2.resize(mask, (0,0), fx=scaleFactor, fy=scaleFactor) 


# cv2..calibrateCamera(objectPoints, imagePoints, imageSize[, cameraMatrix[, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]]]) → retval, cameraMatrix, distCoeffs, rvecs, tvecs
# Python: cv2.CalibrateCamera2(objectPoints, imagePoints, pointCounts, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags=0) → None¶

# cv2..drawChessboardCorners(image, patternSize, corners, patternWasFound) → None
retval, corners = cv2.findChessboardCorners(img, patternSize, None)
cv2.drawChessboardCorners(img, patternSize, corners, True)

print(corners)

# solvePnP
cv2.namedWindow("test")
cv2.imshow('Good Matches & Object detection', img)
while True:
	k = cv2.waitKey(1)
	if k%256 == 27:
	    # ESC pressed
	    print("Escape hit, closing...")
	    break
print("closing...")
cv2.destroyAllWindows()