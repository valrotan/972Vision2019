import numpy as np
import cv2 as cv
import glob

scaleFactor = .25
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Load previously saved data with np.load('B.npz') as X:
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),2)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),2)
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
        #     cv.circle(img, , 1, (0, 255, 255), thickness=14)
        # cv.drawContours(img,[cnt],0,(0,0,255),-1)

def key(val):
    return val[0]

tx = 10
ty = 10

# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.float32([(-5.936, -.501, 0),(-4, 0, 0),(-5.377, 5.325, 0),(-7.313, 4.824, 0),
#         (5.936, -.501, 0),(4, 0, 0),(5.377, 5.325, 0),(7.313, 4.824, 0)])
objp = np.float32([(-5.936, -.501, 0),(-4, 0, 0),(-5.377, 5.325, 0),(-7.313, 4.824, 0),
        (5.936, -.501, 0),(4, 0, 0),(5.377, 5.325, 0),(7.313, 4.824, 0)])
# objp = np.float32([(0 + tx, 0, 0),(0 + tx, 5.5, 0),(2 + tx, 0, 0),(2 + tx, 5.5, 0)])

# axis = np.float32([[0,0,0], [5,0,0], [0,5,0], [0,0,-5]]).reshape(-1,3)
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

# cam = cv2.VideoCapture(0)


img = cv.imread('test/left.jpg')
img = cv.resize(img, (0,0), fx=scaleFactor, fy=scaleFactor) 



corners = detectCorners(img)

corners = np.float32(sorted(corners, key=lambda x: x[0]))
objp = np.float32(sorted(objp, key=lambda x: x[0]))

print('\n')
print(objp)
print(corners)
print('\n')

ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

print('R: %s' % rvecs)
print('T: %s' % tvecs)
print('obj: %s' % objp)

# project 3D points to image plane
imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

print(imgpts)
print('')
for val in mtx:
    for v in val:
        print('(float) %s, ' % v)

corner = tuple(imgpts[0].ravel())
img = draw(img,corner,imgpts)
cv.imshow('img',img)
k = cv.waitKey(-1) & 0xFF
if k == ord('s'):
    cv.imwrite(fname[:6]+'.png', img)
cv.destroyAllWindows()

