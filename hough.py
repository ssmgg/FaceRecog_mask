import numpy as np
import cv2
import os

def houghCircle(img_path, j):
    img = cv2.imread(img_path, 0)
    img_copy = img.copy()

    img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
    imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=55, param2=50, minRadius=0, maxRadius=60)
    print(circles)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 0), 1)

        cv2.imshow('HoughCricles', img)
        cv2.imwrite('{}.jpeg'.format(j), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Cannot Find any circle.')

img = cv2.imread('./minji.jpg', 0)
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=25, minRadius=0, maxRadius=50)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('img', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


img_path = './minji.jpg'
j = 6
# houghCircle(img_path, j)
