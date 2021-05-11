import cv2
import numpy as np
import glob
import os

font = cv2.FONT_ITALIC


def faceDetect():
    eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")  # 눈찾기 haar 파일

    frames = glob.glob('./eye/eyes/*.jpg')
    print(frames)
    for f in frames:
        print(f)
        frame = cv2.imread(f)
        #     frame = cv2.GaussianBlur(frame, (9, 9), 0)
        #     frame = cv2.medianBlur(frame, 3)

        #         roi_gray= gray[y:y+h, x:x+w]
        #         roi_color= frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(frame)

        for (ex, ey, ew, eh) in eyes:
            ROI = frame[ey:ey + eh, ex:ex + ew]
            # cv2.imshow("", ROI)
            # cv2.waitKey(0)
            ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            threshold = 35
            kernel = np.ones((3, 3), np.uint8)
            new_frame = cv2.bilateralFilter(ROI_gray, 10, 15, 15)
            new_frame = cv2.erode(new_frame, kernel, iterations=4)
            ret, thresh = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)
            contour,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(ROI, contour, -1, (255, 0, 0), 2)
            try:
                x, y, w, h = cv2.boundingRect(contour[1])
            except IndexError:
                print('list index out of rang')
            print(x, y, w, h)
            cv2.rectangle(ROI, (x, y), (x+w, y+h), (255, 255, 0), 3)
            # cv2.imshow("",ROI)
            # cv2.waitKey(0)

        # cv2.imshow('Recording', frame)
        cv2.imwrite('./eye/dd/{}'.format(os.path.basename(f)), frame)
        # k = cv2.waitKey(0)


faceDetect()
