import glob
import cv2
import os

paths = glob.glob('./final/final/*')

for path in paths:
    images = glob.glob(os.path.join(path+'/*.jpeg'))

    p = os.path.basename(path)
    try:
        if not os.path.exists('./final/final_rec/black/{}'.format(p)):
            os.makedirs('./final/final_rec/black/{}'.format(p))
            print('[*] Make dir ' + './final/final_rec/black/{}'.format(p))
    except OSError:
        print('Cannot creat directory - ./final/final_rec/black/{}'.format(p))

    for image in images:
        img = cv2.imread(image)
        h, w, _ = img.shape

        cv2.rectangle(img, (0, int(h/2)), (w, h), (0, 0, 0), -1)
        #
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        #
        # print('[*] Saving ----------' + './data_half/jy/crop/{}/{}'.format(p, os.path.basename(image)))
        cv2.imwrite('./final/final_rec/black/{}/{}'.format(p, os.path.basename(image)), img)

