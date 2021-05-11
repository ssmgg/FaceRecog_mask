import glob
import cv2
import os

paths = glob.glob('./final/final/*')

for path in paths:
    images = glob.glob(os.path.join(path+'/*.jpeg'))

    p = os.path.basename(path)
    try:
        if not os.path.exists('./final/final_half_embds/{}'.format(p)):
            os.makedirs('./final/final_half_embds/{}'.format(p))
            print('[*] Make dir ' + './final/final_half_embds/{}'.format(p))
    except OSError:
        print('Cannot creat directory - ./final/final_half_embds/{}'.format(p))

    for image in images:
        img = cv2.imread(image)
        h, w, _ = img.shape
        h = int(h/2)
        crop = img[:h, :, :]

        # cv2.imshow('', image)
        # cv2.waitKey(0)
        # cv2.imshow('', crop)
        # cv2.waitKey(0)

        # print('[*] Saving ----------' + './data_half/jy/crop/{}/{}'.format(p, os.path.basename(image)))
        cv2.imwrite('./final/final_half_embds/{}/{}'.format(p, os.path.basename(image)), crop)

