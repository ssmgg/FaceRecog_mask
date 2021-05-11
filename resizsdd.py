import cv2
import os
import glob
import numpy as np


# h = []
# w = []
# images_9 = glob.glob('./data/minji/crop/dis09/*.jpg')
# for imgs in images_9:
#     img = cv2.imread(imgs)
#     he, we, _ = img.shape
#     h.append(he)
#     w.append(we)
#
# avg_h = int(np.mean(he))
# avg_w = int(np.mean(we))
#
# for image in images:
#     img = cv2.imread(image)
#     img_re = img.copy()
#     img_re = cv2.resize(img_re, (avg_w, avg_h))
#     cv2.imwrite('./data/{}'.format(os.path.basename(image)), img_re)
#


img_path = glob.glob('./data/black/img_rec_white/*')
i=0
for paths in img_path:
    path = os.path.basename(paths)

    try:
        if not os.path.exists('./data/black/img_rec_white_112/{}'.format(path)):
            os.makedirs('./data/black/img_rec_white_112/{}'.format(path))
            print('[*] Make dir '+'./data/black/img_rec_white_112/{}'.format(path))
    except OSError:
        print('Cannot creat directory - ./data/black/img_rec_white_112/{}'.format(path))

    print('[*] Reading ' + paths)
    imgs = glob.glob(os.path.join(paths + '/*.jpg'))  # n000001의 사진 경로 list

    # i = 1
    for img in imgs:
        print('[*] Reading ' + img)
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        img_raw = cv2.imread(img)

        img_raw = cv2.resize(img_raw, (112, 112), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('./data/black/img_rec_white_112/{}/{}'.format(path, os.path.basename(img)), img_raw)
        i+=1

print(i)
