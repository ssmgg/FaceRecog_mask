import numpy as np
import os
import glob
import cv2


#
# img_path = glob.glob('./data/img_n/*')
#
# for paths in img_path:
#     path = os.path.basename(paths)
#
#     imgs = glob.glob(os.path.join(paths + './*.jpeg'))
#
#     for img in imgs:
#         print(img)
#         new = img.replace('JPEG', 'jpg')
#         print(new)
#         os.rename(img, new)
#         #
#         # old = os.path.basename(img)
#         # print(old)
#         # new = old.replace('JPEG', 'jpg')
#         # print(new)
#         # os.rename(old, new)


img_path = glob.glob('C:/Users/smgg/Desktop/202006_jeju/dispic/data/mmj/crop/*')  # [dis01, dis02, ... , dis11]
print(img_path)
avg = []
for paths in img_path:  # dis01
    imgs = glob.glob(os.path.join(paths + '/*.jpg'))    # [01.jpg, 02.jpg, ...]
    dis = []
    j = 0
    for img in imgs:
        i = cv2.imread(img)
        print('reading..' + img)
        _, w, _ = i.shape
        dis.append(w)
        j += 1
        if j == 5:
            print(dis)
            a = np.average(dis)
            print(a)
            avg.append(a)
print(avg)


