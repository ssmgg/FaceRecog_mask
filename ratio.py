import glob
import os
import cv2


files_dir = glob.glob('./data/crop/*') # [dis01, dis02, ...]

for files in files_dir:
    imgs = glob.glob(os.path.join(files + '/*.jpg')) # [dis01_1_crop.jpg, dis01_2_crop,..]

    imgs_9 = cv2.imread('./data/crop/dis09/dis09_1_crop.jpg')
    h_9, w_9, _ = imgs_9.shape


    for img in imgs: #dis01_1_crop.jpg
        m = cv2.imread(img)
        height, width, _ = m.shape

        ra_h = height/h_9
        ra_w = width/w_9

        print(ra_h, ra_w, ra_h/ra_w)
        # print(m.shape)
    print('---------')
