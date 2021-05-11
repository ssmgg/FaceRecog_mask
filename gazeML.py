import os
from modules.models_elg import KerasELG
from keras import backend as K

import numpy as np
import cv2
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]='-1'

model = KerasELG()
model.net.load_weights('./checkpoints/elg_keras.h5')

fn = './minji.jpg'
input_img = cv2.imread(fn)

# ELG has fixed input shape of (108, 180, 1)
inp = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
inp = cv2.equalizeHist(inp)
inp = cv2.resize(inp, (180, 108))[np.newaxis, ..., np.newaxis]

# cv2.imshow('.', inp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print(inp.shape)

pred = model.net.predict(inp/255 * 2 - 1)

hm_r = np.max(pred[...,:8], axis=-1, keepdims=True)
hm_g = np.max(pred[...,8:16], axis=-1, keepdims=True)
hm_b = np.max(pred[...,16:], axis=-1, keepdims=True)

hm = np.concatenate([hm_r, hm_g, hm_b], axis=-1)
plt.imshow('eye heatmaps', hm)
# cv2.imshow('eye heatmaps', hm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
