from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time
from scipy.spatial import distance as dist

from modules.models_retina import RetinaFaceModel
from modules.utils import *

import glob


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
# flags.DEFINE_float('down_scale_factor', 0.3, 'down-scale factor for inputs')


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    paths = glob.glob('./data/black/img/*')   # [ch, jh, jy, ... ]
    ddd = []

    for path in paths:  # mmj
        images = glob.glob(os.path.join(path+'/*.jpg'))    # [dis09_1_crop.jpg, dis09_2_crop.jpg, ... ]
        # dd = []

        p = os.path.basename(path)
        try:
            if not os.path.exists('./data/black/img_ratio_3/{}'.format(p)):
                os.makedirs('./data/black/img_ratio_3/{}'.format(p))
                print('[*] Make dir ' + './data/black/img_ratio_3/{}'.format(p))
        except OSError:
            print('Cannot creat directory - ./data/black/img_ratio_3/{}'.format(p))

        # i = 0
        for image in images:    # dis09_1_crop.jpg
            # if i < 3:
            #     d = []
                img_raw = cv2.imread(image)
                img_raw = cv2.resize(img_raw, dsize=(112, 112))
                img_height_raw, img_width_raw, _ = img_raw.shape
                img_copy = np.float32(img_raw.copy())

                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                # print(img_copy.shape)

                # pad input image to avoid unmatched shape problem
                img_copy, pad_params = pad_input_image(img_copy, max_steps=max(cfg['steps']))

                # run model
                outputs = model(img_copy[np.newaxis, ...]).numpy()

                # recover padding effect
                outputs = recover_pad_output(outputs, pad_params)

                # detection
                bbox = get_bbox(img_raw, outputs[0], img_height_raw, img_width_raw)
                keypoints = bbox[0]['keypoints']
                le_eye = keypoints['left_eye']
                ri_eye = keypoints['right_eye']
                nose = keypoints['nose']

                dis0 = ri_eye[0]-le_eye[0]

                dis1 = dist.euclidean([le_eye], [nose])
                dis2 = dist.euclidean([ri_eye], [nose])
                # print(dis0, dis1, dis2)

                # d = [[ri_eye[0]-le_eye[0], le_eye[1], ri_eye[1], le_eye[0], 112-ri_eye[0], 112-le_eye[1], 112-ri_eye[1]]]
                d = [[dis0, dis1, dis2]]
                print(d)
                np.save('./data/black/img_ratio_3/{}/{}.npy'.format(p, os.path.basename(image).split('.')[0]), d)

                # if outputs[0][14] > 0:
                #     cv2.circle(img_raw, (keypoints['left_eye']), 1, (255, 255, 0), 2)
                #     cv2.circle(img_raw, (keypoints['right_eye']), 1, (0, 255, 255), 2)

                # cv2.imshow('', img_raw)
                # cv2.waitKey(0)

                # dd.append(d)
                # print(dd)
            #     i += 1
            # else:
            #     break
        # ddd.append(dd)
    #     print(ddd)
    # print(ddd)

    # print('start!')
    # ddd_copy = ddd
    # f = open('data.csv', 'w')
    # for a in range(100):
    #     for k in range(2):
    #         for i in range(100):
    #             for j in range(2):
    #                 print((a, k), (i, j), ddd[a][k], ddd_copy[i][j])
    #                 dis = distance(ddd[a][k], ddd_copy[i][j], 1)
    #                 f.write(str(dis[0])+',')
    #         f.write('\n')

    # for i in range(100):
    #     for j in range(100):
    #         dis = distance(ddd[i][0], ddd_copy[j][1], 1)
    #         f.write(str(dis[0])+',')
    #     print('\n')
    #     f.write('\n')
    # print('end')
    # f.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
