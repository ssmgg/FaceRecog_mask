from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time

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
            if not os.path.exists('./data/black/img_n/{}'.format(p)):
                os.makedirs('./data/black/img_n/{}'.format(p))
                print('[*] Make dir ' + './data/black/img_n/{}'.format(p))
        except OSError:
            print('Cannot creat directory - ./data/black/img_n/{}'.format(p))

        i = 0
        for image in images:    # dis09_1_crop.jpg
            if i < 5:
                img_raw = cv2.imread(image)
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

                nose = keypoints['nose']
                crop = img_raw[:nose[1], :, :]


                # if outputs[0][14] > 0:
                #     cv2.circle(crop, (keypoints['left_eye']), 1, (255, 255, 0), 2)
                #     cv2.circle(crop, (keypoints['right_eye']), 1, (0, 255, 255), 2)
                #     cv2.circle(crop, (keypoints['nose']), 1, (255, 0, 0), 2)

                # cv2.imshow('', img_raw)
                # cv2.waitKey(0)
                # cv2.imshow('', crop)
                # cv2.waitKey(0)
                cv2.imwrite('./data/black/img_n/{}/{}'.format(p, os.path.basename(image)), crop)


                i += 1
            else:
                break



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
