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

    images = glob.glob('./eye/crop/*.jpg')    # [dis09_1_crop.jpg, dis09_2_crop.jpg, ... ]

    for image in images:    # dis09_1_crop.jpg
        print(image)
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
        # print(outputs)
        if len(outputs) > 0:
            e = 0
            for o in outputs[0]:
                if o < 0:
                    e = 1

            if e == 0:
                for prior_index in range(len(outputs)):
                    # draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw, img_width_raw)

                    # cv2.imshow('.', img_raw)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # detection
                    ann = outputs[prior_index]
                    width = 20
                    height = 20

                    left_xmin = int(ann[4] * img_width_raw) - width
                    left_xmax = left_xmin + width*2
                    left_ymin = int(ann[5] * img_height_raw) - height
                    left_ymax = left_ymin + height*2
                    # left_img = img_raw[left_ymin:left_ymax, left_xmin: left_xmax, :]

                    right_xmin = int(ann[6] * img_width_raw) - width
                    right_xmax = right_xmin + width*2
                    right_ymin = int(ann[7] * img_height_raw) - height
                    right_ymax = right_ymin + height*2
                    # right_img = img_raw[right_ymin:right_ymax, right_xmin: right_xmax, :]

                    xmin = min(right_xmin, left_xmin)
                    xmax = max(right_xmax, left_xmax)
                    ymin = min(right_ymin, left_ymin)
                    ymax = max(right_ymax, left_ymax)
                    print(ymin, ymax)
                    img = img_raw[ymin:ymax, 0:img_width_raw, :]

                    # if outputs[0][14] > 0:
                    #     cv2.circle(crop, (keypoints['left_eye']), 1, (255, 255, 0), 2)
                    #     cv2.circle(crop, (keypoints['right_eye']), 1, (0, 255, 255), 2)
                    #     cv2.circle(crop, (keypoints['nose']), 1, (255, 0, 0), 2)

                    cv2.imwrite('./eye/eyes/{}_{}.jpg'.format(os.path.basename(image).split('.')[0], prior_index), img)
        else:
            print('cannot find eye for image {}'.format(image))




if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
