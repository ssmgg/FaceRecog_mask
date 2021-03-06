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
flags.DEFINE_string('img_path', 'C:/Users/smgg/Desktop/202006_jeju/lab_crop', 'path to input image')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 0.3, 'down-scale factor for inputs')


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

    if not os.path.exists(FLAGS.img_path):
        print(f"cannot find image path from {FLAGS.img_path}")
        exit()

    print("[*] Processing on single image {}".format(FLAGS.img_path))

    img_path = glob.glob(os.path.join(FLAGS.img_path + '/*'))  # [n00000001, ... ]
    print('[*] Reading ' + FLAGS.img_path)
    # i=1
    for paths in img_path:  # paths = n000001
        path = os.path.basename(paths)
        try:
            if not os.path.exists('./eye/crop/{}'.format(path)):
                os.makedirs('./eye/crop/{}'.format(path))
                print('[*] Make dir '+'./eye/crop/{}'.format(path))
        except OSError:
            print('Cannot creat directory - ./eye/crop/{}'.format(path))

        print('[*] Reading ' + paths)
        imgs = glob.glob(os.path.join(paths + '/*.jpg'))  # n000001??? ?????? ?????? list
        # i = 1
        for img in imgs:
            print('[*] Reading ' + img)
            # cv2.imshow('', img)
            # cv2.waitKey(0)
            img_raw = cv2.imread(img)
            img_height_raw, img_width_raw, _ = img_raw.shape
            img_copy = np.float32(img_raw.copy())

            if FLAGS.down_scale_factor < 1.0:
                img_copy = cv2.resize(img_copy, (0, 0), fx=FLAGS.down_scale_factor, fy=FLAGS.down_scale_factor, interpolation=cv2.INTER_LINEAR)
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img_copy, pad_params = pad_input_image(img_copy, max_steps=max(cfg['steps']))

            # run model
            outputs = model(img_copy[np.newaxis, ...]).numpy()

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)

            # conf_arr = []
            # for prior_index in range(len(outputs)):
            #     bbox = get_bbox(img_raw, outputs[prior_index], img_height_raw,
            #                     img_width_raw)
            #     print(bbox)
            #     print(prior_index)
            #
            #     conf = bbox[prior_index]['confidence']
            #     conf_arr.append(conf)
            # max_conf_index = conf_arr.index(max(conf_arr))
            # print(max_conf_index)
            # print(conf_arr)

            # print(outputs)
            if len(outputs > 0):
                e = 0
                for o in outputs[0]:
                    if o < 0:
                        e = 1

                if e == 0:
                    prior_index = 0
                    bbox = get_bbox(img_raw, outputs[prior_index], img_height_raw,
                                    img_width_raw)
                    xmin, ymin, width, height = bbox[0]['box']
                    xmax = xmin + width
                    ymax = ymin + height
                    crop = img_raw[ymin:ymax, xmin: xmax, :]
                    # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    # cv2.imshow('', crop)
                    # cv2.waitKey(0)

                    # print('[*] saving -------- ' + './data/crop/{}/{}_{}_crop.jpg'.format(os.path.basename(paths), os.path.basename(paths), i))
                    # cv2.imwrite('./data/crop/{}/{}_{}_crop.jpg'.format(os.path.basename(paths), os.path.basename(paths), i), crop)


                    # print('[*] saving -------- ' + './images_500/images_500_crop/{}/{}.jpg'.format(path, os.path.basename(img)))
                    # cv2.imwrite('./data/crop/{}/{}_{}-m_crop.jpg'.format(os.path.basename(paths), os.path.basename(paths), i), crop)
                    cv2.imwrite('./eye/crop/{}/{}'.format(path, os.path.basename(img)), crop)

                    # i += 1


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
