from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time

from modules.models_arc import ArcFaceModel
from modules.models_retina import RetinaFaceModel
from modules.utils import *

import glob

flags.DEFINE_string('cfg_path_retina', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('cfg_path_arc', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', './data/dis', 'path to input image')
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

    cfg_arc = load_yaml(FLAGS.cfg_path_arc)
    cfg_retina = load_yaml(FLAGS.cfg_path_retina)

    # define network
    model_arc = ArcFaceModel(size=cfg_arc['input_size'],
                         backbone_type=cfg_arc['backbone_type'],
                         training=False)
    model_retina = RetinaFaceModel(cfg_retina, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint

    arc_ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg_arc['sub_name'])

    if arc_ckpt_path is not None:
        print("[*] load ckpt from {}".format(arc_ckpt_path))
        model_arc.load_weights(arc_ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(arc_ckpt_path))
        exit()

    retina_checkpoint_dir = './checkpoints/' + cfg_retina['sub_name']
    retina_checkpoint = tf.train.Checkpoint(model=model_retina)
    if tf.train.latest_checkpoint(retina_checkpoint_dir):
        retina_checkpoint.restore(tf.train.latest_checkpoint(retina_checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(retina_checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(retina_checkpoint_dir))
        exit()

    if not os.path.exists(FLAGS.img_path):
        print(f"cannot find image path from {FLAGS.img_path}")
        exit()

    print("[*] Processing on single image {}".format(FLAGS.img_path))

    img_path = glob.glob(os.path.join(FLAGS.img_path + '/*'))  # [dis01, dis02, ...]
    print('[*] Reading ' + FLAGS.img_path)
    for paths in img_path:  # paths = dis01
        print('[*] Reading ' + paths)
        imgs = glob.glob(os.path.join(paths + '/*.jpg'))  # dis01의 사진 경로 list
        i = 1
        for img in imgs:
            print('[*] Reading ' + img)
            img_raw = cv2.imread(img)
            img_height_raw, img_width_raw, _ = img_raw.shape
            img_copy = np.float32(img_raw.copy())

            if FLAGS.down_scale_factor < 1.0:
                img_copy = cv2.resize(img_copy, (0, 0), fx=FLAGS.down_scale_factor, fy=FLAGS.down_scale_factor, interpolation=cv2.INTER_LINEAR)
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img_copy, pad_params = pad_input_image(img_copy, max_steps=max(cfg_retina['steps']))

            # run model
            outputs = model_retina(img_copy[np.newaxis, ...]).numpy()

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

            for prior_index in range(len(outputs)):
                bbox = get_bbox(img_raw, outputs[prior_index], img_height_raw,
                                img_width_raw)
                xmin, ymin, width, height = bbox[0]['box']
                xmax = xmin + width
                ymax = ymin + height
                crop = img_raw[ymin:ymax, xmin: xmax, :]
                # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                # cv2.imshow('', crop)
                # cv2.waitKey(0)

                crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                img_resize = cv2.resize(crop, (cfg_arc['input_size'], cfg_arc['input_size']))
                img_resize = img_resize.astype(np.float32) / 255.
                if len(img_resize.shape) == 3:
                    img_resize = np.expand_dims(img_resize, 0)
                embeds = l2_norm(model_arc(img_resize))

                # print('[*] saving -------- ' + './data/crop/{}/{}_{}_crop.jpg'.format(os.path.basename(paths), os.path.basename(paths), i))
                # cv2.imwrite('./data/crop/{}/{}_{}_crop.jpg'.format(os.path.basename(paths), os.path.basename(paths), i), crop)
                np.save('./data_half/ch/mask/embd/{}/mj-{}.npy'.format(os.path.basename(paths),
                                                                            os.path.basename(paths)), embeds)
                print('[*] saving -------- ' + './data/crop/{}/{}_{}_crop.jpg'.format(os.path.basename(paths), os.path.basename(paths), i))
                cv2.imwrite('./data/crop/{}/{}_{}_crop.jpg'.format(os.path.basename(paths), os.path.basename(paths), i), crop)
                i += 1


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
