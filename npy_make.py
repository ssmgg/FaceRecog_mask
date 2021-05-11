import math


from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf

import glob


from modules.evaluations import get_val_data, perform_val
from modules.models_arc import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm
import time

flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')

flags.DEFINE_string('gpu', '0', 'which gpu to use')

def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml("./configs/arc_res50.yaml")

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])

    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()



    img_path = glob.glob('./data/black/img_rec_white/*')

    for paths in img_path:
        path = os.path.basename(paths)
        print('[*] Reading ' + paths)

        try:
            if not os.path.exists('./data/black/img_rec_white_embds/{}'.format(path)):
                os.makedirs('./data/black/img_rec_white_embds/{}'.format(path))
                print('[*] Make dir '+'./data/black/img_rec_white_embds/{}'.format(path))
        except OSError:
            print('Cannot creat directory - ./data/black/img_rec_white_embds/{}'.format(path))

        images = glob.glob(paths+'/*.jpg')

        i=0
        for image in images:
            if i < 5:
                img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
                img = img.astype(np.float32) / 255.
                if len(img.shape) == 3:
                    img = np.expand_dims(img, 0)
                embeds = l2_norm(model(img))

                # np.save('./data_half/ch/mask/embd/{}/ch-half-{}.npy'.format(os.path.basename(paths),
                #                                         os.path.basename(image).split('_crop')[0]), embeds)
                # # print('saving --------  ./data/embd/{}/si-{}.npy'.format(os.path.basename(paths).split('_')[0],
                # #                                         os.path.basename(image).split('_crop')[0]))
                print('[*] saving -------- ' + './data/black/img_rec_white_embds/{}/{}.npy'.format(path, os.path.basename(image).split('.')[0]))
                np.save('./data/black/img_rec_white_embds/{}/{}.npy'.format(path, os.path.basename(image).split('.')[0]), embeds)
                i += 1
            else:
                break
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass