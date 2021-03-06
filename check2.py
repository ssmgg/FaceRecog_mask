import math

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf

import glob
from mtcnn import MTCNN


from modules.evaluations import get_val_data, perform_val
from modules.models_arc import ArcFaceModel
from modules.utils import *
from modules.models_retina import RetinaFaceModel

import time

flags.DEFINE_string('cfg_path_arc', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')


def distance(embeddings1, embeddings2, distance_metric=1):
    if distance_metric == 0:
        # Euclidian distance
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg_arc = load_yaml(FLAGS.cfg_path_arc)
    cfg_retina = load_yaml(FLAGS.cfg_path_retina)

    model_arc = ArcFaceModel(size=cfg_arc['input_size'],
                         backbone_type=cfg_arc['backbone_type'],
                         training=False)
    model_retina = RetinaFaceModel(cfg_retina, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)


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


    print("Check Start!!!")

    file_dir = "C:/Users/chaehyun/PycharmProjects/ArcFace37_TF2x/lab_testimg/20200605_170317.jpg"
    # file_dir = "C:/Users/chaehyun/PycharmProjects/ArcFace37_TF2x/lab/m_mmj_img/.jpg"
    # ???????????? ????????? ??????
    npy_dir = "C:/Users/chaehyun/PycharmProjects/ArcFace37_TF2x/lab/lab_npy/*.npy"
    # ?????? ????????? npy ??????

    img_list = glob.glob(file_dir)
    # ????????? ????????? ??????
    npy_list = glob.glob(npy_dir)
    #npy ????????? ??????

    detector = MTCNN()

    for img_name in img_list:

        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        data_list = detector.detect_faces(img)

        for data in data_list:

            xmin, ymin, width, height = data['box']
            xmax = xmin + width
            ymax = ymin + height

            face_image = img[ymin:ymax, xmin: xmax, :]
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            img_resize = cv2.resize(crop, (cfg_arc['input_size'], cfg_arc['input_size']))
            img_resize = img_resize.astype(np.float32) / 255.
            if len(img_resize.shape) == 3:
                img_resize = np.expand_dims(img_resize, 0)
            embeds = l2_norm(model_arc(img_resize))

            # ????????? ?????? ?????????
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            # ?????? ?????? ?????????
            count = 0
            flag = False

            for npy_name in npy_list:
                # ????????? ????????? ??????
                name_embeds = np.load(npy_name)
                # ??????????????? ?????? ?????????
                dis = distance(embeds, name_embeds, 1)

                if dis < 0.4:
                    temp_name = npy_name.split('/')[-1].split('\\')[1].split('.npy')[0] #-> ???????????? ?????? ?????????
                    temp_dis = dis

                    if count == 0:
                        min_name = temp_name
                        min_dis = temp_dis
                        count += 1
                    else:
                        if min_dis > temp_dis:
                            min_dis = temp_dis
                            min_name = temp_name
                    flag = True

            if not flag:
                # ????????? ????????? ?????? ?????? Unknown
                cv2.putText(img, "Unknown", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            else:
                # ????????? ?????? ?????? ??????
                name = min_name + "_{}".format(min_dis)
                cv2.putText(img, name, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0 , 0), 1, cv2.LINE_AA)

        # height, width = img.shape[:2] -> img height, width ????????????
        # # print(width, height)
        # img = cv2.resize(img, (int(height * 0.5), int(width * 0.5)),interpolation = cv2.INTER_CUBIC) -> resize

        # ????????? ??????
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
