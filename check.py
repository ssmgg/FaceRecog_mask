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
from modules.utils import set_memory_growth, load_yaml, l2_norm
import time

flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
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

    cfg = load_yaml(FLAGS.cfg_path)

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

    print("Check Start!!!")

    file_dir = "C:/Users/chaehyun/PycharmProjects/ArcFace37_TF2x/lab_testimg/20200605_170317.jpg"
    # file_dir = "C:/Users/chaehyun/PycharmProjects/ArcFace37_TF2x/lab/m_mmj_img/.jpg"
    # 테스트할 이미지 경로
    npy_dir = "C:/Users/chaehyun/PycharmProjects/ArcFace37_TF2x/lab/lab_npy/*.npy"
    # 현재 저장된 npy 파일

    img_list = glob.glob(file_dir)
    # 이미지 리스트 읽기
    npy_list = glob.glob(npy_dir)
    #npy 리스트 읽기

    detector = MTCNN()

    for img_name in img_list:

        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        data_list = detector.detect_faces(img)

        for data in data_list:

            xmin, ymin, width, height = data['box']
            xmax = xmin + width
            ymax = ymin + height

            face_image = img[ymin:ymax, xmin: xmax, :]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            img_resize = cv2.resize(face_image, (cfg['input_size'], cfg['input_size']))
            img_resize = img_resize.astype(np.float32) / 255.
            if len(img_resize.shape) == 3:
                img_resize = np.expand_dims(img_resize, 0)
            embeds = l2_norm(model(img_resize))

            # 검출된 박스 그리기
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            # 해당 변수 초기화
            count = 0
            flag = False

            for npy_name in npy_list:
                # 저장된 임베딩 읽기
                name_embeds = np.load(npy_name)
                # 임베딩끼리 거리 구하기
                dis = distance(embeds, name_embeds, 1)

                if dis < 0.4:
                    temp_name = npy_name.split('/')[-1].split('\\')[1].split('.npy')[0] #-> 경로에서 이름 자르기
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
                # 검출된 사람이 없을 경우 Unknown
                cv2.putText(img, "Unknown", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            else:
                # 검출된 사람 이름 넣기
                name = min_name + "_{}".format(min_dis)
                cv2.putText(img, name, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0 , 0), 1, cv2.LINE_AA)

        # height, width = img.shape[:2] -> img height, width 가져오기
        # # print(width, height)
        # img = cv2.resize(img, (int(height * 0.5), int(width * 0.5)),interpolation = cv2.INTER_CUBIC) -> resize

        # 이미지 출력
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
