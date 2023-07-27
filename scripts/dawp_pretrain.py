# -*- coding: utf-8 -*-
# File              : dawp_pretrain.py
# Author            : Joy
# Create Date       : 2023/03/11
# Last Modified Date: 2023/06/03
# Last Modified By  : Joy
# Reference         : NA
# Description       : split video data into image data, then use lightweight pretrain models to have smaller and high quality inputs
# ******************************************************

import os
import cv2
import numpy as np
import pandas as pd

def read_videos(train_video_dirs='./data/task3/train.txt'):
    """读取视频为图片"""
    x = []
    y = []
    with open(train_video_dirs) as f:
        for n,line in enumerate(f.readlines()):
            filename, label = line.split(" ")
            train_video_dirs_cur = train_video_dirs.rstrip(train_video_dirs.split('/')[-1])
            if not os.path.exists(train_video_dirs_cur + filename):
                print("No", n, filename)
                continue
            print("Read: ", n, filename)
            vin = cv2.VideoCapture(train_video_dirs_cur + filename)  # '1_climb.mp4'
            length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
            i = 1
            frm_rate = 5
            while i < length:
                ret, frame = vin.read()
                if ret:
                    if (i % frm_rate == 0) & (i>=15):
                        x.append(cv2.resize(frame, (224, 224)))  # (224, 256)
                        y.append(int(label))
                    i += 1
    return x, y

def embd_mblv2(imgs_input):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    # imgs_input = [cv2.resize(frame, (224, 224)) for frame in imgs_input]
    preprocess_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True,
                                                         input_shape=(224, 224, 3))
    # preprocess_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=True,
    #                                                      input_shape=(224, 224, 3))
    preprocessed_model = Model(inputs=preprocess_model.input,
                               outputs=preprocess_model.get_layer('global_average_pooling2d').output)
    mblv2_imgs_input = np.array(preprocessed_model.predict(np.array(imgs_input) / 255.0))
    print("train: ", mblv2_imgs_input.shape, "class distr: ", {i: (imgs_y == i).mean() for i in range(num_class)})
    np.save(imgs_path, mblv2_imgs_input)
    return mblv2_imgs_input

def embd_vitxs(imgs_input):
    import torch
    from mmpretrain import get_model

    imgs = np.rollaxis(np.array(imgs_input), 3, 1) / 255.0
    print("train: ", imgs.shape)

    model = get_model('mobileone-s0_8xb32_in1k', pretrained=True)
    # model = get_model('mobilevit-xsmall_3rdparty_in1k', pretrained=True)
    imgs = torch.tensor(imgs).to(torch.float32)
    result = []
    batch_size = 15
    for i in range(int(len(imgs) / batch_size)):
        out = imgs[i * batch_size:i * batch_size + batch_size]
        batch = model.extract_feat(out)
        result.extend(batch[0].tolist())
        print(i * batch_size + batch_size)
    if len(imgs) % batch_size != 0:
        batch = model.extract_feat(imgs[i * batch_size + batch_size:])
        result.extend(batch[0].tolist())
    return np.array(result)

if __name__ == '__main__':
    data_dir = '../../data/task3'
    # data_dir = './data/task3'
    emd_mdl = "_mo"

    train_dir = data_dir + "/train4.txt"
    imgs_path = data_dir + "/train4_action_imgs"+emd_mdl+"_9k.npy"
    y_path = data_dir + "/train3_action_labels_9k.npy"

    preprocess_modeling = 'MobileViT'

    if not os.path.exists(imgs_path):
        imgs_input, imgs_y = read_videos(train_dir)
        # imgs_y = pd.Series(imgs_y).replace(6,1) # wear_helmet
        num_class = imgs_y.nunique()
        # np.save(y_path, imgs_y.values)

        if preprocess_modeling == 'MoblieNet': imgs_embd = embd_mblv2(imgs_input)
        elif preprocess_modeling == 'MobileViT': imgs_embd = embd_vitxs(imgs_input)
        np.save(imgs_path, imgs_embd)
        print("train: ", imgs_embd.shape, "class distr: ", {i: (imgs_y == i).mean() for i in range(num_class)})

    # num_class = 3
    # y = np.load(y_path)
    # inpt = np.load('./data/task3/train2_action_imgs_vit_small_9k.npy')
    # print("train: ", inpt.shape, "class distr: ", {i: (y == i).mean() for i in range(num_class)})
