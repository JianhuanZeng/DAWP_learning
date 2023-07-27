# -*- coding: utf-8 -*-
# File              : dawp_main.py
# Author            : Joy
# Create Date       : 2023/03/21
# Last Modified Date: 2023/04/03
# Last Modified By  : Joy
# Reference         : 1. swarm learning; 2. learning-to-collaborate
# Description       : dawp learning
# ******************************************************
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dropout, Input, Dense, GRU, concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
# from scripts.DeepGradientCompressionOptimizer import AdamGradientCompressionOptimizer


from swarm import SwarmCallback

default_max_epochs = 30
default_min_peers = 1


def build_mulmod_model():
    # Multimodality-learning
    main_input = Input(shape=(4, 1280), name='image')
    # classifier_activation='sigmoid',
    # # 冻结前面的层，训练最后五层
    # for layers in covn_base.layers[:-3]:
    #     layers.trainable = True
    x1 = GRU(256, dropout=0.5, recurrent_dropout=0.1, return_sequences=True,
             input_shape=(4, 1280))(main_input)  # 参数 512
    x1 = GRU(128, dropout=0.5, recurrent_dropout=0.1)(x1)
    x1 = Dense(64)(x1)

    auxiliary_input = Input(shape=(4290,), name='feature')
    # x2 = Dense(128)(auxiliary_input)
    # x2 = Dropout(0.5)(x2)
    # x2 = Dense(64)(x2)
    x2 = Dense(1024)(auxiliary_input)
    x2 = Dense(512)(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(128)(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(64)(x2)

    x = concatenate([x1, x2])
    # x = Dense(128)(x)
    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    pred = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                 activity_regularizer=regularizers.l1(0.001), name="label")(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=pred)
    return model


def main(v_input, f_input, output, file_front):
    model = build_mulmod_model()
    # tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    # opt = AdamGradientCompressionOptimizer(clipvalue=5, learning_rate=0.001)
    # [optimizers.Adam, optimizers.RMSprop]
    opt = optimizers.Adam(clipvalue=5, learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(), 'acc']
                #   metrics=['mae', 'acc']
                  )

    swarmCallback = SwarmCallback(sync_interval=300,
                                  min_peers=default_min_peers,
                                  val_data=(
                                      {"image": v_input, "feature": f_input}, output),
                                  val_batch_size=32)

    history = model.fit(
        {"image": v_input, "feature": f_input},
        output,
        batch_size=32,
        epochs=default_max_epochs,
        validation_split=0.3,
        class_weight={0: 1.0, 1: 2.0},
        callbacks=[swarmCallback]
    )

    # save history
    pd.DataFrame(history.history).to_csv(
        '/platform/swarmml/model/save/model_history_' + str(i) + '.csv')

    # Save model and weights
    model.save('/platform/swarmml/model/save/yawn_tf_' +
               str(i) + '.h5', save_format='h5')
    print('Saved the trained model!')

    return history, model


def evaluate_exp(i):
    video_path_test = dir + "/test_split_300_500_video.npy"
    y_path_test = dir + "/test_split_300_500_y.npy"
    feature_path_test = dir + "/test_split_300_500_psg_feature_1.csv"
    video_y_test = np.load(y_path_test)
    video_input_test = np.load(video_path_test)
    feature_input_test = pd.read_csv(feature_path_test, index_col=0).values

    print("test set shape:")
    print(video_y_test.shape, video_input_test.shape, feature_input_test.shape)

    evaluateResult = model.evaluate(
        {"image": video_input_test, "feature": feature_input_test[0:1005]}, (video_y_test > 1).astype(int))
    print("evaluateResult:")
    print(evaluateResult)

    print("test set shape:")
    print(video_y_test.shape, video_input_test.shape, feature_input_test.shape)

    ind = str(i)
    with open('/platform/swarmml/model/save/test_0820_0.csv', "a") as f:
        f.write('exp' + ind + ', ' + str(evaluateResult)[1:-1] + '\n')


if __name__ == '__main__':
    dir = '/platform/swarmml/data/SegVideos/1015'
    video_path = dir + "/train_split_300_500_video.npy"
    y_path = dir + "/train_split_300_500_y.npy"
    feature_path = dir + "/train_split_300_500_psg_feature_1.csv"

    ori_video_y = np.load(y_path)
    ori_video_input = np.load(video_path)
    ori_feature_input = pd.read_csv(feature_path, index_col=0).values

    ids = np.random.choice(range(2318), 2318, replace=False)
    video_y = np.load(y_path)[ids]
    video_input = np.load(video_path)[ids]
    feature_input = pd.read_csv(feature_path, index_col=0).values[ids]

    print("train set shape:")
    print(video_y.shape, video_input.shape, feature_input.shape)

    i=1
    his, model = main(video_input, feature_input, (video_y > 1).astype(int), '/platform/swarmml/model/save')

    print("train set shape:")
    print(video_y.shape, video_input.shape, feature_input.shape)

    print("origin train set shape:")
    print(ori_video_y.shape, ori_video_input.shape, ori_feature_input.shape)

    evaluate_exp(i)
