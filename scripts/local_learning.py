# -*- coding: utf-8 -*-
# File              : local_learning.py
# Author            : Joy
# Create Date       : 2023/03/21
# Last Modified Date: 2023/04/13
# Last Modified By  : Joy
# Reference         : NA
# Description       : local learning and central learning methods
# ******************************************************
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, Dense


def build_model():
    main_input = Input(shape=(1280), name='image')
    x1 = Dense(64)(main_input)
    x = Dropout(0.5)(x1)
    pred = Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.01),
                 activity_regularizer=keras.regularizers.l1(0.001), name="label")(x)
    model = Model(inputs=[main_input], outputs=pred)
    return model

def build_actions_model(embd_size):
    main_input = Input(shape=(embd_size), name='img')
    x = Dense(128, activation='relu')(main_input)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    pred = Dense(3, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.01),
                 activity_regularizer=keras.regularizers.l1(0.001), name="label")(x)
    model = Model(inputs=[main_input], outputs=pred)
    return model


@how_much_time
def evaluate_exp(dirs, emd_mdl, model):
    x_test_path = dirs + "/test_action_imgs"+emd_mdl+"_9k.npy" #"/test_action_imgs_7k.npy"
    y_test_path = dirs + "/test_action_labels_9k.npy" #"/test_action_labels_7k.npy"
    n_classes = 3
    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    print("test : ", x_test.shape, "class distr: ", {i: (y_test == i).mean() for i in range(n_classes)})
    print("test : ", x_test.shape, "class distr: ", {i: (y_test == i).sum() for i in range(n_classes)})
    prediction_probs = model.predict(x_test, verbose=2)
    evaluation = compute_result(prediction_probs, y_test)
    # prediction_y.resize(img_y.shape)
    return prediction_probs, y_test, evaluation


def read_train_data(data_dir,emd_mdl):
    img_input = []
    img_y = []
    for i in range(3,4):
        img_embedding_path = data_dir + "/train"+str(i)+"_action_imgs"+emd_mdl+"_9k.npy"
        y_path = data_dir + "/train"+str(i)+"_action_labels_9k.npy"
        # img_y = np.loadtxt(y_path, delimiter=",", dtype=str).astype(int)
        img_y.extend(np.load(y_path).tolist())
        img_input.extend(np.load(img_embedding_path).tolist())
    img_input = np.array(img_input)[747:]
    img_y = pd.Series(img_y)[747:]
    print("train : ", img_input.shape, "class distr: ", {i: (img_y == i).mean() for i in range(n_classes)})
    img_input, img_y = random_train(img_input, img_y)
    return img_input, img_y


if __name__ == '__main__':
    import time
    data_dir = '../../data/task012'
    # data_dir = './data/task3'
    n_classes = 3
    emd_mdl = "_vit_xs" #_vit_xs, _vit_small， _mo
    embd_sz = 384  # 384, 640, 1024, 1280
    j = 4
    img_input, img_y = read_train_data(data_dir, emd_mdl)
    # img_input, img_y = read_new_data('./data/danger_detection/new', emd_mdl)
    print("train : ", img_input.shape, "class distr: ", {i: (img_y == i).mean() for i in range(n_classes)})
    Central = False
    if Central == True:
        model = build_actions_model(embd_sz)
        opt = keras.optimizers.Adam(clipvalue=8, learning_rate=0.0001)
        model.compile(optimizer=opt,
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), #keras.losses.BinaryCrossentropy(),
                      metrics=['acc'])#, keras.metrics.AUC(multi_label=True)]) # keras.metrics.Precision(),
        start = time.time()
        history = model.fit(
            img_input,
            img_y,
            batch_size=64,
            epochs=100,
            validation_split=0.3,
            class_weight={0: 1, 1: 1.5, 2: 0.5}
            # callbacks=[swarmCallback]
        )
        prediction_probs, y_test, evaluation = evaluate_exp(data_dir, emd_mdl, model)
    else:
        # new_model = keras.models.load_model('../../models/models/model'+emd_mdl+'_6'+str(j)+'.h5')
        # new_model = keras.models.load_model('../../models/cam_global_models_mbl' + emd_mdl + '_6' + str(j) + '.h5')
        # new_model = keras.models.load_model('../../models/fig9_node8/model_vit_xs_fig9_exp1.h5')
        # for j in range(1,3):
        new_model = keras.models.load_model('../../models/fig9_node10/exp3_model_vit_xs_fig9.h5')
        start = time.time()
        history = new_model.fit(
            img_input,
            img_y,
            batch_size=256,
            epochs=2,
            validation_split=0.3,
            # class_weight={0: 0.5, 1: 2, 2: 1}
            # callbacks=[swarmCallback]
        )
        prediction_probs, y_test, evaluation = evaluate_exp(data_dir, emd_mdl, new_model)
    evaluation['cost_time'] = time.time() - start
    print("-"*80)
    print(evaluation)
    print(evaluation['acc'], "sen: ", evaluation['sen_macro'], "f1: ", evaluation['f1_macro'], "auc: ", evaluation['auc_macro'])
    print(history.history)

# Epoch 50/50 8ms/step - loss: 0.5563 - acc: 1.0000 - val_loss: 0.5551 - val_acc: 1.0000
# Epoch 50/50 8ms/step - loss: 1.0019 - acc: 0.5293 - val_loss: 1.1061 - val_acc: 0.4026

# pd.DataFrame(history.history).to_csv('./report/history_1218_' + EXPERIMENT_IND + '.csv')
#
# # Save model and weights
# model.save('../../models/cam_mini_mbl_local04_v17.h5', save_format='h5')
# print('Saved the trained model!')

