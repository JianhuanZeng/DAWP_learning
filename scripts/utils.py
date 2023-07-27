# -*- coding: utf-8 -*-
# File              : utils.py
# Author            : Joy
# Create Date       : 2023/04/21
# Last Modified Date: 2023/04/29
# Last Modified By  : Joy
# Reference         : NA
# Description       : utils
# ******************************************************
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, Dense

EXPERIMENT_IND = "v0"

def how_much_time(func):
    def inner(*args, **kwargs):
        # 开始时间
        t_start = time.time()
        z = func(*args, **kwargs)
        t_end = time.time()
        print("一共花费了{:3f}秒时间".format(t_end - t_start, ))
        return z

    return inner

def random_train(inpt, y):
    ids = np.random.choice(range(len(y)), len(y), replace=False)
    img_y = y.iloc[ids]
    img_input = inpt[ids]
    return img_input, img_y

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


def civil_noniid(dataset, num_users, iid=False):
    dict_users = {}
    if not iid:
        ids = np.random.randint(0, len(dataset), len(dataset))
    else:
        ids = list(range(len(dataset)))
    num_samples_per_usr = int(len(ids) / num_users)
    for usr in range(num_users-1):
        dict_users[usr] = ids[usr * num_samples_per_usr:(usr + 1) * num_samples_per_usr]
    dict_users[num_users-1] = ids[(usr + 1) * num_samples_per_usr:]
    return dict_users

def compute_result(prediction_probs, y):
    "sensitivity is recall; in multiolassical case, specificity is recall'"
    from sklearn.metrics import confusion_matrix
    result = {}
    n_classes = 3
    prediction_y = np.argmax(prediction_probs, axis=1)
    conf_matrix = confusion_matrix(prediction_y, y)
    result['acc'] = conf_matrix.trace()/conf_matrix.sum()
    result['pres'] = [conf_matrix[i][i]/conf_matrix[:,i].sum() for i in range(n_classes)]
    result['sens'] = [conf_matrix[i][i]/conf_matrix[i].sum() for i in range(n_classes)]
    result['f1_scores'] = [2*pre*rec/(pre+rec) for pre,rec in zip(result['pres'], result['sens'])]
    result['spes'] = [] # TN / (TN + FP)
    result['aucs'] = []
    for i in range(n_classes):
        if n_classes==2:
            conf_matrix_i = confusion_matrix
        else:
            conf_matrix_i = confusion_matrix((prediction_y != i).astype('int'), (y != i).astype('int'))
        fpr = conf_matrix_i[1][0]/conf_matrix_i[1].sum()
        tpr = conf_matrix_i[0][0]/conf_matrix_i[0].sum()
        result['spes'].append(1-fpr)
        result['aucs'].append(abs(fpr-tpr))
    result['pre_macro'] = np.mean(result['pres'])
    result['sen_macro'] = np.mean(result['sens'])
    result['spe_macro'] = np.mean(result['spes'])
    result['auc_macro'] = np.mean(result['aucs'])
    result['f1_macro'] = 2*result['pre_macro']*result['sen_macro']/(result['pre_macro']+result['sen_macro'])
    return result



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

def read_new_data(data_dir,emd_mdl):
    img_embedding_path = data_dir + "/image_embedding_for_danger_detection"+emd_mdl+".npy"
    y_path = data_dir + "/joy_labels_for_danger_detection.csv"
    img_input = np.load(img_embedding_path)
    img_y = pd.read_csv(y_path,header=None)
    print("train : ", img_input.shape, "class distr: ", img_y.mean().to_list())
    # img_input, img_y = random_train(img_input, img_y)
    return img_input, img_y
