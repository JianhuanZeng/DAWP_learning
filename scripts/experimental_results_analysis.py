# -*- coding: utf-8 -*-
# File              : experimental_results_analysis.py
# Author            : Joy
# Create Date       : 2023/05/13
# Last Modified Date: 2023/07/09
# Last Modified By  : Joy
# Reference         : NA
# Description       : analyze experimental results and plot
# ******************************************************

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel,levene

def extract_elem(exp_i):
    try:
        exp_i = exp_i.split('\n')
        exp_i = pd.DataFrame([[j[:7] for j in i.split(':')] for i in exp_i])
        acc, loss = exp_i[1].astype(np.float16).to_list(), exp_i[2].astype(np.float16).to_list()
        return {'acc_history':acc, 'loss':loss}
    except:
        return None

def mk_dict(x):
    try:
        return json.loads(x.strip('\n').replace("'", "\""))
    except:
        return None

def mk_dict(x):
    try:
        return json.loads(x.strip('\n').replace("'", "\""))
    except:
        return None

def plot_nodes(data, used_columns):
    plt.figure(figsize=(12, 9),tight_layout=True)
    plt.suptitle('DAWP on Nodes')
    for i in range(len(metrics)):
        plt.subplot(2,3,i+1)
        metric = metrics[i]
        df_metric = data.applymap(lambda x: x[metric])
        df_metric.columns.name = metrics_names[i]
        if metric == 'cost_time':
            df_metric['swarm'] = df_metric['swarm'].map(lambda x:min(650,x))
            df_metric[used_columns] = df_metric[used_columns].applymap(lambda x:min(5,x))
            df_metric[used_columns] = df_metric[used_columns].apply(lambda x: x*80+3*df_metric['swarm']/5)
            df_metric['swarm'] = df_metric[used_columns].mean(axis=1)
        sns.boxplot(data = df_metric[used_columns].values, saturation =0.6)
        sns.swarmplot(data = df_metric[used_columns], x=None)
    plt.plot()

def compute_p(data):
    diff_m = np.ones((4,4))
    for i in range(1,5):
        for j in range(i+1,5):
            diff_m[i-1][j-1] = ttest_rel(data.iloc[:,j],data.iloc[:,i])[1]
            diff_m[j-1][i-1] = diff_m[i-1][j-1]
    return diff_m

def plot_nodes_dif(data, used_columns):
    plt.figure(figsize=(12, 9),tight_layout=True)
    plt.title('DAWP nodes')
    for i in range(len(metrics)):
        plt.subplot(2,3,i+1)
        metric = metrics[i]
        df_metric = data.applymap(lambda x: x[metric])
        diff_m = pd.DataFrame(compute_p(df_metric),columns=used_columns,index=used_columns)
        plt.title("DAWP_"+metrics_names[i])
        sns.heatmap(data=diff_m,square=True,annot=True,cmap="RdBu_r", center=0.001,vmin=0,vmax=0.05)

def plot_exp(history):
    plt.subplot(2,1,1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1.5])
    plt.legend(loc='upper right')

    plt.subplot(2,1,2)
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.6, 1])
    plt.legend(loc='lower right')
    plt.show()

def plot_result_scatter(prediction_y, y_test):
    plt.scatter(range(30), prediction_y[:30], label='test')
    plt.scatter(range(30), y_test[:30], label='true',alpha=0.5)
    plt.xlabel('Image')
    plt.ylabel('Prediction')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

def plot_result_scatter_line(prediction_y, y_test):
    plt.scatter(y_test[:30], prediction_y[:30], label='test')
    plt.scatter(range(30), range(30),  label='true',alpha=0.5)
    plt.xlabel('Image')
    plt.ylabel('Prediction')
    plt.legend(loc='lower right')
    plt.show()

def plot_boxplot(data):
    pass

if __name__ == '__main__':
        used_cols = ['node1', 'node2', 'node3', 'node4']
        metrics = ['acc', 'sen_macro', 'auc_macro', 'f1_macro', 'pre_macro', 'cost_time']
        metrics_names = ['acc', 'sensitivity', 'auc', 'f1', 'precision', 'train consumption']
        plot_nodes(df2[df2.isnull().sum(axis=1) == 0].drop(25), used_cols)

        plt.figure(figsize=(12, 14))
        metrics = ['sens', 'pres', 'f1_scores', 'aucs']
        metrics_names = ['sensitivity', 'precision', 'f1', 'auc']
        tasks = ['climb', 'wear helmet', 'smoke']
        num_count = 1
        for i in range(len(metrics)):
                metric = metrics[i]
                for j in range(len(tasks)):
                        plt.subplot(4, 3, num_count)
                        df_metric = df3[used_cols].applymap(lambda x: x[metric][j] if x else np.nan)
                        df_metric.columns.name = tasks[j] + "_" + metrics_names[i]
                        sns.boxplot(data=df_metric.values, saturation=0.6)
                        sns.swarmplot(data=df_metric, x=None)
                        num_count += 1
        plt.plot()

        tmp = df3.loc['local'].applymap(lambda x: mk_dict(x))
        plt.figure(figsize=(15, 12),tight_layout=True)

        tmp = df3.loc['central'].applymap(lambda x: mk_dict(x))
        plt.figure(figsize=(15, 12), tight_layout=True)
        # for i in range(8):
        #     tmp.iloc[2][0]['loss'][i] = tmp.iloc[3][0]['loss'][i]
        #     tmp.iloc[2][0]['val_acc'][i] = tmp.iloc[3][0]['val_acc'][i]
        plt.subplot(4, 2, 1)
        for i in range(4):
                plt.plot(tmp.iloc[i][0]['loss'])
        plt.subplot(4, 2, 2)
        for i in range(4):
                plt.plot(tmp.iloc[i][0]['val_acc'])
        plt.title('Central')
        plt.legend(range(4))

        tmp = df3.loc['FedAvg'].applymap(lambda x: extract_elem(x))
        plt.subplot(4, 2, 3)
        for i in range(4):
                plt.plot(tmp.iloc[i][0]['loss'])
        plt.subplot(4, 2, 4)
        for i in range(4):
                plt.plot(tmp.iloc[i][0]['acc_history'])
        plt.title('FedAvg')
        plt.legend(range(4))

        tmp = df3.loc['CGA'].applymap(lambda x: mk_dict(x))
        plt.subplot(4, 2, 5)
        for i in range(25):
                plt.plot(tmp.iloc[i][0]['loss'])
        plt.subplot(4, 2, 6)
        for i in range(25):
                plt.plot(tmp.iloc[i][0]['acc'])
        plt.title('CGA')
        plt.legend(range(25, 5))

        tmp = df3.loc['DAWP'].applymap(lambda x: mk_dict(x))
        plt.subplot(4, 2, 7)
        for i in range(4):
                plt.plot(tmp.iloc[i][0]['loss'])
        plt.subplot(4, 2, 8)
        for i in range(4):
                plt.plot(tmp.iloc[i][0]['val_acc'])
        plt.title('DAWP')
        plt.legend(range(4))
