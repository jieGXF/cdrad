import sklearn
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
# from dcurves import dca, plot_graphs
from statkit.decision import NetBenefitDisplay

def fun(x,index):
    #返回列表中第几个元素
    return x[index]

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, moda, color='crimson'):

    ax.plot(thresh_group, net_benefit_model, color=color, label=moda)
    #Figure Configuration， 美化一下细节
    ax.set_xlim(0, 1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_ylim(-0.25, net_benefit_model.max() + 0.15)  # adjustify the y axis limitation

    ax.set_xlabel(
        xlabel='High Risk Threshold',
        fontdict={'family': 'Times New Roman', 'fontsize': 13}
        )
    ax.set_ylabel(
        ylabel='Net Benefit',
        fontdict={'family': 'Times New Roman', 'fontsize': 13}
        )
    ax.grid('off')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')
    return ax


# def plot_DCAs(tag, preds, save_dir):
#     # plt.figure(figsize=(8, 8))
#     colors = ['red', 'yellow', 'blue', "green", "orange"]
#     for clf in preds[list(preds.keys())[0]]:
#         idx = 0
#         fig, ax = plt.subplots()
#         fig.set_size_inches(10, 8)
#         for moda in preds.keys():
#             y_label = preds[moda][clf]['labels']
#             y_pred_score = preds[moda][clf]['pred_scores'][:, 1][:, np.newaxis]
#             thresh_group = np.arange(0, 1, 0.01)
#             net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
#             net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)
#             if idx==0:
#                 ax.plot(thresh_group, net_benefit_all, color='black', label='ALL')
#                 ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='None')
#             ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, moda, color=colors[idx])
#             idx = idx + 1
#
#         plt.show()
#         plt.savefig(os.path.join(save_dir, tag + "_" + clf + '_DCA.png'))

def plot_DCAs(tag, preds, save_dir):
    # plt.figure(figsize=(8, 8))
    colors = ['red', 'yellow', 'blue', "green", "orange"]
    for clf in preds[list(preds.keys())[0]]:
        idx = 0
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 8)
        for moda in preds.keys():
            y_label = preds[moda][clf]['labels'].values.ravel()
            y_pred_score = preds[moda][clf]['pred_scores'][:, 1].ravel()
            if idx == len(preds.keys())-1:
                NetBenefitDisplay.from_predictions(y_label, y_pred_score, name=moda, ax=plt.gca())
            else:
                NetBenefitDisplay.from_predictions(y_label, y_pred_score, name=moda, show_references=False, ax=plt.gca())
            idx = idx + 1
        # plt.show()
        plt.savefig(os.path.join(save_dir, tag + "_" + clf + '_DCA.png'))
        plt.clf()



