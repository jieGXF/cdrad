#!/usr/bin/env python

# Copyright 2016-2021 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import tikzplotlib
import pandas as pd
import argparse
# from WORC.plotting.compute_CI import compute_confidence as CI
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.metrics import precision_recall_curve, RocCurveDisplay
import csv
import os
# from WORC.plotting.plot_estimator_performance import plot_estimator_performance
from sklearn.preprocessing import StandardScaler, OneHotEncoder,  LabelBinarizer
from itertools import combinations

def plot_single_ROC(y_truth, y_score, verbose=False, returnplot=False):
    '''
    Get the False Positive Ratio (FPR) and True Positive Ratio (TPR)
    for the ground truth and score of a single estimator. These ratios
    can be used to plot a Receiver Operator Characteristic (ROC) curve.
    '''
    # Sort both lists based on the scores
    y_truth = np.asarray(y_truth)
    y_truth = np.int_(y_truth)
    y_score = np.asarray(y_score)
    inds = y_score.argsort()
    y_truth_sorted = y_truth[inds]
    y_score = y_score[inds]

    # Compute the TPR and FPR for all possible thresholds
    FP = 0
    TP = 0
    fpr = list()
    tpr = list()
    thresholds = list()
    fprev = -np.inf
    i = 0
    N = float(np.bincount(y_truth)[0])
    if len(np.bincount(y_truth)) == 1:
        # No class = 1 present.
        P = 0
    else:
        P = float(np.bincount(y_truth)[1])

    if N == 0:
        print('[WORC Warning] No negative class samples found, cannot determine ROC. Skipping iteration.')
        return fpr, tpr, thresholds
    elif P == 0:
        print('[WORC Warning] No positive class samples found, cannot determine ROC. Skipping iteration.')
        return fpr, tpr, thresholds

    while i < len(y_truth_sorted):
        if y_score[i] != fprev:
            fpr.append(1 - FP/N)
            tpr.append(1 - TP/P)
            thresholds.append(y_score[i])
            fprev = y_score[i]

        if y_truth_sorted[i] == 1:
            TP += 1
        else:
            FP += 1

        i += 1

    if verbose or returnplot:
        roc_auc = auc(fpr, tpr)
        f = plt.figure()
        ax = plt.subplot(111)
        lw = 2
        ax.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

    if not returnplot:
        return fpr[::-1], tpr[::-1], thresholds[::-1]
    else:
        return fpr[::-1], tpr[::-1], thresholds[::-1], f


def plot_single_PRC(y_truth, y_score, verbose=False, returnplot=False):
    '''
    Get the precision and recall (=true positive rate)
    for the ground truth and score of a single estimator. These ratios
    can be used to plot a Precision Recall Curve (ROC).
    '''
    # Sort both lists based on the scores
    y_truth = np.asarray(y_truth)
    y_truth = np.int_(y_truth)
    y_score = np.asarray(y_score)
    inds = y_score.argsort()
    y_truth_sorted = y_truth[inds]
    y_score = y_score[inds]

    # Compute the TPR and FPR for all possible thresholds
    N = float(np.bincount(y_truth)[0])
    if len(np.bincount(y_truth)) == 1:
        # No class = 1 present.
        P = 0
    else:
        P = float(np.bincount(y_truth)[1])

    if N == 0:
        print('[WORC Warning] No negative class samples found, cannot determine PRC. Skipping iteration.')
        return list(), list(), list()
    elif P == 0:
        print('[WORC Warning] No positive class samples found, cannot determine PRC. Skipping iteration.')
        return list(), list(), list()

    precision, tpr, thresholds =\
        precision_recall_curve(y_truth_sorted, y_score)

    # Convert to lists
    precision = precision.tolist()
    tpr = tpr.tolist()
    thresholds = thresholds.tolist()


    if verbose or returnplot:
        prc_auc = auc(tpr, precision)
        f = plt.figure()
        ax = plt.subplot(111)
        lw = 2
        ax.plot(tpr, precision, color='darkorange',
                lw=lw, label='PR curve (area = %0.2f)' % prc_auc)
        ax.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")

    if not returnplot:
        return tpr[::-1], precision[::-1], thresholds[::-1]
    else:
        return tpr[::-1], precision[::-1], thresholds[::-1], f


def curve_thresholding(metric1t, metric2t, thresholds, nsamples=20):
    '''
    Construct metric1 and metric2 (either FPR and TPR, or TPR and Precision)
    ratios at different thresholds for the scores of an estimator.
    '''
    # Combine all found thresholds in a list and create samples
    T = list()
    for t in thresholds:
        T.extend(t)
    T = sorted(T)
    tsamples = np.linspace(0, len(T) - 1, nsamples)

    # Compute the metric1s and metric2s at the sample points
    nrocs = len(metric1t)
    metric1 = np.zeros((nsamples, nrocs))
    metric2 = np.zeros((nsamples, nrocs))

    th = list()
    for n_sample, tidx in enumerate(tsamples):
        tidx = int(tidx)
        th.append(T[tidx])
        for i_roc in range(0, nrocs):
            idx = 0
            while float(thresholds[i_roc][idx]) > float(T[tidx]) and idx < (len(thresholds[i_roc]) - 1):
                idx += 1
            metric1[n_sample, i_roc] = metric1t[i_roc][idx]
            metric2[n_sample, i_roc] = metric2t[i_roc][idx]

    return metric1, metric2, th


def plot_ROC_CIc(y_truth, y_score, N_1, N_2, plot='default', alpha=0.95,
                 verbose=False, DEBUG=False, tsamples=20):
    '''
    Plot a Receiver Operator Characteristic (ROC) curve with confidence intervals.

    tsamples: number of sample points on which to determine the confidence intervals.
              The sample pointsare used on the thresholds for y_score.
    '''
    # Compute ROC curve and ROC area for each class
    fprt = list()
    tprt = list()
    roc_auc = list()
    thresholds = list()
    for yt, ys in zip(y_truth, y_score):
        fpr_temp, tpr_temp, thresholds_temp = plot_single_ROC(yt, ys)
        if fpr_temp:
            roc_auc.append(roc_auc_score(yt, ys))
            fprt.append(fpr_temp)
            tprt.append(tpr_temp)
            thresholds.append(thresholds_temp)

    # Sample FPR and TPR at numerous points
    fpr, tpr, th = curve_thresholding(fprt, tprt, thresholds, tsamples)

    # Compute the confidence intervals for the ROC
    CIs_tpr = list()
    CIs_fpr = list()
    for i in range(0, tsamples):
        if i == 0:
            # Point (1, 1) is always in there, but shows as (nan, nan)
            CIs_fpr.append([1, 1])
            CIs_tpr.append([1, 1])
        else:
            cit_fpr = CI(fpr[i, :], N_1, N_2, alpha)
            CIs_fpr.append([cit_fpr[0], cit_fpr[1]])
            cit_tpr = CI(tpr[i, :], N_1, N_2, alpha)
            CIs_tpr.append([cit_tpr[0], cit_tpr[1]])

    # The point (0, 0) is also always there but not computed
    CIs_fpr.append([0, 0])
    CIs_tpr.append([0, 0])

    # Calculate also means of CIs after converting to array
    CIs_tpr = np.asarray(CIs_tpr)
    CIs_fpr = np.asarray(CIs_fpr)
    CIs_tpr_means = np.mean(CIs_tpr, axis=1).tolist()
    CIs_fpr_means = np.mean(CIs_fpr, axis=1).tolist()

    # compute AUC CI
    roc_auc = CI(roc_auc, N_1, N_2, alpha)

    f = plt.figure()
    lw = 2
    subplot = f.add_subplot(111)
    subplot.plot(CIs_fpr_means, CIs_tpr_means, color='orange',
                 lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))

    for i in range(0, len(CIs_fpr_means)):
        if CIs_tpr[i, 1] <= 1:
            ymax = CIs_tpr[i, 1]
        else:
            ymax = 1

        if CIs_tpr[i, 0] <= 0:
            ymin = 0
        else:
            ymin = CIs_tpr[i, 0]

        if CIs_tpr_means[i] <= 1:
            ymean = CIs_tpr_means[i]
        else:
            ymean = 1

        if CIs_fpr[i, 1] <= 1:
            xmax = CIs_fpr[i, 1]
        else:
            xmax = 1

        if CIs_fpr[i, 0] <= 0:
            xmin = 0
        else:
            xmin = CIs_fpr[i, 0]

        if CIs_fpr_means[i] <= 1:
            xmean = CIs_fpr_means[i]
        else:
            xmean = 1

        if DEBUG:
            print(xmin, xmax, ymean)
            print(ymin, ymax, xmean)

        subplot.plot([xmin, xmax],
                     [ymean, ymean],
                     color='black', alpha=0.15)
        subplot.plot([xmean, xmean],
                     [ymin, ymax],
                     color='black', alpha=0.15)

    subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    if verbose:
        plt.show()

        f = plt.figure()
        lw = 2
        subplot = f.add_subplot(111)
        subplot.plot(CIs_fpr_means, CIs_tpr_means, color='darkorange',
                     lw=lw, label='ROC curve (AUC = (%0.2f, %0.2f))' % (roc_auc[0], roc_auc[1]))

        for i in range(0, len(CIs_fpr_means)):
            if CIs_tpr[i, 1] <= 1:
                ymax = CIs_tpr[i, 1]
            else:
                ymax = 1

            if CIs_tpr[i, 0] <= 0:
                ymin = 0
            else:
                ymin = CIs_tpr[i, 0]

            if CIs_tpr_means[i] <= 1:
                ymean = CIs_tpr_means[i]
            else:
                ymean = 1

            if CIs_fpr[i, 1] <= 1:
                xmax = CIs_fpr[i, 1]
            else:
                xmax = 1

            if CIs_fpr[i, 0] <= 0:
                xmin = 0
            else:
                xmin = CIs_fpr[i, 0]

            if CIs_fpr_means[i] <= 1:
                xmean = CIs_fpr_means[i]
            else:
                xmean = 1

            subplot.plot([xmin, xmax],
                         [ymean, ymean],
                         color='black', alpha=0.15)
            subplot.plot([xmean, xmean],
                         [ymin, ymax],
                         color='black', alpha=0.15)

        subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

    return f, CIs_fpr, CIs_tpr


def plot_PRC_CIc(y_truth, y_score, N_1, N_2, plot='default', alpha=0.95,
                 verbose=False, DEBUG=False, tsamples=20):
    '''
    Plot a Precision-Recall curve with confidence intervals.

    tsamples: number of sample points on which to determine the confidence intervals.
              The sample pointsare used on the thresholds for y_score.
    '''
    # Compute PR curve and PR area for each class
    tprt = list()
    precisiont = list()
    prc_auc = list()
    thresholds = list()
    for yt, ys in zip(y_truth, y_score):
        tpr_temp, precision_temp, thresholds_temp = plot_single_PRC(yt, ys)
        if tpr_temp:
            prc_auc.append(auc(tpr_temp, precision_temp))
            tprt.append(tpr_temp)
            precisiont.append(precision_temp)
            thresholds.append(thresholds_temp)

    # Sample TPR and precision at numerous points
    tpr, precisionr, th = curve_thresholding(tprt, precisiont, thresholds, tsamples)

    # Compute the confidence intervals for the ROC
    CIs_precisionr = list()
    CIs_tpr = list()
    for i in range(0, tsamples):
        if i == 0:
            # Point (0, 0) is always in there, but shows as (nan, nan)
            CIs_tpr.append([1, 1])
            CIs_precisionr.append([0, 0])
        else:
            cit_tpr = CI(tpr[i, :], N_1, N_2, alpha)
            CIs_tpr.append([cit_tpr[0], cit_tpr[1]])

            cit_precisionr = CI(precisionr[i, :], N_1, N_2, alpha)
            CIs_precisionr.append([cit_precisionr[0], cit_precisionr[1]])

    # The point (0, 1) is also always there but not computed
    CIs_tpr.append([0, 0])
    CIs_precisionr.append([1, 1])

    # Calculate also means of CIs after converting to array
    CIs_precisionr = np.asarray(CIs_precisionr)
    CIs_tpr = np.asarray(CIs_tpr)
    CIs_precisionr_means = np.mean(CIs_precisionr, axis=1).tolist()
    CIs_tpr_means = np.mean(CIs_tpr, axis=1).tolist()

    # compute AUC CI
    prc_auc = CI(prc_auc, N_1, N_2, alpha)

    f = plt.figure()
    lw = 2
    subplot = f.add_subplot(111)
    subplot.plot(CIs_tpr_means, CIs_precisionr_means, color='orange',
                 lw=lw, label='PR curve (AUC = (%0.2f, %0.2f))' % (prc_auc[0], prc_auc[1]))

    for i in range(0, len(CIs_tpr_means)):
        if CIs_precisionr[i, 1] <= 1:
            ymax = CIs_precisionr[i, 1]
        else:
            ymax = 1

        if CIs_precisionr[i, 0] <= 0:
            ymin = 0
        else:
            ymin = CIs_precisionr[i, 0]

        if CIs_precisionr_means[i] <= 1:
            ymean = CIs_precisionr_means[i]
        else:
            ymean = 1

        if CIs_tpr[i, 1] <= 1:
            xmax = CIs_tpr[i, 1]
        else:
            xmax = 1

        if CIs_tpr[i, 0] <= 0:
            xmin = 0
        else:
            xmin = CIs_tpr[i, 0]

        if CIs_tpr_means[i] <= 1:
            xmean = CIs_tpr_means[i]
        else:
            xmean = 1

        if DEBUG:
            print(xmin, xmax, ymean)
            print(ymin, ymax, xmean)

        subplot.plot([xmin, xmax],
                     [ymean, ymean],
                     color='black', alpha=0.15)
        subplot.plot([xmean, xmean],
                     [ymin, ymax],
                     color='black', alpha=0.15)

    subplot.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")

    if verbose:
        plt.show()

        f = plt.figure()
        lw = 2
        subplot = f.add_subplot(111)
        subplot.plot(CIs_tpr_means, CIs_precisionr_means, color='darkorange',
                     lw=lw, label='PRC curve (AUC = (%0.2f, %0.2f))' % (prc_auc[0], prc_auc[1]))

        for i in range(0, len(CIs_tpr_means)):
            if CIs_precisionr[i, 1] <= 1:
                ymax = CIs_precisionr[i, 1]
            else:
                ymax = 1

            if CIs_precisionr[i, 0] <= 0:
                ymin = 0
            else:
                ymin = CIs_precisionr[i, 0]

            if CIs_precisionr[i] <= 1:
                ymean = CIs_precisionr[i]
            else:
                ymean = 1

            if CIs_tpr[i, 1] <= 1:
                xmax = CIs_tpr[i, 1]
            else:
                xmax = 1

            if CIs_tpr[i, 0] <= 0:
                xmin = 0
            else:
                xmin = CIs_tpr[i, 0]

            if CIs_tpr_means[i] <= 1:
                xmean = CIs_tpr_means[i]
            else:
                xmean = 1

            if DEBUG:
                print(xmin, xmax, ymean)
                print(ymin, ymax, xmean)

            subplot.plot([xmin, xmax],
                         [ymean, ymean],
                         color='black', alpha=0.15)
            subplot.plot([xmean, xmean],
                         [ymin, ymax],
                         color='black', alpha=0.15)

        subplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")

    return f, CIs_tpr, CIs_precisionr


def plot_ovo_roc(tag, y_true, y_probs, save_dir):
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    label_binarizer = LabelBinarizer().fit(y_true)
    from itertools import combinations
    pair_list = list(combinations(np.unique(y_true), 2))
    print(pair_list)
    pair_scores = []
    mean_tpr = dict()
    for ix, (label_a, label_b) in enumerate(pair_list):
        y_true = np.array(y_true)
        y_true = np.squeeze(y_true)
        a_mask = y_true == label_a
        b_mask = y_true == label_b
        ab_mask = np.logical_or(a_mask, b_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0]
        idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]

        fpr_a, tpr_a, _ = roc_curve(a_true, y_probs[ab_mask, idx_a])
        fpr_b, tpr_b, _ = roc_curve(b_true, y_probs[ab_mask, idx_b])

        mean_tpr[ix] = np.zeros_like(fpr_grid)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr[ix] /= 2
        mean_score = auc(fpr_grid, mean_tpr[ix])
        pair_scores.append(mean_score)

    macro_roc_auc_ovo = roc_auc_score(y_true, y_probs, multi_class="ovo", average="macro")
    # micro_roc_auc_ovo = roc_auc_score(y_true, y_probs, multi_class="ovo", average="micro")
    # print(f"Macro-averaged One-vs-One ROC AUC score:\n{macro_roc_auc_ovo:.2f}")

    ovo_tpr = np.zeros_like(fpr_grid)
    fig, ax = plt.subplots(figsize=(6, 6))
    for ix, (label_a, label_b) in enumerate(pair_list):
        ovo_tpr += mean_tpr[ix]
        plt.plot(fpr_grid, mean_tpr[ix],
                 label=f"Mean {label_a} vs {label_b} (AUC = {pair_scores[ix]:.2f})", )

    ovo_tpr /= sum(1 for pair in enumerate(pair_list))
    plt.plot(fpr_grid, ovo_tpr,
             label=f"One-vs-One macro-average (AUC = {macro_roc_auc_ovo:.2f})",
             linestyle=":",
             linewidth=4,)

    plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-One multiclass")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(save_dir, tag + '_rocs.png'))


def plot_ovr_roc(tag, y_true, y_probs, save_dir):
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    label_binarizer = LabelBinarizer().fit(y_true)
    y_onehot_true = label_binarizer.transform(y_true)
    n_classes = y_onehot_true.shape[1]
    from sklearn.metrics import auc, roc_curve
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_true.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_true[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

    macro_roc_auc_ovr = roc_auc_score(
        y_true,
        y_probs,
        multi_class="ovr",
        average="macro",
    )
    # print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")

    from itertools import cycle
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_true[:, class_id],
            y_probs[:, class_id],
            # name=f"ROC curve for {target_names[class_id]}",
            name=f"ROC curve for {class_id}",
            color=color,
            ax=ax,
            # plot_chance_level=(class_id == 2),
        )
    plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(save_dir, tag + '_rocs.png'))
    plt.cla()
    plt.close("all")


def plot_cv_roc(tag, save_dir, mean_fpr,  mean_tpr_train, fprs_train, tprs_train, aucs_train):
    fig, ax = plt.subplots(figsize=(7, 7))
    for k in range(5):
        plt.plot(fprs_train[k], tprs_train[k], lw=1, alpha=0.5,
                 label='ROC fold %d (AUC = %0.2f)' % (k, aucs_train[k]))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(mean_tpr_train, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_train)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(mean_tpr_train, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    # mean_acc = np.mean(ACCs_val, axis=0)
    # std_acc = np.std(ACCs_val, axis=0)
    # mean_sen = np.mean(SENs_val, axis=0)
    # std_sen = np.std(SENs_val, axis=0)
    # mean_spe = np.mean(SPEs_val, axis=0)
    # std_spe = np.std(SPEs_val, axis=0)
    # print("RF test: AUC_mean = %f, AUC_std = %f" % (mean_auc, std_auc))
    # print("RF test: ACC_mean = %f, ACC_std = %f" % (mean_acc, std_acc))
    # print("RF test: SEN_mean = %f, SEN_std = %f" % (mean_sen, std_sen))
    # print("RF test: SPE_mean = %f, SPE_std = %f" % (mean_spe, std_spe))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.tick_params(labelsize=12)
    plt.xlabel('1-Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title('Cross-Validation ROC of RF', fontsize=14)
    plt.legend(loc="lower right", prop={'size': 10})
    plt.show()
    plt.savefig(os.path.join(save_dir, tag + '_cv_rocs.png'))

    return mean_auc

def main():
    parser = argparse.ArgumentParser(description='Plot the ROC Curve of an estimator')
    parser.add_argument('-prediction', '--prediction', metavar='prediction',
                        nargs='+', dest='prediction', type=str, required=True,
                        help='Prediction file (HDF)')
    parser.add_argument('-pinfo', '--pinfo', metavar='pinfo',
                        nargs='+', dest='pinfo', type=str, required=True,
                        help='Patient Info File (txt)')
    parser.add_argument('-ensemble_method', '--ensemble_method', metavar='ensemble_method',
                        nargs='+', dest='ensemble_method', type=str, required=True,
                        help='Method for creating ensemble (string)')
    parser.add_argument('-ensemble_size', '--ensemble_size', metavar='ensemble_size',
                        nargs='+', dest='ensemble_size', type=str, required=False,
                        help='Length of ensemble (int)')
    parser.add_argument('-label_type', '--label_type', metavar='label_type',
                        nargs='+', dest='label_type', type=str, required=True,
                        help='Label name that is predicted (string)')
    parser.add_argument('-ROC_png', '--ROC_png', metavar='ROC_png',
                        nargs='+', dest='ROC_png', type=str, required=False,
                        help='File to write ROC to (PNG)')
    parser.add_argument('-ROC_csv', '--ROC_csv', metavar='ROC_csv',
                        nargs='+', dest='ROC_csv', type=str, required=False,
                        help='File to write ROC to (csv)')
    parser.add_argument('-ROC_tex', '--ROC_tex', metavar='ROC_tex',
                        nargs='+', dest='ROC_tex', type=str, required=False,
                        help='File to write ROC to (tex)')
    parser.add_argument('-PRC_png', '--PRC_png', metavar='PRC_png',
                        nargs='+', dest='PRC_png', type=str, required=False,
                        help='File to write PR to (PNG)')
    parser.add_argument('-PRC_csv', '--PRC_csv', metavar='PRC_csv',
                        nargs='+', dest='PRC_csv', type=str, required=False,
                        help='File to write PR to (csv)')
    parser.add_argument('-PRC_tex', '--PRC_tex', metavar='PRC_tex',
                        nargs='+', dest='PRC_tex', type=str, required=False,
                        help='File to write PR to (tex)')
    args = parser.parse_args()

    plot_ROC(prediction=args.prediction,
             pinfo=args.pinfo,
             ensemble_method=args.ensemble_method,
             ensemble_size=args.ensemble_size,
             label_type=args.label_type,
             ROC_png=args.ROC_png,
             ROC_tex=args.ROC_tex,
             ROC_csv=args.ROC_csv,
             PRC_png=args.PRC_png,
             PRC_tex=args.PRC_tex,
             PRC_csv=args.PRC_csv)


def plot_ROC(prediction, pinfo, ensemble_method='top_N',
             ensemble_size=1, label_type=None,
             ROC_png=None, ROC_tex=None, ROC_csv=None,
             PRC_png=None, PRC_tex=None, PRC_csv=None):
    # Convert the inputs to the correct format
    if type(prediction) is list:
        prediction = ''.join(prediction)

    if type(pinfo) is list:
        pinfo = ''.join(pinfo)

    if type(ensemble_method) is list:
        ensemble_method = ''.join(ensemble_method)

    if type(ensemble_size) is list:
        ensemble_size = int(ensemble_size[0])

    if type(ROC_png) is list:
        ROC_png = ''.join(ROC_png)

    if type(ROC_csv) is list:
        ROC_csv = ''.join(ROC_csv)

    if type(ROC_tex) is list:
        ROC_tex = ''.join(ROC_tex)

    if type(PRC_png) is list:
        PRC_png = ''.join(PRC_png)

    if type(PRC_csv) is list:
        PRC_csv = ''.join(PRC_csv)

    if type(PRC_tex) is list:
        PRC_tex = ''.join(PRC_tex)

    if type(label_type) is list:
        label_type = ''.join(label_type)

    # Read the inputs
    prediction = pd.read_hdf(prediction)
    if label_type is None:
        # Assume we want to have the first key
        label_type = prediction.keys()[0]
    elif len(label_type.split(',')) != 1:
        # Multiclass, just take the prediction label
        label_type = prediction.keys()[0]

    N_1 = len(prediction[label_type].Y_train[0])
    N_2 = len(prediction[label_type].Y_test[0])

    # Determine the predicted score per patient
    print('Determining score per patient.')
    y_truths, y_scores, _, _ =\
        plot_estimator_performance(prediction, pinfo, [label_type],
                                   alpha=0.95, ensemble_method=ensemble_method,
                                   ensemble_size=ensemble_size,
                                   output='decision')

    # Check if we can compute confidence intervals
    config = prediction[label_type].config
    crossval_type = config['CrossValidation']['Type']

    # --------------------------------------------------------------
    # ROC Curve
    if crossval_type == 'LOO':
        print("LOO: Plotting the ROC without confidence intervals.")
        y_truths = [i[0] for i in y_truths]
        y_scores = [i[0] for i in y_scores]
        fpr, tpr, _, f = plot_single_ROC(y_truths, y_scores, returnplot=True)
    else:
        # Plot the ROC with confidence intervals
        print("Plotting the ROC with confidence intervals.")
        f, fpr, tpr = plot_ROC_CIc(y_truths, y_scores, N_1, N_2)

    # Save the outputs
    if ROC_png is not None:
        f.savefig(ROC_png)
        print(("ROC saved as {} !").format(ROC_png))

    if ROC_tex is not None:
        tikzplotlib.save(ROC_tex)
        print(("ROC saved as {} !").format(ROC_tex))

    if ROC_csv is not None:
        with open(ROC_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['FPR', 'TPR'])
            for i in range(0, len(fpr)):
                data = [str(fpr[i]), str(tpr[i])]
                writer.writerow(data)

        print(("ROC saved as {} !").format(ROC_csv))

    # --------------------------------------------------------------
    # PR Curve
    if crossval_type == 'LOO':
        tpr, precisionr, _, f = plot_single_PRC(y_truths, y_scores, returnplot=True)
    else:
        f, tpr, precisionr = plot_PRC_CIc(y_truths, y_scores, N_1, N_2)

    if PRC_png is not None:
        f.savefig(PRC_png)
        print(("PRC saved as {} !").format(PRC_png))

    if PRC_tex is not None:
        tikzplotlib.save(PRC_tex)
        print(("PRC saved as {} !").format(PRC_tex))

    if PRC_csv is not None:
        with open(PRC_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Recall', 'Precision'])
            for i in range(0, len(tpr)):
                data = [str(tpr[i]), str(precisionr[i])]
                writer.writerow(data)

        print(("PRC saved as {} !").format(PRC_csv))

    return f, fpr, tpr


if __name__ == '__main__':
    main()
