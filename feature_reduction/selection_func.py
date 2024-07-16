from shutil import which

from scipy.stats import shapiro, mannwhitneyu, levene, ttest_ind
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
# import pymrmr
# pip install git+https://github.com/smazzanti/mrmr
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
# pip install boruta
import pandas as pd
# from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle
import sklearn_relief as relief
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
import os
from batchgenerators.utilities.file_and_folder_operations import *

np.random.seed(12345)

def chi2_test_selector(feature, y_train, kwargs):
    """
    :param feature:
    :param label:
    :return:
    """
    # print('please input the number of feature to keep:')
    # num_feature = input()
    # ValueError: Input X must be non-negative.
    if 'n_feature' in kwargs.keys():
        num_feature = int(kwargs['n_feature'])
    elif 'p_feature' in kwargs.keys():
        num_feature = int(float(kwargs['p_feature']) * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting
    # scaler = MinMaxScaler()
    # new_feature = scaler.fit_transform(feature)
    new_feature = feature
    selector = SelectKBest(chi2, k=num_feature).fit(new_feature, y_train)
    # chi2_statistics,p_value = chi2(new_feature,label)
    support = selector.get_support()
    return support


def anova_f_value_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = int(kwargs['n_feature'])
    elif 'p_feature' in kwargs.keys():
        num_feature = int(float(kwargs['p_feature']) * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting

    # print('please input the number of feature to keep:')
    # num_feature = input()
    selector = SelectKBest(f_classif, k=num_feature).fit(feature, label)
    support = selector.get_support()
    return support


def mutual_information_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = int(kwargs['n_feature'])
    elif 'p_feature' in kwargs.keys():
        num_feature = int(float(kwargs['p_feature']) * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting

    selector = SelectKBest(mutual_info_classif, k=num_feature).fit(feature, label)
    support = selector.get_support()
    return support


def mRMR_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = int(kwargs['n_feature'])
    elif 'p_feature' in kwargs.keys():
        num_feature = int(float(kwargs['p_feature']) * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting

    selected_features = mrmr_classif(feature, label, K=num_feature)
    # convert the feature index to support
    support = np.array([False] * feature.shape[1])
    support[selected_features] = True
    return support


def refief_selector(feature, label, save_dir, kwargs):
    """

    :param feature:
    :param label:
    :param kwargs:
    :return:
    """
    if 'n_feature' in kwargs.keys():
        num_feature = int(kwargs['n_feature'])
    elif 'p_feature' in kwargs.keys():
        num_feature = int(float(kwargs['p_feature']) * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting

    selector = relief.Relief().fit(feature, label)
    weight = selector.w_
    weight_sorted = np.flip(np.sort(selector.w_))
    weight_threshold = weight_sorted[num_feature]
    support = weight > weight_threshold
    return support


def RFE_selector(feature, label, save_dir, kwargs):
    """
     recursive feature elimination  with automatic tuning of the number of features selected with cross-validation
    :param feature:
    :param label:
    :param kwargs: the classifier should have `coef_` or `feature_importances_` attribute.
    :return:
    """
    classifier = SVC(kernel="linear")
    min_features_to_select = 1
    rfecv = RFECV(classifier, step=1, cv=StratifiedKFold(5), scoring='accuracy',
                  min_features_to_select=min_features_to_select)
    rfecv.fit(feature, label)
    # save the cross-validation scores for plot self
    print("Optimal number of features : %d" % rfecv.n_features_)
    # plot number of features vs. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)

    plt.axvline(x=np.argmax(rfecv.grid_scores_), linestyle='dashed', c='black', lw=2)
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'recursive_feature_elimination_process.png'))
    support = rfecv.get_support()
    return support


def lasso_selector(feature, label, save_dir, kwargs):
    """
    #Todo
    :param feature:
    :param label:
    :return:
    """
    feature_name = kwargs['feature_name']
    if feature_name is None:
        feature_name = ['feature_{}'.format(i) for i in range(feature.shape[1])]
    # classifier = LassoCV(cv=5, random_state=32).fit(feature, label.values.ravel())
    # _, coef_path, _ = classifier.path(feature, label)
    # lasso_coef_positive = classifier.coef_[classifier.coef_ > 0]
    # alphas = classifier.alphas_

    # plot criterion of cv
    # criterion_all = classifier.mse_path_
    # criterion_min = criterion_all.min(axis=1)
    # criterion_max = criterion_all.max(axis=1)
    # criterion_mean = criterion_all.mean(axis=1)
    # log_alphas_lasso = np.log10(alphas)
    # log_alpha = np.log10(classifier.alpha_)
    # fig, ax = plt.subplots(figsize=(8, 6))
    # yerr = [np.subtract(criterion_mean, criterion_min), np.subtract(criterion_max, criterion_mean)]
    # ax.errorbar(log_alphas_lasso, criterion_mean, yerr=yerr, capsize=5, c='black', linewidth=1)
    # ax.scatter(log_alphas_lasso, criterion_mean, c='red', marker='o')
    # ymin, ymax = plt.ylim()
    # ax.axvline(x=log_alpha, linestyle='dashed', c='red')
    # ax.set_xlabel('Log Lambda')
    # ax.set_ylabel('Criterion')
    # plt.savefig(os.path.join(save_dir, 'lasso_loss_plot.png'))

    # plt.show()
    # plot coefficient
    # plt.figure(figsize=(8, 6))
    # # ymin, ymax = plt.ylim()
    # for i in range(coef_path.shape[0]):
    #     plt.plot(log_alphas_lasso, coef_path[i, :])
    # plt.axvline(x=log_alpha, linestyle='dashed', c='red')
    # plt.xlabel('Log Lambda')
    # plt.ylabel('Coefficient')
    # plt.savefig(os.path.join(save_dir, 'lasso_coefficient_plot.png'))
    # plt.show()
    # alpha_index = np.where(alphas == classifier.alpha_)[0][0]
    # coefficient = coef_path[:, alpha_index]
    # coef_df = pd.DataFrame(coefficient[np.newaxis], columns=feature_name)
    # coef_df = pd.DataFrame(coefficient, columns=feature_name)
    # coef_df.to_excel(os.path.join(save_dir, 'lasso_coefficient.xlsx'))
    # support = coefficient != 0
    selector = SelectFromModel(LassoCV(), max_features=7).fit(feature, label.ravel())
    support = selector.get_support()

    # support = classifier.coef_ != 0
    # new_name =
    # coef_no_zero = coefficient[support]
    # coef_no_zero_name = feature_name[support]
    # x_ticks = coef_no_zero_name
    # heights = coef_no_zero
    # x_pos = [i for i, _ in enumerate(x_ticks)]
    # plt.figure()
    # plt.bar(x_pos, heights, color='gold')
    # plt.xlabel("feature_name")
    # plt.ylabel("importance")
    # plt.title("feature importance after lasso")
    # plt.xticks(x_pos, x_ticks, fontsize=9, rotation='45')
    # plt.savefig(os.path.join(save_dir, 'lasso_feature_importance.png'))
    return support


def RF_selector(feature, label, save_dir, kwargs):
    if 'n_feature' in kwargs.keys():
        num_feature = int(kwargs['n_feature'])
    elif 'p_feature' in kwargs.keys():
        num_feature = int(float(kwargs['p_feature']) * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting
    feature_name = kwargs['feature_name']
    forest = RandomForestClassifier(random_state=0)
    forest.fit(feature, label)
    importances = forest.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in forest.estimators_], axis=0)
    if feature_name is None:
        feature_name = ['feature_{}'.format(i) for i in range(feature.shape[1])]

    forest_importances = pd.Series(importances, index=feature_name)  # for plot
    forest_importances.to_excel(os.path.join(save_dir, 'feature_importance_with_random_forest.xlsx'))
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_with_random_forest.png'))

    importances_sorted = np.sort(importances)[::-1]  # sorted max-> min
    importances_threshold = importances_sorted[num_feature]  # get the threshold for filtering
    support = importances >= importances_threshold
    return support


def SFS_selector(feature, label, save_dir, kwargs):
    print("========")
    print(kwargs)
    if 'n_feature' in kwargs.keys():
        num_feature = int(kwargs['n_feature'])
    elif 'p_feature' in kwargs.keys():
        num_feature = int(float(kwargs['p_feature']) * feature.shape[1])
    else:
        num_feature = feature.shape[1] // 2  # default setting
    direction = kwargs['direction']
    if direction == None:
        direction = 'forward'
    classifier_tag = kwargs['classifier']
    if classifier_tag == 'lasso':
        classifier = LassoCV()
    elif classifier_tag == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=3)
    elif classifier_tag == 'svm':
        classifier = SVC()
    elif classifier_tag == 'random_forest':
        classifier = RandomForestClassifier(random_state=0)

    sfs_selector = SequentialFeatureSelector(classifier, n_features_to_select=num_feature, direction=direction).fit(
        feature, label)
    support = sfs_selector.get_support()

    return support


def Boruta_selector(feature, label, save_dir, kwargs):
    forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(feature, label)
    support = feat_selector.support_
    return support


def T_MWU_selector(feature, label, save_dir, kwargs):
    feature = pd.DataFrame(feature, columns=kwargs["feature_name"])
    label = pd.DataFrame(label, columns=[kwargs["label_column_name"]])
    data = pd.concat([label, feature], axis=1)

    data_pat = data.loc[data[kwargs['label_column_name']] == kwargs["pat_label"], :]
    data_sub = data.loc[data[kwargs['label_column_name']] == kwargs["con_label"], :]

    index_lists = []
    for index, colName in enumerate(data.columns[1:]):
        if shapiro(data_pat[colName]).pvalue > 0.05 and shapiro(data_sub[colName]).pvalue > 0.05:  # 检查是否为正态分布，大于0.05表示符合正态分布，参数检验
            if levene(data_pat[colName], data_sub[colName])[1] > 0.05:  # levene 是否有方差齐性,大于0.05表示方差齐性
                if ttest_ind(data_pat[colName], data_sub[colName])[1] < 0.05:
                    index_lists.append(index)
            else:  # 方差非齐，需要校正再ttest
                if ttest_ind(data_pat[colName], data_sub[colName], equal_var=False)[1] < 0.05:
                    index_lists.append(index)
        else:
            # 非正态分布，非参数检验
            _, p = mannwhitneyu(data_pat[colName], data_sub[colName])
            if p < 0.05:
                index_lists.append(index)
    support = np.zeros(feature.shape[1], dtype=np.bool)
    for i in index_lists:
        support[i] = True
    return support

def Correlation_selector(feature, label, save_dir, kwargs):
    feature = pd.DataFrame(feature, columns=kwargs["feature_name"])
    label = pd.DataFrame(label, columns=[kwargs["label_column_name"]])
    data = pd.concat([label, feature], axis=1)
    print(data)

    norm_result = feature.apply(lambda x: shapiro(x).pvalue)
    # print(norm_result)
    # print(norm_result>=0.05)
    # print(np.where(norm_result>=0.05))
    norm_names = []
    non_norm_names = []
    for i,p in enumerate(norm_result):
        # print(norm_result.index[i], p)
        if p>=0.05:
            norm_names.append(norm_result.index[i])
        else:
            non_norm_names.append(norm_result.index[i])
    # print(norm_names)
    # print(non_norm_names)
    norm_features = feature[norm_names]
    # print(norm_features)
    non_norm_names = feature[non_norm_names]
    # print(non_norm_names)
    feature_new = pd.concat([norm_features, non_norm_names],axis=1)
    # print(feature_new)
    cor_nor = norm_features.corr(method="pearson")
    # print(cor_nor)
    cor_all = feature_new.corr(method="spearman")
    # print(cor_all)
    num_nor = cor_nor.shape[1]
    # print(num_nor)
    cor_all.iloc[0:num_nor,0:num_nor] = cor_nor

    # ###### condition1——tingfan-R #####
    # colNames = cor_all.columns
    # print(cor_all)
    # final_cor_all = pd.DataFrame(np.tril(cor_all, -1), columns=colNames, index=colNames)
    # print(final_cor_all)
    # final_features = []
    # for i in range(0, len(final_cor_all)):
    #     corr_col = final_cor_all.iloc[:, i]
    #     abs_corr_col = abs(corr_col)
    #     if np.max(abs_corr_col)>=0.75:
    #         continue
    #     else:
    #         final_features.append(final_cor_all.columns[i])
    # print(final_features)
    # ##### #####

    ##### condition2——reference mean absolute correlation
    final_cor_all = cor_all
    final_cor_all.values[[np.arange(final_cor_all.shape[0])]*2] = 0
    print(final_cor_all)
    final_features = []
    for i in range(0, len(final_cor_all)):
        final_corr_col = final_cor_all.iloc[:, i]
        abs_corr_col = abs(final_corr_col)
        if np.max(abs_corr_col) >= 0.75:
            max_loc = np.argmax(abs_corr_col)
            print(max_loc, np.max(abs_corr_col), abs_corr_col[max_loc], abs_corr_col.index[max_loc])
            feature_1_name = abs_corr_col.index[i]
            feature_2_name = abs_corr_col.index[max_loc]
            print(feature_1_name, feature_2_name)
            feature_1 = abs(cor_all.iloc[:, i])
            feature_2 = abs(cor_all.iloc[:, max_loc])
            mean_feature_1 = np.mean(feature_1)
            mean_feature_2 = np.mean(feature_2)
            print(mean_feature_1, mean_feature_2)
            if mean_feature_1 >= mean_feature_2:
                final_features.append(feature_2_name)
            else:
                final_features.append(feature_1_name)
        else:
            print("max: {}, {} no over 0.75".format(np.max(abs_corr_col),abs_corr_col.index[i]))
            final_features.append(abs_corr_col.index[i])
    final_features = list(set(final_features))

    print(kwargs["feature_name"])
    support = np.zeros(feature.shape[1], dtype=np.bool)
    for ind in final_features:
        i = kwargs["feature_name"].tolist().index(ind)
        support[i] = True
    return support

def VarianceThreshold_selector(feature, label, save_dir, kwargs):
    selector = VarianceThreshold(1e-10)
    selector.fit_transform((feature))
    support = selector.get_support()
    return support