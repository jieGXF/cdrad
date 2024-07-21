import json
import os
import sys
import pandas as pd
from glob import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_reduction.reliability_analysis import filter_features_ICC
from feature_extraction.feature_extractor import FeatureExtractor
from feature_reduction.feature_selector import FeatureSelector
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn_features.transformers import DataFrameSelector
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_regression, f_classif, RFECV
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import RocCurveDisplay, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, auc
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC, SVR

from collections import OrderedDict
# from plotting.plot_ROC import plot_single_ROC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from preprocessing.util import DataFrameMerge
# from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib
import matplotlib.pyplot as plt
from run.train_binary_classification import train_binary_classification
from run.train_regression import train_regression
from run.train_multiclass_classification import train_multiclass_classification
from run.infer_binary_classification import infer_binary_classification
# from run.infer_regression import infer_regression
from run.infer_multiclass_classification import infer_multiclass_classification
from plotting.plot_ROC import plot_cv_roc
import seaborn as sns
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
import yaml


def train(input_path, output_path, config_path):
    with open(os.path.join(config_path, 'pora.yaml'), 'r', encoding='utf-8') as f:
        pora_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    internal_image_path = os.path.join(input_path, 'radiology')
    internal_omics_path = os.path.join(input_path, 'multiomics')
    # download_HeadAndNeck(datafolder=input, nsubjects=nsubjects)  # example data
    # TODO data checking
    subject_name = pora_config["subject_name"]
    fex_params = pora_config["fex_params"]
    label_type = pora_config["label_type"]
    label_name = list(pora_config["label_name"].keys())

    # step 3: extract different image feature
    if pora_config["is_fex"]:
        fex = FeatureExtractor()
        for img_dir in os.listdir(internal_image_path):
            params = os.path.join(config_path, fex_params[img_dir][0] + ".yaml")
            img_path = os.path.join(internal_image_path, img_dir)
            fex.feature_extractor(internal_omics_path, img_path, params)
    else:
        pass

    # reliability analysis (ICC)
    # ICC_path = os.path.join(output, 'ICC')
    ICC_internal_path = os.path.join(output_path, 'ICC', 'internal')
    maybe_mkdir_p(ICC_internal_path)
    threshold = 0.8
    img_names = os.listdir(internal_image_path)
    img_names = [s for s in img_names if 'rater' not in s]
    for img in img_names:
        all_features = []
        feat_all = pd.read_csv(os.path.join(internal_omics_path, "radiomics", img + '.csv'))
        feat_icc_path = glob(os.path.join(internal_omics_path, "radiomics", img + '_rater*'))
        feat_icc = pd.read_csv(feat_icc_path[0])
        feat_rater0 = feat_all[feat_all['ID'].isin(feat_icc["ID"].to_list())].drop(columns='ID')
        all_features.append(feat_rater0)
        for i in range(len(feat_icc_path)):
            globals()["feat_rater" + str(i+1)] = pd.read_csv(feat_icc_path[i]).drop(columns='ID')
            all_features.append(globals()["feat_rater" + str(i+1)])
        icc_idx = filter_features_ICC(all_features, csv_out=os.path.join(output_path, 'ICC', img + '_ICC.csv'),
                                      features_out=False, threshold=threshold)
        globals()[img] = feat_all.filter(items=icc_idx)
        globals()[img].to_csv(os.path.join(ICC_internal_path, img + '.csv'), index=False)

    # # step 4: sort all feature sheets according to ID
    # # TODO: check whether the IDs are consistent ?
    nonrad_list = os.listdir(internal_omics_path)
    nonrad_list = [s for s in nonrad_list if "radiomics" not in s]
    for sub_file in nonrad_list:
        df = pd.read_csv(os.path.join(internal_omics_path, sub_file))
        df = df.loc[:,  ~df.columns.str.contains("^Unnamed")]
        # df.sort_values(by="ID", ascending=True, inplace=True)#TODO: debug sorting
        df.to_csv(os.path.join(ICC_internal_path, sub_file), index=False)

    multiomics_list = os.listdir(ICC_internal_path)
    df_clinvars = pd.read_csv(os.path.join(ICC_internal_path, pora_config["label_file"]))
    for sub_file in multiomics_list:
        if sub_file != pora_config["label_file"]:
            df = pd.read_csv(os.path.join(ICC_internal_path, sub_file))
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            # df.sort_values(by="ID", ascending=True, inplace=True) #TODO: debug sorting
            if not set(df["ID"]) == set(df_clinvars["ID"]):
                print("check whether the IDs are consistent")
                sys.exit()
            for i in range(len(label_name)):
                df.insert(i + 1, label_name[i], df_clinvars[label_name[i]])
            df.to_csv(os.path.join(ICC_internal_path, sub_file), index=False)

    # train_multiclass_classification(input_path, output_path, pora_config)
    train_binary_classification(input_path, output_path, pora_config)
    # train_regression(input_path, output_path, pora_config)
    # # step 5:
    # if label_type == "enum" and len(label_name) == 1 and len(pora_config["label_name"].get(label_name[0])) == 2:
    #     train_binary_classification(input_path, output_path, pora_config)
    # elif label_type == "enum" and len(label_name) == 1 and len(pora_config["label_name"].get(label_name[0])) > 2:
    #     train_multiclass_classification(input_path, output_path, pora_config)
    # elif label_type == "enum" and len(label_name) > 1 and len(pora_config["label_name"].get(label_name[0])) != 0:
    #     modus = 'multilabel_classification'
    #     # multilabel_classification()
    # elif label_type == "quant" and len(label_name) == 1:
    #     train_regression(input_path, output_path, pora_config)
    # elif label_type == "quant" and len(label_name) > 1:
    #     modus = 'multioutput_regression'
    #     # multioutput_regression()
    # else:
    #     print("check the configuration of label_name!")


def infer(input_path, output_path, config_path):
    with open(os.path.join(config_path, 'pora.yaml'), 'r', encoding='utf-8') as f:
        pora_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    # Define the folder this script is in, so we can easily find the example data
    external_image_path = os.path.join(input_path, 'radiology')
    external_omics_path = os.path.join(input_path, 'multiomics')
    maybe_mkdir_p(output_path)
    # download_HeadAndNeck(datafolder=data_path, nsubjects=nsubjects)  # example data
    # TODO data checking
    subject_name = pora_config["subject_name"]
    fex_params = pora_config["fex_params"]
    label_type = pora_config["label_type"]
    label_name = list(pora_config["label_name"].keys())
    # step 3: extract different image feature
    if pora_config["is_fex"]:
        fex = FeatureExtractor()
        for img_dir in os.listdir(external_image_path):
            params = os.path.join(config_path, fex_params[img_dir][0] + ".yaml")
            img_path = os.path.join(external_image_path, img_dir)
            fex.feature_extractor(external_omics_path, img_path, params)
    else:
        pass

    # reliability analysis (ICC)
    ICC_path = os.path.join(output_path, 'ICC')
    ICC_external_path = os.path.join(output_path, 'ICC', 'external')
    maybe_mkdir_p(ICC_external_path)
    threshold = 0.8
    img_names = os.listdir(external_image_path)
    img_names = [s for s in img_names if 'rater' not in s]
    for img in img_names:
        icc = pd.read_csv(os.path.join(ICC_path, img + '_ICC.csv'))
        icc_idx = icc.loc[(icc["ICC"] > threshold)]
        icc_idx = icc_idx["feature_label"].to_list()
        icc_idx.insert(0, "ID")
        ex_feat_all = pd.read_csv(os.path.join(external_omics_path, 'radiomics', img + '.csv'))
        globals()[img] = ex_feat_all.filter(items=icc_idx)
        globals()[img].to_csv(os.path.join(ICC_external_path, img + '.csv'), index=False)

    # # step 4: sort all feature sheets according to ID
    # # TODO: check whether the IDs are consistent ?
    nonrad_list = os.listdir(external_omics_path)
    nonrad_list = [s for s in nonrad_list if "radiomics" not in s]
    for sub_file in nonrad_list:
        df = pd.read_csv(os.path.join(external_omics_path, sub_file))
        df.sort_values(by="ID", ascending=True, inplace=True)
        df.to_csv(os.path.join(ICC_external_path, sub_file), index=False)

    multiomics_list = os.listdir(ICC_external_path)
    df_clinvars = pd.read_csv(os.path.join(ICC_external_path, pora_config["label_file"]))
    for sub_file in multiomics_list:
        if sub_file != pora_config["label_file"]:
            df = pd.read_csv(os.path.join(ICC_external_path, sub_file))
            df.sort_values(by="ID", ascending=True, inplace=True)
            if not set(df["ID"]) == set(df_clinvars["ID"]):
                print("check whether the IDs are consistent")
                sys.exit()
            for i in range(len(label_name)):
                df.insert(i + 1, label_name[i], df_clinvars[label_name[i]])
            df.to_csv(os.path.join(ICC_external_path, sub_file), index=False)

    if label_type == "enum" and len(label_name) == 1 and len(pora_config["label_name"].get(label_name[0])) == 2:
        infer_binary_classification(ICC_path, pora_config)
    elif label_type == "enum" and len(label_name) == 1 and len(pora_config["label_name"].get(label_name[0])) > 2:
        modus = 'multiclass_classification'
        infer_multiclass_classification(ICC_path, pora_config)
    elif label_type == "enum" and len(label_name) > 1 and len(pora_config["label_name"].get(label_name[0])) != 0:
        modus = 'multilabel_classification'
        # multilabel_classification()
    elif label_type == "quant" and len(label_name) == 1:
        modus = 'regression'
        # infer_regression(ICC_path, pora_config)
    elif label_type == "quant" and len(label_name) > 1:
        modus = 'multioutput_regression'
        # multioutput_regression()
    else:
        print("check the configuration of label_name!")


if __name__ == "__main__":
    input_path = "/media/zjl/MedIA/Molecular-Imaging/GNP-Delivery-WangShouju/dataflow/DS02-Radiomics/input/internal/"
    output_path = "/media/zjl/MedIA/Molecular-Imaging/GNP-Delivery-WangShouju/dataflow/DS02-Radiomics/output/"
    config_path = "/media/zjl/MedIA/Molecular-Imaging/GNP-Delivery-WangShouju/dataflow/DS02-Radiomics/config/"
    section = "train"
    if section == "train":
        train(input_path, output_path, config_path)
    elif section == "infer":
        infer(input_path, output_path, config_path)



