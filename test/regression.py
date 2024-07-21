import json
import os
import pandas as pd
from glob import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_reduction.reliability_analysis import filter_features_ICC
from feature_extraction.feature_extractor import FeatureExtractor
from sklearn.model_selection import train_test_split
from feature_reduction.feature_selector import FeatureSelector
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn_features.transformers import DataFrameSelector
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_regression, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import RocCurveDisplay, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC, SVR
import matplotlib
import matplotlib.pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join
from collections import OrderedDict
from plotting.plot_ROC import plot_single_ROC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from feature_reduction.selector_2nd import best_feature_selection
from sklearn.preprocessing import OneHotEncoder
from feature_reduction.feat_transformer import feat_transformer
from preprocessing.feat_encoder import onehot_encoder
from sklearn.compose import make_column_transformer
# from sklearn.calibration import calibration_curve, CalibrationDisplay
matplotlib.use('Agg')
# matplotlib.use('TkAgg')

def regression(ICC_dir, pora_config):
    # Step 1: data description
    experiment_dir = pora_config["input_dir"]
    if ('external' or 'external_cohort') in os.listdir(experiment_dir):
        external_cohort = True
    else:
        external_cohort = False
    results_dir = pora_config["output_dir"]
    maybe_mkdir_p(results_dir)

    subject_name = pora_config["subject_name"]
    label_name = list(pora_config["label_name"].keys())
    internal_path = os.path.join(experiment_dir, 'internal')
    external_path = os.path.join(experiment_dir, 'external')
    internal_image_path = os.path.join(internal_path, 'radiology')
    internal_omics_path = os.path.join(internal_path, 'multiomics')
    cv_method = pora_config["cv_method"]
    selector_1st = pora_config["selector_1st"]
    feat_transformation = pora_config["feat_transformation"]
    models = pora_config["models"]
    num_trials = pora_config["num_trials"]


    if cv_method == "Simple":
        results_summ = pd.DataFrame(columns=['seed', 'model', 'selector', 'regressor', 'train_r2', 'test_r2'])
        for seed in range(50, num_trials):
            # split according to labels
            df = pd.read_csv(os.path.join(ICC_dir, pora_config["label_file"]))
            train, test = train_test_split(df, test_size=0.3, random_state=seed)
            train = train.sort_index(axis=0)
            y_train = train[label_name]
            # y_train_counts = y_train[label_name].value_counts()
            # y_train_counts = y_train_counts.to_dict()
            test = test.sort_index(axis=0)
            y_test = test[label_name]
            # y_test_counts = y_test[label_name].value_counts()
            # y_test_counts = y_test_counts.to_dict()

            # save label counts
            summary_each_seed = OrderedDict()
            # summary_each_seed["train label counts"] = []
            # summary_each_seed["test label counts"] = []
            # train_label_count = dict()
            # test_label_count = dict()
            # for i in range(len(y_train_counts.index)):
            #     train_label_count[str(y_train_counts.index[i][0])] = int(y_train_counts[i])
            #     test_label_count[str(y_test_counts.index[i][0])] = int(y_test_counts[i])
            # summary_each_seed["train label counts"].append(train_label_count)
            # summary_each_seed["test label counts"].append(test_label_count)
            save_dir = os.path.join(results_dir, cv_method, str(seed))
            maybe_mkdir_p(save_dir)
            json_each_seed = os.path.join(results_dir, cv_method, str(seed), "summary.json")
            save_json(summary_each_seed, json_each_seed)

            # feature encoder
            moda_list = os.listdir(ICC_dir)
            moda_list = [s for s in moda_list if 'ICC' not in s]
            non_rad_list = os.listdir(internal_omics_path)
            non_rad_list.remove("radiomics")
            for moda in moda_list:
                df = pd.read_csv(os.path.join(ICC_dir, moda))
                if moda == "GNP-cls.csv":
                # if non_rad_list and moda in non_rad_list:
                    df = onehot_encoder(df, subject_name, label_name)
                    X_train = df.iloc[train.index]
                    X_test = df.iloc[test.index]
                elif moda in ["GNP-curve.csv", "GNP-size.csv"]:
                    df_tmp = df.drop(subject_name + label_name, axis=1)
                    df = pd.concat([df[subject_name + label_name], df_tmp], axis=1)
                    X_train = df.iloc[train.index]
                    X_train_tmp = X_train.drop(subject_name + label_name, axis=1)
                    scaler = MinMaxScaler()
                    X_train_tmp2 = scaler.fit_transform(X_train_tmp)
                    X_train_tmp2 = pd.DataFrame(X_train_tmp2, columns=X_train_tmp.columns)
                    X_train_tmp2.index = train.index
                    X_train = pd.concat([X_train[subject_name + label_name], X_train_tmp2], axis=1)

                    X_test = df.iloc[test.index]
                    X_test_tmp = X_test.drop(subject_name + label_name, axis=1)
                    X_test_tmp2 = scaler.transform(X_test_tmp)
                    X_test_tmp2 = pd.DataFrame(X_test_tmp2, columns=X_test_tmp.columns)
                    X_test_tmp2.index = test.index
                    X_test = pd.concat([X_test[subject_name + label_name], X_test_tmp2], axis=1)
                else:
                    df_tmp = df.drop(subject_name + label_name, axis=1)
                    df_tmp = df_tmp.rename(columns=lambda x: moda + "_" + x)
                    df = pd.concat([df[subject_name + label_name], df_tmp], axis=1)
                    X_train = df.iloc[train.index]
                    X_train_tmp = X_train.drop(subject_name + label_name, axis=1)
                    scaler = StandardScaler()
                    X_train_tmp2 = scaler.fit_transform(X_train_tmp)
                    X_train_tmp2 = pd.DataFrame(X_train_tmp2, columns=X_train_tmp.columns)
                    X_train_tmp2.index = train.index
                    X_train = pd.concat([X_train[subject_name + label_name], X_train_tmp2], axis=1)

                    X_test = df.iloc[test.index]
                    X_test_tmp = X_test.drop(subject_name + label_name, axis=1)
                    X_test_tmp2 = scaler.transform(X_test_tmp)
                    X_test_tmp2 = pd.DataFrame(X_test_tmp2, columns=X_test_tmp.columns)
                    X_test_tmp2.index = test.index
                    X_test = pd.concat([X_test[subject_name + label_name], X_test_tmp2], axis=1)

                locals()[moda[:-4] + "_train"] = X_train
                locals()[moda[:-4] + "_test"] = X_test
                if external_cohort:
                    df_ex = pd.read_csv(os.path.join(external_omics_path, moda))
                    ex_val[moda] = df_ex

            if pora_config["is_feat_transformation"]:
                for reduced_moda in feat_transformation.keys():
                    from preprocessing.util import DataFrameMerge
                    dfm = DataFrameMerge(label_name)
                    union_list = list(feat_transformation[reduced_moda].keys())[0].split('+')
                    union_train = dict()
                    union_test = dict()
                    for moda in union_list:
                        union_train[moda] = locals()[moda + "_train"]
                        union_test[moda] = locals()[moda + "_test"]
                        if external_cohort:
                            df_ex = pd.read_csv(os.path.join(external_omics_path, moda))
                            ex_val[moda] = df_ex
                    X_train, X_test = dfm.merge(union_list, union_train, union_test)
                    # X_train = X_train.drop(subject_name + label_name, axis=1)
                    # X_test = X_test.drop(subject_name + label_name, axis=1)
                    save_dir = os.path.join(results_dir, cv_method, str(seed), "feat_transformation")
                    maybe_mkdir_p(save_dir)
                    fett = feat_transformer(X_train, y_train, X_test, subject_name, label_name, reduced_moda)
                    df_train, df_test = fett.run_transformer(list(feat_transformation[reduced_moda].values())[0])
                    locals()[reduced_moda + "_train"] = df_train.set_index(train.index)
                    locals()[reduced_moda + "_test"] = df_test.set_index(test.index)
                    locals()[reduced_moda + "_train"].to_csv(os.path.join(save_dir, reduced_moda + "_train.csv"))
                    locals()[reduced_moda + "_test"].to_csv(os.path.join(save_dir, reduced_moda + "_test.csv"))

            # construct pipeline
            all_train_fpr = dict()
            all_train_tpr = dict()
            all_train_auc = dict()
            all_test_fpr = dict()
            all_test_tpr = dict()
            all_test_auc = dict()
            for model in models.keys():
                # combine features
                union_list = models[model]
                union_train = dict()
                union_test = dict()
                for moda in union_list:
                    union_train[moda] = locals()[moda + "_train"]
                    union_test[moda] = locals()[moda + "_test"]
                    if external_cohort:
                        df_ex = pd.read_csv(os.path.join(external_omics_path, moda))
                        ex_val[moda] = df_ex
                from preprocessing.util import DataFrameMerge
                dfm = DataFrameMerge(label_name)
                X_train, X_test = dfm.merge(union_list, union_train, union_test)
                X_train = X_train.drop(subject_name + label_name, axis=1)
                X_test = X_test.drop(subject_name + label_name, axis=1)

                selectors = {
                    # 'MUI': {"selector_2nd": SelectKBest(mutual_info_regression),
                    #         "selector_2nd__k": [2, 3, 4, 5, 6, 7, 8],
                    #         },
                    #
                    'UFS': {"selector_2nd": SelectKBest(f_regression),
                            "selector_2nd__k": [2, 3, 4, 5, 6, 7, 8],
                            },

                    # 'RFE':  {"selector_2nd": RFE(SVR(kernel="linear")),
                    #          "selector_2nd__n_features_to_select": [2, 3, 4, 5, 6, 7, 8],
                    #          },
                            
                }
                classifiers = {
                    # 'LR': {'classifier': LinearRegression(),
                    #         # 'classifier__C': [0.001, 0.01]
                    #        },
                    # 'SVR': {'classifier': SVR(kernel='linear'),
                    #         # 'classifier__kernel': ['linear', 'rbf'],
                    #         'classifier__C': [0.01, 0.1]
                    #         },
                    # "KNN": {"classifier": KNeighborsRegressor(),
                    #         "classifier__n_neighbors": [2, 3, 5, 8, 10],
                    #         "classifier__algorithm": ["auto",  "ball_tree", "kd_tree", "brute"]
                    #         },
                    "MLP": {"classifier": MLPRegressor(random_state=14),
                            "classifier__hidden_layer_sizes": [(10,), (10, 10)],
                             "classifier__solver": ['adam', 'sgd', 'lbfgs'],
                             "classifier__max_iter": [20]
                            }
                    # 'AdaBoost': {"classifier": AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                    #              "classifier__base_estimator__criterion": ["gini", "entropy"],
                    #              "classifier__base_estimator__splitter":  ["best", "random"],
                    #              "classifier__n_estimators": [1, 2]},
                    # "GradientBoosting": {'classifier': GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,
                    #                                                               min_samples_leaf=20,
                    #                                                               max_features='sqrt',
                    #                                                               subsample=0.8, random_state=10),
                    #                      'classifier__max_depth': range(3, 14, 2),
                    #                      'classifier__min_samples_split': range(100, 801, 200)},
                }
                if len(union_list) == 1:
                    for j, clf in enumerate(classifiers):
                        pp = Pipeline(steps=[(classifiers[clf].keys(), (classifiers[clf].values()))])
                        search_space = [clf]
                        grid_search = GridSearchCV(clf, search_space, cv=5, refit=True,
                                                   scoring='r2')
                        grid_search.fit(X_train, y_train)
                        proba_train = grid_search.predict_proba(X_train)
                        proba_test = grid_search.predict_proba(X_test)

                else:
                    for i, sel in enumerate(selectors):
                        for j, clf in enumerate(classifiers):
                            pp = Pipeline(steps=[('selector_2nd', list(selectors[sel].values())[0]),
                                                 ('classifier', list(classifiers[clf].values())[0])])
                            search_space = dict()
                            search_space.update(selectors[sel])
                            search_space.update(classifiers[clf])
                            del search_space["selector_2nd"]
                            del search_space["classifier"]
                            grid_search = GridSearchCV(pp, search_space, cv=5,
                                                       scoring='r2')
                            grid_search.fit(X_train, y_train.values.ravel())
                            fit_train = grid_search.predict(X_train)
                            fit_test = grid_search.predict(X_test)
                            train_r2 = round(r2_score(y_train, fit_train), 3)
                            test_r2 = round(r2_score(y_test, fit_test), 3)
                            print('train R2: %.3f' % r2_score(y_train, fit_train))
                            print('test R2: %.3f' % r2_score(y_test, fit_test))

                            # summarize best
                            # print('Best MAE: %.3f' % grid_search.best_score_)
                            # print('Best Config: %s' % grid_search.best_params_)
                            # TODO optimal feature selection process
                            save_dir = os.path.join(results_dir, cv_method, str(seed), "model-comparison",
                                                    model, sel, clf)
                            maybe_mkdir_p(save_dir)
                            best_feature_selection(X_train, sel, grid_search, save_dir)
                            # summary_each_pp = OrderedDict()
                            # selector_2nd = grid_search.best_estimator_.named_steps.selector_2nd.estimator_
                            # selector_2nd_idx = grid_search.best_estimator_.named_steps.selector_2nd.get_support()
                            # feature_name = X_train.columns.values[selector_2nd_idx]
                            # feature_name = pd.DataFrame(feature_name.transpose())
                            # feature_name.to_csv(os.path.join(save_dir, 'feature_name.csv'))
                            # json_output = os.path.join(save_dir, "summary.json")
                            # summary_each_pp["best_params"] = grid_search.best_params_
                            fit_train = pd.DataFrame(fit_train)
                            fit_train.to_csv(os.path.join(save_dir, 'fit_train.csv'))
                            fit_test = pd.DataFrame(fit_test)
                            fit_test.to_csv(os.path.join(save_dir, 'fit_test.csv'))

                            results_summ = pd.concat([results_summ, pd.DataFrame(
                                        {'seed': [seed], 'model': [model], 'selector': [sel], 'regressor': [clf],
                                         'train_r2': [train_r2], 'test_r2': [test_r2]})], ignore_index=True)

                            # calculate and plot metrics
                            # train/test ROCs comparison (cutoff value)
                            # train_fpr, train_tpr, thresholds = metrics.roc_curve(y_train, proba_train[:, 1],
                            #                                                      pos_label=1)
                            # train_auc = metrics.roc_auc_score(y_train, proba_train[:, 1])
                            # optimal_idx = np.argmax(train_tpr - train_fpr)
                            # optimal_threshold = thresholds[optimal_idx]
                            # all_train_fpr[model] = train_fpr
                            # all_train_tpr[model] = train_tpr
                            # all_train_auc[model] = train_auc
                            # youden = train_tpr - train_fpr
                            # cutoff = thresholds[np.argmax(youden)]

                            # test_fpr, test_tpr, _ = metrics.roc_curve(y_test, proba_test[:, 1], pos_label=1)
                            # test_auc = metrics.roc_auc_score(y_test, proba_test[:, 1])
                            # all_test_fpr[model] = test_fpr
                            # all_test_tpr[model] = test_tpr
                            # all_test_auc[model] = test_auc
                            #
                            # fig, ax = plt.subplots(figsize=(7.5, 7.5))
                            # plt.plot(train_fpr, train_tpr, label='train (AUC = %0.2f)' % train_auc)
                            # plt.plot(test_fpr, test_tpr, label='test (AUC = %0.2f)' % test_auc)
                            # plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='chance level')
                            # plt.xlim([-0.05, 1.05])
                            # plt.ylim([-0.05, 1.05])
                            # plt.xlabel('False positive rate')
                            # plt.ylabel('True positive rate')
                            # plt.legend(loc="lower right")
                            # plt.savefig(os.path.join(save_dir, 'ROCs.png'))

                            # confusion matrix
                            # train_report = classification_report(y_train, grid_search.predict(X_train),
                            #                                      output_dict=True)
                            # summary_each_pp["train_report"] = train_report
                            # # train_report = pd.DataFrame(train_report).transpose()
                            # # train_report.to_csv(os.path.join(save_dir, 'train_report.csv'))
                            # test_report = classification_report(y_test, grid_search.predict(X_test), output_dict=True)
                            # # test_report = pd.DataFrame(test_report).transpose()
                            # # test_report.to_csv(os.path.join(save_dir, 'test_report.csv'))
                            # summary_each_pp["test_report"] = test_report
                            # with open(json_output, "w") as f:
                            #     json.dump(summary_each_pp, f, sort_keys=True, indent=4, separators=(',', ':'),
                            #               cls=NpEncoder)
                            # train_cm = confusion_matrix(y_true=y_train, y_pred=grid_search.predict(X_train))
                            # vis = ConfusionMatrixDisplay(confusion_matrix=train_cm)
                            # vis.plot()
                            # plt.savefig(os.path.join(save_dir, 'train_confusion_matrix.png'))
                            # test_cm = confusion_matrix(y_true=y_test, y_pred=grid_search.predict(X_test))
                            # vis = ConfusionMatrixDisplay(confusion_matrix=test_cm)
                            # vis.plot()
                            # plt.savefig(os.path.join(save_dir, 'test_confusion_matrix.png'))

            # ROCs comparison of all models
            # fig, ax = plt.subplots(figsize=(7.5, 7.5))
            # for i in models.keys():
            #     plt.plot(all_train_fpr[i], all_train_tpr[i], label=i + '(AUC = %0.2f)' % all_train_auc[i])
            # plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='chance level')
            # plt.xlim([-0.05, 1.05])
            # plt.ylim([-0.05, 1.05])
            # plt.xlabel('False positive rate')
            # plt.ylabel('True positive rate')
            # plt.legend(loc="lower right")
            # plt.savefig(os.path.join(results_dir, cv_method, str(seed),
            #                          "model-comparison", 'train_ROCs.png'))
            # fig, ax = plt.subplots(figsize=(7.5, 7.5))
            # for i in models.keys():
            #     plt.plot(all_test_fpr[i], all_test_tpr[i], label=i + '(AUC = %0.2f)' % all_test_auc[i])
            # plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='chance level')
            # plt.xlim([-0.05, 1.05])
            # plt.ylim([-0.05, 1.05])
            # plt.xlabel('False positive rate')
            # plt.ylabel('True positive rate')
            # plt.legend(loc="lower right")
            # plt.savefig(os.path.join(results_dir, cv_method, str(seed),
            #                          "model-comparison", 'test_ROCs.png'))

        # step 9: summarize the AUCs of target models with different seed and their DeLong tests
        results_summ.to_csv(os.path.join(results_dir, cv_method, 'all_seeds_r2.csv'), sep=",", index=False)
        # best selector
        # TODO record information

    elif cv_method == "K-Fold":

        # TODO: onehotEncoder
        # cat_transformer.fit(X_train, y_train)
        # num_features = X_train_tm.columns.values.tolist()
        # num_features = [n for n in num_features if n not in cat_features + subject_name + label_name]

        # TODO: feature reduction
        # feat_Tx = dict()
        # for moda in selector_1st.keys():
        #     if selector_1st[moda][1] in ["f_classif", "mutual_info_classif", "chi2", "ANOVA F-value"]:
        #         locals()[moda+"selector_1st"] = SelectKBest(selector_1st[moda][1])
        #     else:
        #         locals()[moda + "selector_1st"] = SelectFromModel(locals()[selector_1st[moda][1]](cv=5))
        #
        #     feat_Tx[moda + "selector_1st"] = Pipeline(steps=[('selector', DataFrameSelector(rad_features)),
        #                                              ('scaler', StandardScaler()),
        #                                              ('selector_1st', rad_selector_1st)])
        selectors = {
            'LASSO': {"selector_2nd": SelectFromModel(Lasso()),
                      # "selector_2nd__alphas": np.linspace(0, 0.2, 21).tolist()
                      },
            # 'RFE': {"selector_2nd": RFE(SVR(kernel="linear"))},
        }
        classifiers = {
            # 'LogisticRegression': {'classifier': LogisticRegression(solver='lbfgs'),
            #                        'classifier__C': [0.1, 1.0, 10, 100]
            #                       },
            # 'SupportVectorMachine': {'classifier': SVC(),
            #                          'classifier__kernel': ['linear'],
            #                          'classifier__gamma': [1e-3, 1e-4]},
            "DecisionTree": {"classifier": DecisionTreeClassifier(criterion="entropy", max_depth=15),
                             },
            "RandomForest": {"classifier": RandomForestClassifier(random_state=14),
                             "classifier__n_estimators": [10, 15, 20],
                             "classifier__criterion": ["gini", "entropy"],
                             "classifier__min_samples_leaf": [2, 4, 6]},
            'AdaBoost': {"classifier": AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                         "classifier__base_estimator__criterion": ["gini", "entropy"],
                         "classifier__base_estimator__splitter": ["best", "random"],
                         "classifier__n_estimators": [1, 2]},
            "GradientBoosting": {'classifier': GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,
                                                                          min_samples_leaf=20,
                                                                          max_features='sqrt',
                                                                          subsample=0.8, random_state=10),
                                 'classifier__max_depth': range(3, 14, 2),
                                 'classifier__min_samples_split': range(100, 801, 200)},
        }

        for model in models.keys():
            union_list = models[model]
            union = dict()
            for moda in union_list:
                union[moda] = pd.read_csv(os.path.join(internal_omics_path, moda+".csv"))
                if external_cohort:
                    df_ex = pd.read_csv(os.path.join(external_omics_path, moda))
                    ex_val[moda] = df_ex

            dfm = DataFrameMerge()
            union_data = dfm.merge2(union_list, union)
            X = union_data.drop(subject_name + label_name, axis=1)
            y = union_data[label_name]
            for i, sel in enumerate(selectors):
                for j, clf in enumerate(classifiers):
                    pp = Pipeline(steps=[('selector', list(selectors[sel].values())[0]),
                                         ('classifier', list(classifiers[clf].values())[0])])
                    s_folder = StratifiedKFold(n_splits=5, random_state=0)
                    search_space = dict()
                    search_space.update(selectors[sel])
                    search_space.update(classifiers[clf])
                    del search_space["selector_2nd"]
                    del search_space["classifier"]
                    for train_idx, test_idx in s_folder.split(X, y):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        grid_search = GridSearchCV(pp, search_space, cv=5, refit=True, scoring='accuracy')
                        grid_search.fit(X_train, y_train)
                        proba_train = grid_search.predict_proba(X_train)
                        proba_test = grid_search.predict_proba(X_test)


        # X_tmp = num_transformer.fit_transform(X_train, y_train)
        full_pipeline = FeatureUnion([("clin_pipeline", clin_transformer),
                                      ("rad_pipeline", rad_transformer)])

        # X_features = full_pipeline.fit(X_train, y_train).transform(X_train)
        # Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', full_pipeline),
                              ('selector_2nd', SelectFromModel(LassoCV(cv=5))),
                              ('classifier', RandomForestRegressor())])
        # for sub_file in os.listdir(internal_omics_path):  # determine the best feature selector for each dataset
        #     for n in fold:
        #         # step 6: data normalization
        #         df = pd.read_csv(os.path.join(internal_omics_path, sub_file))
        #         X_train = df[train_idx]
        #         X_test = df[test_idx]
        #         std_scale = StandardScaler().fit(X_train)
        #         X_train_std = std_scale.transform(X_train)
        #         X_test_std = std_scale.transform(X_test)
        #         if external_cohort:
        #             df_ex = pd.read_csv(os.path.join(external_omics_path, sub_file))
        #             X_ex_std = std_scale.transform(df_ex)
        #
        #         # step 7: feature selection
        #         idx, X_train_selected = feature_selection(X_train_std, y_train)  # best selector
        #         # TODO record information
        #         # non-clinical variables can be converted
        #
        # target_inclusion_data =
        # idx, X_train_selected = feature_selection(X_train_std, y_train)
        # contrast_inclusion_data_1 =
        # idx, X_train_selected = feature_selection(X_train_std, y_train)  # feature selection again
        # for n in fold:
        #     target_model = model_construct(target_inclusion_data, label_name, modus)
        #     contrast_model_1 = model_construct(contrast_inclusion_data_1, label_name, modus)
        #
        #     train_result = target_model.fit(X_train_selected)
        #     test_result = target_model.fit(X_test_selected)
        #     train_result = contrast_model_1.fit(X_train_selected)
        #     test_result = contrast_model_1.fit(X_test_selected)

    elif cv_method == "Leave-One-Out":

        # TODO: onehotEncoder
        # cat_transformer.fit(X_train, y_train)
        # num_features = X_train_tm.columns.values.tolist()
        # num_features = [n for n in num_features if n not in cat_features + subject_name + label_name]

        # TODO: feature reduction
        # feat_Tx = dict()
        # for moda in selector_1st.keys():
        #     if selector_1st[moda][1] in ["f_classif", "mutual_info_classif", "chi2", "ANOVA F-value"]:
        #         locals()[moda+"selector_1st"] = SelectKBest(selector_1st[moda][1])
        #     else:
        #         locals()[moda + "selector_1st"] = SelectFromModel(locals()[selector_1st[moda][1]](cv=5))
        #
        #     feat_Tx[moda + "selector_1st"] = Pipeline(steps=[('selector', DataFrameSelector(rad_features)),
        #                                              ('scaler', StandardScaler()),
        #                                              ('selector_1st', rad_selector_1st)])
        selectors = {
            'LASSO': {"selector_2nd": SelectFromModel(Lasso()),
                      # "selector_2nd__alphas": np.linspace(0, 0.2, 21).tolist()
                      },
            # 'RFE': {"selector_2nd": RFE(SVR(kernel="linear"))},
        }
        classifiers = {
            # 'LogisticRegression': {'classifier': LogisticRegression(solver='lbfgs'),
            #                        'classifier__C': [0.1, 1.0, 10, 100]
            #                       },
            # 'SupportVectorMachine': {'classifier': SVC(),
            #                          'classifier__kernel': ['linear'],
            #                          'classifier__gamma': [1e-3, 1e-4]},
            "DecisionTree": {"classifier": DecisionTreeClassifier(criterion="entropy", max_depth=15),
                             },
            "RandomForest": {"classifier": RandomForestClassifier(random_state=14),
                             "classifier__n_estimators": [10, 15, 20],
                             "classifier__criterion": ["gini", "entropy"],
                             "classifier__min_samples_leaf": [2, 4, 6]},
            'AdaBoost': {"classifier": AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                         "classifier__base_estimator__criterion": ["gini", "entropy"],
                         "classifier__base_estimator__splitter": ["best", "random"],
                         "classifier__n_estimators": [1, 2]},
            "GradientBoosting": {'classifier': GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,
                                                                          min_samples_leaf=20,
                                                                          max_features='sqrt',
                                                                          subsample=0.8, random_state=10),
                                 'classifier__max_depth': range(3, 14, 2),
                                 'classifier__min_samples_split': range(100, 801, 200)},
        }

        for model in models.keys():
            union_list = models[model]
            union = dict()
            for moda in union_list:
                union[moda] = pd.read_csv(os.path.join(internal_omics_path, moda+".csv"))
                if external_cohort:
                    df_ex = pd.read_csv(os.path.join(external_omics_path, moda))
                    ex_val[moda] = df_ex

            dfm = DataFrameMerge()
            union_data = dfm.merge2(union_list, union)
            X = union_data.drop(subject_name + label_name, axis=1)
            y = union_data[label_name]
            for i, sel in enumerate(selectors):
                for j, clf in enumerate(classifiers):
                    pp = Pipeline(steps=[('selector', list(selectors[sel].values())[0]),
                                         ('classifier', list(classifiers[clf].values())[0])])
                    l_folder = LeaveOneOut()
                    search_space = dict()
                    search_space.update(selectors[sel])
                    search_space.update(classifiers[clf])
                    del search_space["selector_2nd"]
                    del search_space["classifier"]
                    for train_idx, test_idx in l_folder.split(X, y):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        grid_search = GridSearchCV(pp, search_space, cv=5, refit=True, scoring='accuracy')
                        grid_search.fit(X_train, y_train)
                        proba_train = grid_search.predict_proba(X_train)
                        proba_test = grid_search.predict_proba(X_test)
