import os
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from nested_cv.nested_cv import NestedCV
from nested_cv.non_nested_cv import NonNestedCV

# def feature_classification_process(classifier_tags, data_obj, do_feature_selection, feature_selection_methods,
#                                    save_dir, subj_column_name, label_column_name, pat_label, con_label, document):
def feature_classification_process(classifier_tags, train_set, test_set, save_dir, subject_name, label_name):
    classifier_dict = {
        0: 'LogsticRegression',
        1: 'GaussianProcessClassification',
        2: 'SVM',
        3: 'SGDClassifier',
        4: 'KNeighborsNearest',
        5: 'DecisionTree',
        6: 'RandomForest',
        7: 'Adaboost',
        8: 'Naive Bayes',
        9: 'Quadratic Discriminant Analysis',
        10: 'LinearDiscriminantAnalysis',
        11: 'Xgboost'
    }

    columns = []
    save_dir_base = save_dir
    for classifier_tag in classifier_tags:
        classification_method = classifier_dict[classifier_tag]
        columns.append(classification_method)
        save_dir = os.path.join(save_dir_base, classification_method)
        os.makedirs(save_dir, exist_ok=True)
        if classifier_tag == 3:  # stochastic gradient descent learning
            param_grid = {
                'loss': ['log', 'modified_huber'],
                # 在特征选择选择1,p_feature,0.5/4,p_feature,0.5/8时，运行该方法可能会报probability estimates are not available for loss='perceptron'的错误,所以去掉perceptron参数
                'penalty': ['l2', 'l1', 'elasticnet'],
                'average': [True, False],
                'l1_ratio': np.linspace(0, 1, num=10),
                'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
            if nested_cv:
                clf = NestedCV(model=SGDClassifier(), params_grid=param_grid,
                               outer_cv=3, inner_cv=2,
                               data_obj=train_set, feature_selection_methods=feature_selection_methods, save_dir=save_dir,
                               subj_column_name=subj_column_name, label_column_name=label_column_name, pat_label=pat_label, con_label=con_label,
                               document=document,
                               cv_options={'metric': roc_auc_score,
                                           'metric_score_indicator_lower': False,
                                           'do_feature_selection': do_feature_selection,
                                           'randomized_search_iter': 30,
                                           'predict_proba': True})
            else:
                inner_cv = KFold(n_splits=4, shuffle=True, random_state=123)
                clf = NonNestedCV(model=SGDClassifier(), params_grid=param_grid, inner_cv)
            clf.fit(X=train_set, y=data_obj.labels)
            clf.fit(X=test_set, y=data_obj.labels)

        elif classifier_tag == 6:
            # Define a parameters grid
            param_grid = {
                'max_depth': [3, None],
                'n_estimators': np.arange(1, 40, 10)
            }

            NCV = NestedCV(model=RandomForestClassifier(), params_grid=param_grid,
                           outer_cv=3, inner_cv=2,

                           data_obj=data_obj,feature_selection_methods=feature_selection_methods, save_dir=save_dir,
                           subj_column_name=subj_column_name, label_column_name=label_column_name, pat_label=pat_label, con_label=con_label,

                           cv_options={'metric': roc_auc_score,
                                       'metric_score_indicator_lower': False,
                                       'do_feature_selection': do_feature_selection,
                                       'randomized_search_iter': 30,
                                       'predict_proba': True})
            NCV.fit(X=data_obj.data, y=data_obj.labels)

            print(NCV.outer_scores)
            print(NCV.best_params)


        elif classifier_tag == 7:  # adaboost
            param_grid = {
                "n_estimators": np.arange(5, 210, 20),
                "algorithm": ['SAMME.R', "SAMME"],
                "learning_rate": np.linspace(0.01, 1, 10)
            }

            NCV = NestedCV(model=AdaBoostClassifier(), params_grid=param_grid,
                           outer_cv=3, inner_cv=2,

                           data_obj=data_obj,feature_selection_methods=feature_selection_methods, save_dir=save_dir,
                           subj_column_name=subj_column_name, label_column_name=label_column_name, pat_label=pat_label, con_label=con_label,

                           cv_options={'metric': roc_auc_score,
                                       'metric_score_indicator_lower': False,
                                       'do_feature_selection': do_feature_selection,
                                       'randomized_search_iter': 30,
                                       'predict_proba': True})
            NCV.fit(X=data_obj.data, y=data_obj.labels)
