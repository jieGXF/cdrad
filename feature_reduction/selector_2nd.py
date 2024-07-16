from collections import OrderedDict
import numpy as np
import pandas as pd
import os

def best_feature_selection(X_train, sel, grid_search, save_dir):
    summary_each_pp = OrderedDict()
    if sel in ["MI", "UFS", "AB"]:
        feature_idx = grid_search.best_estimator_.named_steps.selector.get_support()
        feature_score = grid_search.best_estimator_.named_steps.selector.scores_[feature_idx]
        feature_name = X_train.columns.values[feature_idx]
        features = np.vstack((feature_name, feature_score))
        features = pd.DataFrame(features.transpose(), columns=['name', 'score'])
    else:
        feature_idx = grid_search.best_estimator_.named_steps.selector.get_support()
        feature_name = X_train.columns.values[feature_idx]
        if sel == "RFE":
            feature_coef = grid_search.best_estimator_.named_steps.selector.estimator_.coef_
        else:
            feature_coef = grid_search.best_estimator_.named_steps.selector.estimator_.coef_[feature_idx]
        features = np.vstack((feature_name, feature_coef))
        features = pd.DataFrame(features.transpose(), columns=['name', 'coef'])
    features.to_csv(os.path.join(save_dir, 'final_selected_features.csv'))
    return features
    # json_output = os.path.join(save_dir, "summary.json")
    # summary_each_pp["best_params"] = grid_search.best_params_