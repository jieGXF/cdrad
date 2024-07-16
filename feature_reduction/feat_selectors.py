import numpy as np
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_regression, f_classif, RFECV
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC, SVR

selectors = {
                'LASSO': {"selector": SelectFromModel(LassoCV()),
                          "selector__max_features": [2, 4, 6, 8, 10]
                          },
                'RFE': {"selector": RFE(SVR(kernel="linear")),
                        "selector__n_features_to_select": [2, 4, 6, 8, 10],
                        },
                "AdaBoost": {"selector_2nd": SelectFromModel(AdaBoostRegressor(random_state=0, n_estimators=50),
                                                             threshold='mean'),
                             "selector_2nd__alphas": np.linspace(0, 0.2, 21).tolist()
                             },
}
