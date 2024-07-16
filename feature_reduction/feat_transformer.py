import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (SelectKBest, RFE, SelectFromModel, f_regression, f_classif, mutual_info_classif,
                                       mutual_info_regression)
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, StratifiedKFold, LeaveOneOut
def cal_score(x, lasso_coef):
    return np.round(x * lasso_coef, 5)
class feat_transformer():
    def __init__(self, X_train, y_train, X_test, subject_name, label_name, reduced_moda):
        # median_value = np.median(y_train["dose"])
        # y_train[y_train >= median_value] = 1
        # y_train[y_train < median_value] = 0
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.subject_name = subject_name
        self.label_name = label_name
        self.reduced_moda = reduced_moda

    def run_transformer(self, reduction_method):
        if reduction_method == "Score":
            df_train, df_test, ft_log = self.score()
            return df_train, df_test, ft_log
            # df_train, df_test = self.score()
            # return df_train, df_test

        elif reduction_method == "PCA":
            df_train, df_test = self.pca()
            return df_train, df_test

    def pca(self):
        X_train_tmp = self.X_train.drop(self.subject_name + self.label_name, axis=1)
        X_test_tmp = self.X_test.drop(self.subject_name + self.label_name, axis=1)
        pca = PCA()
        X_train_new = pca.fit_transform(X_train_tmp)
        explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
        threshold = 0.95
        n_components = np.argmax(explained_variance_ratio_cumulative >= threshold) + 1
        pca = PCA(n_components=n_components)
        X_train_new = pca.fit_transform(X_train_tmp)
        X_train_new = pd.DataFrame(X_train_new)
        X_train_new = X_train_new.rename(columns=lambda x: self.reduced_moda + "_" + str(x))
        X_test_new = pca.transform(X_test_tmp)
        X_test_new = pd.DataFrame(X_test_new)
        X_test_new = X_test_new.rename(columns=lambda x: self.reduced_moda + "_" + str(x))
        df_train = X_train_new.set_index(self.X_train.index)
        df_test = X_test_new.set_index(self.X_test.index)
        df_train = pd.concat([self.X_train[self.subject_name + self.label_name], df_train], axis=1)
        df_test = pd.concat([self.X_test[self.subject_name + self.label_name], df_test], axis=1)
        return df_train, df_test
    def lda(self):
        pass



    def score(self):
        X_train_tmp = self.X_train.drop(self.subject_name + self.label_name, axis=1)
        X_test_tmp = self.X_test.drop(self.subject_name + self.label_name, axis=1)
        alphas = 10 ** np.linspace(- 3, 3, 100)
        moda_name = X_train_tmp.columns[1]
        lasso = LassoCV(cv=20, max_iter=10000, alphas=alphas, random_state=13).fit(X_train_tmp,
                                                                                  self.y_train.values.ravel())
        lasso_intercept = lasso.intercept_
        lasso_coef = lasso.coef_
        lasso_coef_intercept = np.append(lasso_coef, lasso_intercept)
        # pd.DataFrame(columns=X_train_tmp.columns+["intercept"], data=)
        lasso_log = pd.DataFrame(lasso_coef_intercept, index=list(X_train_tmp.columns)+["intercept"], columns=["value"])
        # lasso_log = lasso_log.round(3)
        df_train = pd.DataFrame(columns=[self.reduced_moda])
        X_train_tmp2 = X_train_tmp.apply(lambda x: cal_score(x, lasso.coef_), axis=1)
        df_train[self.reduced_moda] = X_train_tmp2.sum(axis=1) + lasso_intercept
        df_test = pd.DataFrame(columns=[self.reduced_moda])
        X_test_tmp2 = X_test_tmp.apply(lambda x: cal_score(x, lasso.coef_), axis=1)
        df_test[self.reduced_moda] = X_test_tmp2.sum(axis=1) + lasso_intercept
        df_train = df_train.set_index(self.X_train.index)
        df_test = df_test.set_index(self.X_test.index)
        df_train = pd.concat([self.X_train[self.subject_name + self.label_name], df_train], axis=1)
        df_test = pd.concat([self.X_test[self.subject_name + self.label_name], df_test], axis=1)
        return df_train, df_test, lasso_log

    # def score(self):
    #
    #     selectors = {
    #         'MI': {"selector": SelectKBest(mutual_info_regression),
    #                 "selector__k": np.arange(1, 10, 2)
    #                 },
    #         'UFS': {"selector": SelectKBest(f_regression),
    #                 "selector__k": np.arange(1, 10, 2),
    #                 },
    #         'RFE': {"selector": RFE(SVR(kernel="linear")),
    #                 "selector__n_features_to_select": np.arange(1, 10, 2),
    #                 },
    #     }
    #
    #     classifiers = {
    #         "KNN": {"regressor": KNeighborsRegressor(),
    #                 "regressor__n_neighbors": [2, 3, 4, 5, 6, 7, 8],
    #                 # "regressor__algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    #                 }
    #
    #     }
    #
    #
    #
    #     X_train_tmp = self.X_train.drop(self.subject_name + self.label_name, axis=1)
    #     X_test_tmp = self.X_test.drop(self.subject_name + self.label_name, axis=1)
    #
    #     pp = Pipeline(steps=[('selector', list(selectors['MI'].values())[0]),
    #                          ('regressor', list(classifiers["KNN"].values())[0])])
    #     search_space = dict()
    #     search_space.update(selectors['MI'])
    #     search_space.update(classifiers['KNN'])
    #     del search_space["selector"]
    #     del search_space["regressor"]
    #     grid_search = GridSearchCV(pp, search_space, cv=10,
    #                                scoring='r2')
    #
    #
    #
    #     grid_search.fit(X_train_tmp, self.y_train.values.ravel())
    #     df_train = pd.DataFrame(columns=[self.reduced_moda])
    #     df_test = pd.DataFrame(columns=[self.reduced_moda])
    #     df_train[self.reduced_moda] = grid_search.predict(X_train_tmp)
    #     df_test[self.reduced_moda] = grid_search.predict(X_test_tmp)
    #
    #     # lasso_intercept = lasso.intercept_
    #     # df_train = pd.DataFrame(columns=[self.reduced_moda])
    #     # X_train_tmp2 = X_train_tmp.apply(lambda x: cal_score(x, lasso.coef_), axis=1)
    #     # df_train[self.reduced_moda] = X_train_tmp2.sum(axis=1) + lasso_intercept
    #     # df_test = pd.DataFrame(columns=[self.reduced_moda])
    #     # X_test_tmp2 = X_test_tmp.apply(lambda x: cal_score(x, lasso.coef_), axis=1)
    #     # df_test[self.reduced_moda] = X_test_tmp2.sum(axis=1) + lasso_intercept
    #     df_train = df_train.set_index(self.X_train.index)
    #     df_test = df_test.set_index(self.X_test.index)
    #     df_train = pd.concat([self.X_train[self.subject_name + self.label_name], df_train], axis=1)
    #     df_test = pd.concat([self.X_test[self.subject_name + self.label_name], df_test], axis=1)
    #     return df_train, df_test
