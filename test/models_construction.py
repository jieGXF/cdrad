# from.classification_base import feature_classification_process
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn_features.transformers import DataFrameSelector
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_regression
from sklearn.linear_model import LogisticRegression
def modeling(modus, train_set, test_set, save_dir, subject_name, label_name):
    if modus == 'binary_classification':
        # classifier_tags = [1, 2, 3, 4]
        cat_features = ['menses', 'MM_PA', 'cervical', 'LVI_1', 'LVI_2', 'grade']
        cat_transformer = Pipeline(steps=[('selector', DataFrameSelector(cat_features)),
                                   ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                   ('onehot', OneHotEncoder(handle_unknown='ignore')),
                                   ('fs', SelectKBest(f_regression))])
        num_features = train_set.columns.to_list() - cat_features - subject_name-label_name,
        num_transformer = Pipeline(steps=[('selector', DataFrameSelector(num_features)),
                                   ('imputer', SimpleImputer(strategy='median')),
                                   ('scaler', StandardScaler()),
                                   ('fs', SelectKBest(f_regression))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)])

        # Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(solver='lbfgs'))])

        X_train = train_set.drop([subject_name]+label_name, axis=1)
        y_train = train_set[label_name]
        X_test = test_set.drop([subject_name] + label_name, axis=1)
        y_test = test_set[label_name]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        print("model score: %.3f" % clf.score(X_test, y_test))

        # feature_classification_process(classifier_tags, train_set, test_set, save_dir, subject_name, label_name)
    elif modus == 'regression':
        pass

    elif modus == 'multiclass_classification':
        pass