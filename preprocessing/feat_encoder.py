import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# def onehot_encoder(df, subject_name, label_name):
#     df_tmp = df.drop(subject_name+label_name, axis=1)
#     df_tmp = df_tmp.astype({'tumor_cls': 'object'})
#     # df_tmp = df_tmp.astype({'Sex': 'object'})
#     categories = list()
#     for col in range(len(df_tmp.columns)):
#         if df_tmp[list(df_tmp)[col]].dtypes is object:
#             cls_list = list(df_tmp.iloc[:, col].unique())
#             cls_list.sort()
#             categories.extend([(df_tmp.columns[col], cls_list)])
#     categories = [('tumor_cls', ['0', '1', '2', '3', '4', '5', '6'])]
#     ohe_columns = [x[0] for x in categories]
#     ohe_categories = [x[1] for x in categories]
#     enc = OneHotEncoder(categories=ohe_categories)
#     # transformer = make_column_transformer((enc, ohe_columns), remainder='pass')
#     transformed = enc.fit_transform(df[ohe_columns]).toarray()
#     transformed_df = pd.DataFrame(
#         transformed,
#         columns=enc.get_feature_names_out(),
#         index=df.index
#     )
#     transformed_df = pd.concat([df.drop(ohe_columns, axis=1), transformed_df], axis=1)
#     return transformed_df

def onehot_encoder(df, subject_name, label_name, C_index):
    df_tmp = df.drop(subject_name+label_name, axis=1)
    # df_tmp = df_tmp.astype({'tumor_cls': 'object'})
    # df_tmp = df_tmp.astype({'Sex': 'object'})
    categories = list()
    for c in C_index:
        cls_list = list(df_tmp[c].unique())
        if len(cls_list) == 2:
            df[c] = LabelEncoder().fit_transform(df[c])
        else:
            cls_list.sort()
            categories.extend([(c, cls_list)])
    # categories = [('tumor_cls', ['0', '1', '2', '3', '4', '5', '6'])]
    ohe_columns = [x[0] for x in categories]
    ohe_categories = [x[1] for x in categories]
    enc = OneHotEncoder(categories=ohe_categories)
    # transformer = make_column_transformer((enc, ohe_columns), remainder='pass')
    transformed = enc.fit_transform(df[ohe_columns]).toarray()
    transformed_df = pd.DataFrame(
        transformed,
        columns=enc.get_feature_names_out(),
        index=df.index
    )
    transformed_df = pd.concat([df.drop(ohe_columns, axis=1), transformed_df], axis=1)
    return transformed_df