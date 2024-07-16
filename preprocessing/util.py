import os
import pandas as pd
import numpy as np
def load_data_base(fileppath, sheetname=None):
    if os.path.isfile(fileppath):
        # Load data
        if fileppath.endswith('.csv'):
            data = pd.read_csv(fileppath)
        elif fileppath.endswith('.xlsx'):
            data = pd.read_excel(fileppath, sheet_name=sheetname, engine="openpyxl")
            # https://blog.csdn.net/qq_34101364/article/details/110498654 参考
        elif fileppath.endswith('.xls'):
            data = pd.read_excel(fileppath, sheet_name=sheetname)
        else:
            raise ValueError('%s extension is not supported now. Please use csv or xlsx extension.'
                             % fileppath.split('.')[-1])
    else:
        raise ValueError("%s is not an exact file" % format(fileppath))

    return data

class DataFrameMerge(object):
    def __init__(self, label_name):
        self.label_name = label_name

    def df_merge(self, inclusion_data, trainset, testset):
        if inclusion_data['Clinic']:
            trainset["Clinic"], testset["Clinic"] = self.merge(inclusion_data['Clinic'], trainset, testset)
        if inclusion_data['Radiology']:
            trainset["Radiology"], testset["Radiology"] = self.merge(inclusion_data['Radiology'], trainset, testset)
        # if inclusion_data['Gene']:
        #     trainset["Gene"], testset["Gene"] = self.merge(inclusion_data['Gene'], trainset, testset)
        # if inclusion_data['Protein']:
        #     trainset["Protein"], testset["Protein"] = self.merge(inclusion_data['Protein'], trainset, testset)
        return trainset, testset

    def df_merge2(self, trainset, testset):
        inclusion_data = list(trainset.keys())
        trainset, testset = self.merge(inclusion_data, trainset, testset)
        return trainset, testset

    # def merge(self, inclusion_list, trainset, testset):
    #     n = 0
    #     X_train = pd.DataFrame()
    #     X_test = pd.DataFrame()
    #     inclusion_list.sort()
    #     for i in inclusion_list:
    #         if n == 0:
    #             # if trainset.get(i).shape[1] == len(['ID']+self.label_name):
    #             #     continue
    #             X_train = trainset.get(i)
    #             train_index = X_train.index
    #             X_test = testset.get(i)
    #             test_index = X_test.index
    #             # trainset.pop(i, None)
    #             # testset.pop(i, None)
    #         else:
    #             if trainset.get(i).shape[1] == len(['ID']+self.label_name):
    #                 continue
    #             X_train = pd.merge(X_train, trainset.get(i), on=['ID'].append(self.label_name))
    #             X_test = pd.merge(X_test, testset.get(i), on=['ID'].append(self.label_name))
    #             # trainset.pop(i, None)
    #             # testset.pop(i, None)
    #         n = n+1
    #     X_train.index = train_index
    #     X_test.index = test_index
    #
    #     return X_train, X_test
    def merge(self, inclusion_list, dataset):
            n = 0
            X = pd.DataFrame()
            inclusion_list.sort()
            for i in inclusion_list:
                if n == 0:
                    X = dataset.get(i)
                    index = X.index
                else:
                    if dataset.get(i).shape[1] == len(['ID']+self.label_name):
                        continue
                    X = pd.merge(X, dataset.get(i), on=['ID'].append(self.label_name))
                n = n+1
            X.index = index
            return X

## 区分DataFrame里面的数值变量和离散变量
## DataFrame_data：待处理的DataFrame类型的变量
## O_index:数值型变量列名
## C_index：离散型变量的列名
def distinguish_Char_Num(DataFrame_data):
    import copy
    m, n = DataFrame_data.shape
    ## 存放数值型变量所在的列
    O = []
    ## 存放离散型变量所在的列
    C = []
    data = copy.deepcopy(DataFrame_data)
    for i in range(n):
        try:
            if isinstance(data.iloc[0, i], int) or isinstance(data.iloc[0, i], float) or isinstance(data.iloc[0, i], np.int64 ) or isinstance(data.iloc[0, i], np.int32):
                O.append(i)
            elif isinstance(data.iloc[0, i], str):
                C.append(i)
            else:
                raise ValueError("the %d column of data is not a number or a string column" % i)
        except TypeError as e:
            print(e)
    # 数值型变量
    O_data = copy.deepcopy(data.iloc[:, O])
    # 分类型变量
    C_data = copy.deepcopy(data.iloc[:, C])
    ##  数值型变量的列名
    O_index = O_data.columns.tolist()
    ## 分类型变量的列名
    C_index = C_data.columns.tolist()
    return O_index, C_index
