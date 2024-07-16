import pandas as pd

from Data.BaseData import BaseData
from Data.util import load_data_base


class Data_2_group(BaseData):
    def __init__(self, subj_column_name, label_column_name, pat_label, con_label, filepath_con, filepath_pat, sheetname):
        super().__init__(subj_column_name, label_column_name, pat_label, con_label)
        print("processing two seperate files")
        self.filepath_pat = filepath_pat
        self.filepath_con = filepath_con
        self.sheetname = sheetname

        self.load_data()


    def load_data(self):
        self.data_con = load_data_base(self.filepath_con)
        self.data_con.insert(1, self.label_column_name, [self.con_label] * self.data_con.shape[0])
        self.data_pat = load_data_base(self.filepath_pat)
        self.data_pat.insert(1, self.label_column_name, [self.pat_label] * self.data_pat.shape[0])
        self.data = pd.concat([self.data_con, self.data_pat], axis=0)

        self.features = self.data.drop([self.subj_column_name, self.label_column_name], axis=1)
        self.subjs = self.data[self.subj_column_name]
        self.labels = self.data[self.label_column_name]
