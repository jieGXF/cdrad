from Data.BaseData import BaseData
from Data.util import load_data_base


class Data_1_both(BaseData):
    def __init__(self, subj_column_name, label_column_name, pat_label, con_label, filepath, sheetname):
        super().__init__(subj_column_name, label_column_name, pat_label, con_label)
        print("processing a full file")
        self.filepath = filepath
        self.sheetname = sheetname

        self.load_data()

    def load_data(self):
        self.data = load_data_base(self.filepath, self.sheetname)
        # self.original_data = self.data
        self.data_pat = self.data.loc[self.data[self.label_column_name] == self.pat_label, :]
        self.data_con = self.data.loc[self.data[self.label_column_name] == self.con_label, :]
        self.features = self.data.drop([self.subj_column_name, self.label_column_name], axis=1)
        self.subjs = self.data[self.subj_column_name]
        self.labels = self.data[self.label_column_name]