class BaseData(object):
    def __init__(self, subj_column_name=None, label_column_name=None, pat_label=None, con_label=None):

        self.subj_column_name = subj_column_name
        self.label_column_name = label_column_name
        self.pat_label = pat_label
        self.con_label = con_label

        self.data = None
        self.data_pat = None
        self.data_con = None
