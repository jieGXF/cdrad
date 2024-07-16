import pickle
import os
from .selection_func import *
from metrics_reports.feature_visualization import plot_featur_heatmap
from models.models_construction import modeling

class FeatureSelector:
    def __init__(self, label_name, save_dir):
        self.save_dir = save_dir
        self.label_name = label_name
        self.subject_name = "ID"
        self.modus = 'binary_classification'
        self.feature_selection_log_before = None
        self.feature_selection_methods_2nd = {
            # 1: 'Chi2Test',
            1: 'ANOVA F-value',
            2: 'Mutual information',
            4: 'mRMR(minimum redundancy maximum relevance)',
            5: 'Relief',
            6: 'RFE(recursive feature elimination)',
            7: 'Lasso',  # two graph of loss and coefficient,one excel
            8: 'RandomForestImportance',  # graph of feature importance,one excel
            9: 'Sequential Feature Selection',
            10: 'Boruta',
            11: 'T MWU Ttest',
            12: 'Correlation Redundancy',
            13: 'VarianceThreshold Selection'
        }
        self.FUNCS = {
            'Chi2Test': chi2_test_selector,
            'ANOVA F-value': anova_f_value_selector,
            'Mutual information': mutual_information_selector,
            'mRMR(minimum redundancy maximum relevance)': mRMR_selector,
            'Relief': refief_selector,
            'RFE': RFE_selector,
            'LASSO': lasso_selector,
            'RandomForestImportance': RF_selector,
            'Sequential Feature Selection': SFS_selector,
            'Boruta': Boruta_selector,
            'T MWU Ttest': T_MWU_selector,
            'Correlation Redundancy': Correlation_selector,
            'VarianceThreshold Selection': VarianceThreshold_selector
        }

    def selector_base(self, selector_function, feature, label, save_dir, kwargs):
        support = selector_function(feature, label, save_dir, kwargs)
        return support

    def load_selected(self):
        file_name = os.path.join(self.save_dir, 'feature_selection_process.pkl')
        if os.path.exists(file_name):
            self.feature_selection_log_before = pickle.load(open(file_name, 'rb'))

    def run_selection(self, feature_selection_methods, X_train_df, y_train_df, X_test_df, y_test_df):
            print("feature selection for each data frame")
            step = 1
            self.feature_selection_methods = feature_selection_methods
            for df_name in X_train_df.keys():
                X_train_single = X_train_df[df_name]
                X_test_single = X_test_df[df_name]
                X_train_single_clean = X_train_single.drop(columns=[self.subject_name]+self.label_name)
                X_train = X_train_single_clean.values
                y_train = y_train_df[self.label_name].values
                feature_name = X_train_single_clean.columns.values
                self.feature_selection_log = {}
                print('The all number of feature is {}'.format(X_train.shape[1]))
                current_method = feature_selection_methods[df_name]
                print(feature_selection_methods.keys(), 'Processing with %s' % current_method)
                selector = self.FUNCS[current_method]  # corresponding selection function
                kwargs = {'feature_name': feature_name, 'label_name': self.label_name}
                self.save_subdir = os.path.join(self.save_dir, 'filter-selection-1st', df_name + '-' + current_method)
                maybe_mkdir_p(self.save_subdir)
                support = self.selector_base(selector, X_train, y_train, self.save_subdir, kwargs)
                X_train = X_train[:, support]
                feature_name = feature_name[support]
                self.feature_selection_log[df_name] = support
                step += 1
                # with open(os.path.join(self.save_dir, 'feature_selection_process.pkl'.format(i, j)), 'wb') as f:
                #     pickle.dump(self.feature_selection_log, f)
                new_train_set = self.get_feature_with_log(X_train_single, data_tag='train')
                new_test_set = self.get_feature_with_log(X_test_single, data_tag='test')
                # TODO save new train and test set
                X_train_df[df_name] = new_train_set
                X_test_df[df_name] = new_test_set
            return X_train_df, X_test_df

    def run_selection_modeling(self, X_train_all, y_train, X_test_all, y_test):
        feature_clean_train = X_train_all.drop(columns=[self.subject_name]+self.label_name)
        X_train = feature_clean_train.values
        y_train = y_train[self.label_name].values
        feature_name = feature_clean_train.columns.values
        self.feature_selection_log = {}
        print('The all number of feature is {}'.format(X_train.shape[1]))
        step = 1
        self.feature_selection_methods = self.feature_selection_methods_2nd
        for process in self.feature_selection_methods_2nd.keys():
            current_method = self.feature_selection_methods_2nd[process]
            print(self.feature_selection_methods_2nd.keys(), 'Processing with %s' % current_method)
            selector = self.FUNCS[current_method]  # corresponding selection function
            kwargs = {'feature_name': feature_name, 'label_name': self.label_name}
            self.save_subdir = os.path.join(self.save_dir, 'filter-selection-2nd', current_method)
            maybe_mkdir_p(self.save_subdir)
            support = self.selector_base(selector, X_train, y_train, self.save_subdir, kwargs)
            X_train_fs = X_train[:, support]
            feature_name = feature_name[support]
            self.feature_selection_log[process] = support
            step += 1
            # with open(os.path.join(self.save_dir, 'feature_selection_process.pkl'.format(i, j)), 'wb') as f:
            #     pickle.dump(self.feature_selection_log, f)
            new_train_set = self.get_feature_with_log(X_train_all, data_tag='train')
            new_test_set = self.get_feature_with_log(X_test_all, data_tag='test')
            all_models = modeling(self.modus, new_train_set, new_test_set, self.save_subdir,
                                  self.subject_name, self.label_name)
            return all_models
            # TODO: save new train and test set
            # TODO: train and test each model
    def get_feature_with_log(self, feature_data, data_tag='train'):
        feature_clean_train = feature_data.drop(columns=[self.subject_name]+self.label_name,
                                                axis=1)  # remove non feature value
        labels = feature_data[self.label_name].values
        case_ids = feature_data[self.subject_name].values
        feature_value = feature_clean_train.values
        feature_name = feature_clean_train.columns.values
        selection_results = {}
        index_pd = []
        for process in self.feature_selection_log.keys():
            support = self.feature_selection_log[process]
            feature_value = feature_value[:, support]
            feature_name = feature_name[support]
            method = self.feature_selection_methods[process]
            selection_results['after_' + method] = feature_name
            index_pd.append(method)
            if len(feature_name) <= 100:  # plot heatmap
                save_name = 'heatmap_of_{}_samples_after_{}.png'.format(data_tag, method)
                new_feature_df = pd.DataFrame(feature_value, columns=feature_name)
                plot_featur_heatmap(new_feature_df, self.save_subdir, save_name)
        new_feature_data = pd.DataFrame(feature_value, columns=feature_name)

        new_feature_data.insert(0, self.subject_name, case_ids)
        new_feature_data.insert(1, self.label_name[0], labels)
        series = pd.DataFrame.from_dict(selection_results, orient='index')
        series.to_csv(os.path.join(self.save_subdir, 'feature_selection_result.csv'), index=False)
        new_feature_data.to_csv(os.path.join(self.save_subdir, '{}_feature_after_selection.csv'.format(data_tag)),
                                index=False)
        return new_feature_data

# class FeatureSelector(object):
#     def __init__(self, feature_selection_methods, save_dir, subject_column_name, label_column_name, pat_label, con_label):
#         self.feature_selection_methods = feature_selection_methods
#         self.save_dir = save_dir
#         self.subject_column_name = subject_column_name
#         self.label_column_name = label_column_name
#         self.pat_label = pat_label
#         self.con_label = con_label
#         self.feature_selection_log_before = None
#         self.feature_selection_method_all = {
#             1: 'Chi2Test',
#             2: 'ANOVA F-value',
#             3: 'Mutual information',
#             4: 'mRMR(minimum redundancy maximum relevance)',
#             5: 'Relief',
#             6: 'RFE(recursive feature elimination)',
#             7: 'Lasso',  # two graph of loss and coefficient,one excel
#             8: 'RandomForestImportance',  # graph of feature importance,one excel
#             9: 'Sequential Feature Selection',
#             10: 'Boruta',
#             11: 'T MWU Ttest',
#             12: 'Correlation Redundancy',
#             13: 'VarianceThreshold Selection'
#         }
#         self.FUNCS = {
#             'Chi2Test': chi2_test_selector,
#             'ANOVA F-value': anova_f_value_selector,
#             'Mutual information': mutual_information_selector,
#             'mRMR(minimum redundancy maximum relevance)': mRMR_selector,
#             'Relief': refief_selector,
#             'RFE(recursive feature elimination)': RFE_selector,
#             'Lasso': lasso_selector,
#             'RandomForestImportance': RF_selector,
#             'Sequential Feature Selection': SFS_selector,
#             'Boruta': Boruta_selector,
#             'T MWU Ttest': T_MWU_selector,
#             'Correlation Redundancy': Correlation_selector,
#             'VarianceThreshold Selection': VarianceThreshold_selector
#         }
#
#     def selector_base(self, selector_function, feature, label, kwargs):
#         support = selector_function(feature, label, self.save_dir, kwargs)
#         return support
#
#     def load_selected(self):
#         file_name = os.path.join(self.save_dir, 'feature_selection_process.pkl')
#         if os.path.exists(file_name):
#             self.feature_selection_log_before = pickle.load(open(file_name, 'rb'))
#
#     def run_selection(self, train_set, test_set,i,j):
#         print('There is(are) {} step(s) in feature selection'.format(len(self.feature_selection_methods.keys())))
#
#         feature_clean_train = train_set.drop(columns=[self.subject_column_name, self.label_column_name])
#         X_train = feature_clean_train.values
#         y_train = train_set[self.label_column_name].values
#         feature_name = feature_clean_train.columns.values
#
#         self.feature_selection_log = {}
#         print('The all number of feature is {}'.format(X_train.shape[1]))
#         step = 1
#         for process in self.feature_selection_methods.keys():
#             current_method = self.feature_selection_method_all[process]
#             print('(step %d of %d) Processing with %s' % (step,
#                                                           len(self.feature_selection_methods.keys()),
#                                                           current_method))
#
#             selector = self.FUNCS[current_method]  # corresponding selection function
#             kwargs = self.feature_selection_methods[process]
#             kwargs['feature_name'] = feature_name
#             kwargs['pat_label'] = self.pat_label
#             kwargs['con_label'] = self.con_label
#             kwargs['label_column_name'] = self.label_column_name
#
#             support = self.selector_base(selector, X_train, y_train, kwargs)
#             X_train = X_train[:, support]
#
#             feature_name = feature_name[support]
#             self.feature_selection_log[process] = support
#             step += 1
#         with open(os.path.join(self.save_dir, 'feature_selection_process.pkl'.format(i,j)), 'wb') as f:
#             pickle.dump(self.feature_selection_log, f)
#         new_train_set = self.get_feature_with_log(train_set, data_tag='train')
#         new_test_set = self.get_feature_with_log(test_set, data_tag='test')
#         return new_train_set, new_test_set
#
#     def get_feature_with_log(self, feature_data, data_tag='train'):
#         feature_clean_train = feature_data.drop(columns=[self.subject_column_name, self.label_column_name], axis=1)  # remove non feature value
#         labels = feature_data[self.label_column_name].values
#         case_ids = feature_data[self.subject_column_name].values
#         feature_value = feature_clean_train.values
#         feature_name = feature_clean_train.columns.values
#         selection_results = {}
#         index_pd = []
#         for process in self.feature_selection_log.keys():
#             support = self.feature_selection_log[process]
#             feature_value = feature_value[:, support]
#             feature_name = feature_name[support]
#             method = self.feature_selection_method_all[process]
#             selection_results['after_' + method] = feature_name
#             index_pd.append(method)
#             if len(feature_name) <= 100:  # plot heatmap
#                 save_name = 'heatmap_of_{}_samples_after_{}.png'.format(data_tag, method)
#                 new_feature_df = pd.DataFrame(feature_value, columns=feature_name)
#                 plot_featur_heatmap(new_feature_df, self.save_dir, save_name)
#         new_feature_data = pd.DataFrame(feature_value, columns=feature_name)
#
#         new_feature_data.insert(0, self.subject_column_name, case_ids)
#         new_feature_data.insert(1, self.label_column_name, labels)
#         series = pd.DataFrame.from_dict(selection_results, orient='index')
#         series.to_excel(os.path.join(self.save_dir, 'feature_selection_result.xlsx'), index=index_pd)
#         new_feature_data.to_excel(os.path.join(self.save_dir, '{}_feature_after_selection.xlsx'.format(data_tag)),
#                                   index=False)
#         return new_feature_data
