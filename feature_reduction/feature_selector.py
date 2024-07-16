import pickle
import os
from .selection_func import *
from metrics_reports.feature_visualization import plot_featur_heatmap
# from models.models_construction import modeling

class FeatureSelector:
    def __init__(self, label_name, save_dir):
        self.save_dir = save_dir
        self.label_name = label_name
        self.subject_name = "ID"
        self.modus = 'binary_classification'
        self.feature_selection_log_before = None
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

    def run_selection(self, feature_selection_methods, X_train_df, y_train_df, X_test_df, save_dir):
            self.feature_selection_methods = feature_selection_methods
            X_train_df2 = X_train_df.drop(columns=[self.subject_name]+self.label_name)
            X_train_arr = X_train_df2.values
            feature_name = X_train_df2.columns.values
            y_train_arr = y_train_df[self.label_name].values
            self.feature_selection_log = {}
            print('The number of features is {}'.format(X_train_df2.shape[1]))
            print('Processing with %s' % feature_selection_methods)
            selector = self.FUNCS[feature_selection_methods]  # corresponding selection function
            kwargs = {'feature_name': feature_name, 'label_name': self.label_name}
            support = self.selector_base(selector, X_train_arr, y_train_arr, save_dir, kwargs)
            self.feature_selection_log = support
            X_train = X_train_arr[:, support]
            feature_name = feature_name[support]
            # with open(os.path.join(self.save_dir, 'feature_selection_process.pkl'.format(i, j)), 'wb') as f:
            #     pickle.dump(self.feature_selection_log, f)

            X_train_fs = self.get_feature_with_log(X_train_df, data_tag='train')
            X_test_fs = self.get_feature_with_log(X_test_df, data_tag='test')
            return X_train_fs, X_test_fs

    def get_feature_with_log(self, feature_data, data_tag='train'):
        feature_clean_train = feature_data.drop(columns=[self.subject_name]+self.label_name,
                                                axis=1)  # remove non feature value
        labels = feature_data[self.label_name].values
        case_ids = feature_data[self.subject_name].values
        feature_value = feature_clean_train.values
        feature_name = feature_clean_train.columns.values
        selection_results = {}
        index_pd = []
        support = self.feature_selection_log
        feature_value = feature_value[:, support]
        feature_name = feature_name[support]
        method = self.feature_selection_methods
        selection_results['after_' + method] = feature_name
        index_pd.append(method)
        if len(feature_name) <= 100:  # plot heatmap
            save_name = 'heatmap_of_{}_samples_after_{}.png'.format(data_tag, method)
            new_feature_df = pd.DataFrame(feature_value, columns=feature_name)
            # plot_featur_heatmap(new_feature_df, self.save_dir, save_name)
        new_feature_data = pd.DataFrame(feature_value, columns=feature_name)
        new_feature_data.insert(0, self.subject_name, case_ids)
        new_feature_data.insert(1, self.label_name[0], labels)
        series = pd.DataFrame.from_dict(selection_results, orient='index')
        series.to_csv(os.path.join(self.save_dir, 'feature_selection_result.csv'), index=False)
        new_feature_data.to_csv(os.path.join(self.save_dir, '{}_feature_after_selection.csv'.format(data_tag)),
                                index=False)
        return new_feature_data