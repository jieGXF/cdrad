from pyirr import read_data, intraclass_correlation
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np
# from WORC.classification.metrics import ICC
import pandas as pd

def ICC(M, ICCtype='inter'):
    '''
    Input:
        M is matrix of observations. Rows: patients, columns: observers.
        type: ICC type, currently "inter" or "intra".
    '''

    n, k = M.shape

    SStotal = np.var(M, ddof=1) * (n*k - 1)
    MSR = np.var(np.mean(M, 1), ddof=1) * k
    MSW = np.sum(np.var(M, 1, ddof=1)) / n
    MSC = np.var(np.mean(M, 0), ddof=1) * n
    MSE = (SStotal - MSR * (n - 1) - MSC * (k -1)) / ((n - 1) * (k - 1))

    if ICCtype == 'intra':
        r = (MSR - MSW) / (MSR + (k-1)*MSW)
    elif ICCtype == 'inter':
        r = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC-MSE)/n)
    else:
        raise ValueError('No valid ICC type given.')

    return r

class ICCThreshold(BaseEstimator, SelectorMixin):
    """
    Object to fit feature selection based on intra- or inter-class correlation
    coefficient as defined by

    Shrout, Patrick E., and Joseph L. Fleiss. "Intraclass correlations: uses
    in assessing rater reliability." Psychological bulletin 86.2 (1979): 420.
    http://rokwa.x-y.net/Shrout-Fleiss-ICC.pdf

    For the intra-class, we use ICC(3,1).For the inter-class ICC, we should use
    ICC(2,1) according to definitions of the paper, but according to radiomics
    literatue (https://www.tandfonline.com/doi/pdf/10.1080/0284186X.2018.1445283?needAccess=true,
    https://www.tandfonline.com/doi/pdf/10.3109/0284186X.2013.812798?needAccess=true),
    we use ICC(3,1) anyway.

    The default threshold of 0.75 is also based on the literature metioned
    above.

    """

    def __init__(self, ICCtype='intra', threshold=0.75):
        """
        Parameters
        ----------
        ICCtype: string, default 'intra'
                Type of ICC used. intra results in ICC(3,1), inter in ICC(2,1)
        threshold: float, default 0.75
                Threshold for ICC-value in order for feature to be selected

        """
        self.ICCtype = ICCtype
        self.threshold = threshold

    def fit(self, X_trains):
        """
        Select only features specificed by the metric and threshold per patient.

        Parameters
        ----------
        X_trains: numpy array, mandatory
                Array containing feature values used for model_selection.
                Number of objects on first axis, features on second axis, observers on third axis.

        Y_train: numpy array, mandatory
                Array containing the binary labels for each object in X_train.
        """

        self.selectrows = list()
        self.metric_values = list()

        # Perform the statistical test for each feature
        n_patient = X_trains.shape[0]
        n_feat = X_trains.shape[1]
        n_observers = X_trains.shape[2]
        for i_feat in range(0, n_feat):
            # Select only this specific feature for all objects
            fv = np.empty((n_patient, n_observers))
            for i_obs in range(0, n_observers):
                fv[:, i_obs] = X_trains[:, i_feat, i_obs]

            # Compute the ICC
            try:
                metric_value = ICC(fv, self.ICCtype)
            except ValueError as e:
                print("[WORC Warning] " + str(e) + '. Replacing metric value by 1.')
                metric_value = 1

            self.metric_values.append(metric_value)
            if metric_value > self.threshold:
                self.selectrows.append(i_feat)

    def transform(self, inputarray):
        """
        Transform the inputarray to select only the features based on the
        result from the fit function.

        Parameters
        ----------
        inputarray: numpy array, mandatory
                Array containing the items to use selection on. The type of
                item in this list does not matter, e.g. floats, strings etc.
        """
        return np.asarray([np.asarray(x)[self.selectrows].tolist() for x in inputarray])

    def _get_support_mask(self):
        # NOTE: metric is required for the Selector class, but can be empty
        pass


def filter_features_ICC(all_features, csv_out=None,
                        features_out=None, threshold=0.75):
    """
    For features from multiple observers, compute ICC, return values,
    and optionally apply thresholding and save output.

    features_in: list, containing one list per observer.
    csv_out: csv file, name of file to which ICC values should be written
    features_out: list, containing file names of output features.
    """
    # all_features = list()

    for i in range(len(all_features)):
        if i == 0:
            image_features = all_features[0].values
        else:
            image_features = np.dstack((image_features, all_features[i].values))
        i = i + 1
    # image_features = np.dstack((all_features[0].values, all_features[1].values, all_features[2].values))
    feature_labels = all_features[0].columns.tolist()

    # Compute the ICC
    # print('Computing ICC.')
    ICCthresholder = ICCThreshold(threshold=threshold)
    ICCthresholder.fit(image_features)

    # Extract the metric values and save to csv if required
    if csv_out:
        # print('\t Saving ICC metric values to csv.')
        ICCs = ICCthresholder.metric_values
        df = pd.DataFrame(zip(feature_labels, ICCs),
                          columns=['feature_label', 'ICC'])
        df.to_csv(csv_out)
        tmp = df.loc[(df["ICC"] > 0.8)]
        tmp2 = tmp["feature_label"].to_list()
        # feat_icc = [i for i, x in enumerate(ICCs) if x > 0.8]
        tmp2.insert(0, "ID")
        return tmp2

    # Save the thresholded features if required:
    if features_out:
        print('\t Saving selected features to hdf5.')
        # Select feature labels
        fl = ICCthresholder.transform([feature_labels])[0]
        for i_obs in range(all_features.shape[2]):
            # Select/hreshold feature values
            fv = np.squeeze(all_features[:, :, i_obs])
            fv = ICCthresholder.transform(fv)

            for i_patient in range(all_features.shape[0]):
                # Extract feature values for this patient
                fv_pat = np.squeeze(fv[i_patient, :])

                # Convert to pandas Series and save as hdf5
                panda_data = pd.Series([fv_pat.tolist(), fl.tolist()],
                                       index=['feature_values',
                                              'feature_labels'],
                                       name='Image features'
                                       )

                output = features_out[i_obs][i_patient]
                print(f'Saving image features to {output}.')
                panda_data.to_hdf(output, 'image_features')

# def pearson(    ):
#     my_rho = np.corrcoef(x_simple, y_simple)
#
#
# def cronbach():
#
#
#
# def kappa():
