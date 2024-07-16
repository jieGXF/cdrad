# -*- coding: utf-8 -*-
import glob

import SimpleITK as sitk
import radiomics
from radiomics.featureextractor import RadiomicsFeatureExtractor
import os
import numpy as np
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.results = list()
        self.indexs = list()
        # self.params = 'Params.yaml'

    def feature_extractor(self, omics_path, img_path, params):

        masks_path = glob.glob(os.path.join(img_path, 'mask*'))
        for mask_path in masks_path:
            features = list()
            indexs = list()
            mask_list = os.listdir(mask_path)
            # mask_list = os.listdir(os.path.join(img_path, 'image'))
            mask_list.sort()
            for id in mask_list:
                image = os.path.join(img_path, 'image', id)
                mask = os.path.join(mask_path, id)
                print(id)
                indexs.append(id.split('.')[0])
                result = self.compute_features(image, mask, params)
                features.append(result)

            df = pd.DataFrame(features, index=indexs)
            df.drop(df.columns[list(range(37))], axis=1, inplace=True)  # drop the non feature
            df = df.add_prefix(img_path.split('/')[-1] + '_')
            df.to_csv(os.path.join(omics_path, "radiomics", img_path.split('/')[-1] + mask_path.split('mask')[-1] + '.csv'),
                      index=True, index_label="ID")

    def compute_features(self, image, mask, params):
        extractor = RadiomicsFeatureExtractor(params)
        feature = extractor.execute(image, mask)
        return feature
