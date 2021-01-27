from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import Counter
from sklearn.base import TransformerMixin

from src.data.logparser import load_drain3


class FeatureExtractor(TransformerMixin):
    def __init__(self, method: str = None, preprocessing: str = None):
        self.method = method
        self.feature_names = None
        self.preprocessing = preprocessing
        self._mu = None
        self._idf = None

    @staticmethod
    def _create_dataframe(data: Dict) -> pd.DataFrame:
        occurrences = []
        for key, block in data.items():
            occ_block = Counter(line[0] for line in block)
            occurrences.append(occ_block)

        df = pd.DataFrame(occurrences, index=data.keys())
        return df.fillna(0)

    def _add_missing_columns(self, data: pd.DataFrame):
        missing_columns = set(self.feature_names) - set(data.columns)
        for col in missing_columns:
            data[col] = 0

    def fit_transform(self, data: Dict, y=None, **fit_params) -> np.ndarray:
        dataframe = self._create_dataframe(data)

        self.feature_names = dataframe.columns

        ret = dataframe.to_numpy()  # tf_{x_y} => the frequency of x in y document

        if self.method == 'tf-idf':
            df = np.sum(ret, axis=0)  # the number of documents containing x
            self._idf = np.log(len(ret) / df)
            ret = ret * self._idf  # tf - idf

        if self.preprocessing == 'mean':
            self._mu = ret.mean(axis=0)
            ret -= self._mu
        return ret

    def get_feature_names(self) -> List:
        return self.feature_names

    def transform(self, data: Dict):
        dataframe = self._create_dataframe(data)

        self._add_missing_columns(dataframe)
        ret = dataframe[self.feature_names].to_numpy()

        if self.method == 'tf-idf':
            ret = ret * self._idf  # tf - idf

        if self.preprocessing == 'mean':
            ret -= self._mu
        return ret


if __name__ == '__main__':
    d = load_drain3('../../data/interim/HDFS1/val-data-Drain3-HDFS1-cv1-1.binlog')

    fe = FeatureExtractor()
    print(np.allclose(fe.fit_transform(d), fe.transform(d)))
    print(fe.get_feature_names())
    print(fe.transform(d))