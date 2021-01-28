from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from typing import Dict, Iterable, Union
import pandas as pd
import numpy as np
import itertools
import os

from src.features.feature_extractor import FeatureExtractor
from src.features.hdfs import check_order
from src.data.logparser import load_drain3
from src.data.hdfs import load_labels
from src.models.metrics import metrics_report, f1_score


def get_labels_from_csv(df: pd.DataFrame, keys: Iterable) -> np.array:
    check_order(keys, df['BlockId'])
    return df['Label'].to_numpy(dtype=np.int8)


def convert_predictions(y_pred: np.array) -> np.array:
    # LocalOutlierFactor returns: 1 inlier, -1 outlier
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    return y_pred


def train_lof(x_train: Dict, x_test: Dict, y_train: np.array, y_test: np.array):
    """
    Novelty detection represents the detection of anomalous data based on a training set consisting of only
    the normal data.
    """
    fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    x_train = fe.fit_transform(x_train)
    x_test = fe.transform(x_test)

    clf = LocalOutlierFactor(n_jobs=os.cpu_count())
    params = {'novelty': [True],
              'n_neighbors': [2, 5, 10, 20, 30, 50],
              'metric': ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski']}
    scores = grid_search((x_train[y_train == 0, :], x_test, None, y_test), clf, params)
    print(sorted(scores, key=lambda x: -x[1]))


def grid_search(data_and_labels: tuple, model: Union[LocalOutlierFactor], params: Dict) -> Dict:
    x_train, x_test, y_train, y_test = data_and_labels

    scores = {}
    for conf in itertools.product(*params.values()):
        kwargs = {k: val for k, val in zip(params.keys(), conf)}

        model.set_params(**kwargs)

        print(f'Training on {len(x_train)} and validating on {len(x_test)}.')
        print(f'Model (hyper)parameters are: {model.get_params()}.')
        model.fit(x_train)
        y_pred = model.predict(x_test)
        y_pred = convert_predictions(y_pred)

        metrics_report(y_test, y_pred)
        scores[conf] = f1_score(y_test, y_pred)
    return scores


if __name__ == '__main__':
    x_train = load_drain3('../../data/interim/HDFS1/train-data-Drain3-HDFS1-cv1-1.binlog')
    y_train = load_labels('../../data/interim/HDFS1/train-labels-HDFS1-cv1-1.csv')
    x_val = load_drain3('../../data/interim/HDFS1/val-data-Drain3-HDFS1-cv1-1.binlog')
    y_val = load_labels('../../data/interim/HDFS1/val-labels-HDFS1-cv1-1.csv')

    y_train = get_labels_from_csv(y_train, x_train.keys())
    y_val = get_labels_from_csv(y_val, x_val.keys())

    train_lof(x_train, x_val, y_train, y_val)
