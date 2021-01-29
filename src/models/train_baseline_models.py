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

SEED = 160121
np.random.seed(SEED)


def get_labels_from_csv(df: pd.DataFrame, keys: Iterable) -> np.array:
    check_order(keys, df['BlockId'])
    return df['Label'].to_numpy(dtype=np.int8)


def convert_predictions(y_pred: np.array) -> np.array:
    # LocalOutlierFactor and IsolationForest returns: 1 inlier, -1 outlier
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    return y_pred


def train_lof(x_train: Dict, x_test: Dict, y_train: np.array, y_test: np.array):
    """
    Novelty detection represents the detection of anomalous data based on a training set consisting of only
    the normal data.
    """
    fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    fe.fit_transform(x_train)
    x_test = fe.transform(x_test)

    clf = LocalOutlierFactor(n_jobs=os.cpu_count())
    params = {'n_neighbors': np.linspace(5, 750, num=10),
              'metric': ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski']}
    scores = grid_search((None, x_test, None, y_test), clf, params)
    print(dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)))


def train_iso_forest(x_train: Dict, x_test: Dict, y_train: np.array, y_test: np.array):
    fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    x_train = fe.fit_transform(x_train)
    x_test = fe.transform(x_test)

    clf = IsolationForest(bootstrap=True, n_jobs=os.cpu_count(), random_state=SEED)
    params = {'n_estimators': [50, 100, 200, 500],
              'max_samples': np.linspace(0.01, 1, num=7),
              'max_features': [int(x) for x in np.linspace(1, x_train.shape[1], num=7)]}
    scores = grid_search((x_train, x_test, None, y_test), clf, params)
    print(dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)))


def grid_search(data_and_labels: tuple, model: Union[LocalOutlierFactor, IsolationForest], params: Dict) -> Dict:
    x_train, x_test, _, y_test = data_and_labels

    scores = {}
    for conf in itertools.product(*params.values()):
        kwargs = {k: val for k, val in zip(params.keys(), conf)}

        model.set_params(**kwargs)

        print(f'Model (hyper)parameters are: {model.get_params()}.')

        if isinstance(model, LocalOutlierFactor):
            y_pred = model.fit_predict(x_test)
        else:
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

    # train_lof(x_train, x_val, y_train, y_val)
    train_iso_forest(x_train, x_val, y_train, y_val)
