from __future__ import annotations

import numpy as np
from typing import List
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from src.models.autoencoder import AutoEncoder
from src.models.vanilla_tcnn import VanillaTCN
from src.models.metrics import metrics_report
from src.visualization.visualization import visualize_distribution_with_labels
from src.data.hdfs import load_labels
from src.features.feature_extractor import FeatureExtractor
from src.models.train_baseline_models import get_labels_from_csv

SEED = 160121
np.random.seed(SEED)


def load_pickle_file(file_path: str) -> List:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class CustomMinMaxScaler(MinMaxScaler):
    def __init__(self):
        super().__init__()
        self.x_min = None
        self.x_max = None

    def fit(self, X: List, y=None) -> CustomMinMaxScaler:
        self.x_min = np.min([x.min(axis=0) for x in X], axis=0)
        self.x_max = np.max([x.max(axis=0) for x in X], axis=0)
        return self

    def fit_transform(self, X: List, y: np.array = None, **fit_params) -> np.array:
        return self.fit(X).transform(X)

    def transform(self, X: List) -> np.array:
        # (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        diff = self.x_max - self.x_min
        return np.asarray([(x - self.x_min) / diff for x in X])


if __name__ == '__main__':
    # X_train = load_pickle_file('/home/martin/bdip25/data/processed/HDFS1/X-train-HDFS1-cv1-1-block.npy')
    # X_val = load_pickle_file('/home/martin/bdip25/data/processed/HDFS1/X-val-HDFS1-cv1-1-block.npy')
    # y_train = np.load('/home/martin/bdip25/data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    # y_val = np.load('/home/martin/bdip25/data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')
    # X_train = load_pickle_file('../../data/interim/HDFS1/train-data-Drain3-HDFS1-cv1-1.binlog')
    # y_train = load_labels('../../data/interim/HDFS1/train-labels-HDFS1-cv1-1.csv')
    X_val = load_pickle_file('../../data/interim/HDFS1/val-data-Drain3-HDFS1-cv1-1.binlog')
    y_val = load_labels('../../data/interim/HDFS1/val-labels-HDFS1-cv1-1.csv')

    # y_train = get_labels_from_csv(y_train, X_train.keys())
    y_val = get_labels_from_csv(y_val, X_val.keys())

    # do standardization, normalization here!!!!

    # sc = CustomMinMaxScaler()
    # X_train = sc.fit_transform(X_train)
    # X_val = sc.transform(X_val)

    fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    # X_train = fe.fit_transform(X_train).astype(np.float32)
    X_val = fe.fit_transform(X_val).astype(np.float32)

    X = X_val[np.random.randint(0, 50000, size=45000)]

    model = AutoEncoder(epochs=5, learning_rate=0.0001)
    model._initialize_model(39, [128, 64, 8, 32], 0.3)
    model.fit(X)
    # torch.save(model, 'model.pth')

    test_indices = np.random.randint(50000, 51000, size=500)
    y_pred = model.predict(X_val[test_indices])

    for th in sorted(y_pred[y_val[test_indices] == 1]):
        tmp = np.zeros(shape=y_pred.shape)
        tmp[y_pred > th] = 1
        print('Threshold:', th)
        metrics_report(y_val[test_indices], tmp)
        
    visualize_distribution_with_labels(y_pred, y_val[test_indices], to_file=False)

    # model = VanillaTCN(epochs=15, learning_rate=0.0001)
    # model.fit(X)
    #
    # test_indices = np.random.randint(0, 50000, size=500)
    # y_pred = model.predict(X_val[test_indices])
    # visualize_distribution_with_labels(y_pred, y_val[test_indices], to_file=True)
