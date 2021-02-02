from __future__ import annotations

import numpy as np
from typing import List
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader

from src.models.vanilla_tcnn import VanillaTCN, EmbeddingDataset
from src.visualization.visualization import visualize_distribution_with_labels

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
    X_train = load_pickle_file('../../data/processed/HDFS1/X-train-HDFS1-cv1-1-block.npy')
    X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.npy')
    y_train = np.load('../../data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')

    sc = CustomMinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    X = X_train[y_train == 0][np.random.randint(0, 400000, size=1000)]  # get only normal training examples

    model = VanillaTCN(epochs=1, learning_rate=0.0001)
    model.fit(X)

    test_indices = np.random.randint(0, 50000, size=500)
    y_pred = model.predict(X_val[test_indices])

    visualize_distribution_with_labels(y_pred, y_val[test_indices], to_file=True)
