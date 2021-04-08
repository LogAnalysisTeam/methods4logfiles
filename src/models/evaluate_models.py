from __future__ import annotations

import torch
import numpy as np
import os
from typing import List, Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.models.train_hybrid_model import get_extracted_features
from src.models.metrics import metrics_report, get_metrics
from src.models.utils import load_pickle_file, classify, load_experiment, convert_predictions

SEED = 160121
np.random.seed(SEED)
torch.manual_seed(SEED)


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
        return np.asarray([(x - self.x_min) / diff for x in X], dtype='object')


def evaluate_tcnn(x_train: List, x_test: List, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_test = sc.fit(x_train).transform(x_test)

    training_stats = load_experiment('../../models/tcn/experiments.json')
    score = evaluate(x_test, y_test, training_stats['experiments'])
    return score


def evaluate_aetcnn(x_train: List, x_test: List, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_test = sc.fit(x_train).transform(x_test)

    training_stats = load_experiment('../../models/aetcn/experiments.json')
    score = evaluate(x_test, y_test, training_stats['experiments'])
    return score


def evaluate_aecnn1d(x_train: List, x_test: List, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_test = sc.fit(x_train).transform(x_test)

    training_stats = load_experiment('../../models/aecnn1d/experiments.json')
    score = evaluate(x_test, y_test, training_stats['experiments'])
    return score


def evaluate_cnn1d(x_train: List, x_test: List, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_test = sc.fit(x_train).transform(x_test)

    training_stats = load_experiment('../../models/cnn1d/experiments.json')
    score = evaluate(x_test, y_test, training_stats['experiments'])
    return score


def evaluate_cnn2d(x_train: List, x_test: List, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_test = sc.fit(x_train).transform(x_test)

    training_stats = load_experiment('../../models/cnn2d/experiments.json')
    score = evaluate(x_test, y_test, training_stats['experiments'])
    return score


def evaluate_tcnn_cnn1d(x_train: List, x_test: List, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_test = sc.fit(x_train).transform(x_test)

    training_stats = load_experiment('../../models/cnn1d_tcn/experiments.json')
    score = evaluate(x_test, y_test, training_stats['experiments'])
    return score


def evaluate_sa_cnn1d(x_train: List, x_test: List, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_test = sc.fit(x_train).transform(x_test)

    training_stats = load_experiment('../../models/sa_cnn1d/experiments.json')
    score = evaluate(x_test, y_test, training_stats['experiments'])
    return score


def evaluate_sa_cnn2d(x_train: List, x_test: List, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_test = sc.fit(x_train).transform(x_test)

    training_stats = load_experiment('../../models/sa_cnn2d/experiments.json')
    score = evaluate(x_test, y_test, training_stats['experiments'])
    return score


def train_hybrid_model_ae(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = AutoEncoder()

    experiments = load_experiment('../../models/AE-AETCN-hybrid-hyperparameters-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_hybrid_model_if(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = IsolationForest(bootstrap=True, n_jobs=1, random_state=SEED)

    experiments = load_experiment('../../models/IF-AETCN-hybrid-hyperparameters-HDFS1.json')
    evaluated_hyperparams = random_search_unsupervised((x_train, x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def find_best_model(experiments: Dict, metric: str = 'f1_score') -> Dict:
    return max(experiments, key=lambda x: x['metrics'][metric])


def evaluate(x_test: np.ndarray, y_test: np.array, experiments: Dict) -> Dict:
    model_config = find_best_model(experiments)

    model = torch.load(model_config['model_path'])
    theta = model_config['threshold']

    y_pred = model.predict(x_test)  # return reconstruction errors

    y_pred = classify(y_pred, theta)
    metrics_report(y_test, y_pred)
    return {
        'val_metrics': model_config['metrics'],
        'test_metrics': get_metrics(y_test, y_pred)
    }


if __name__ == '__main__':
    X_train = load_pickle_file('../../data/processed/HDFS1/X-train-HDFS1-cv1-1-block.pickle')
    X_test = load_pickle_file('../../data/processed/HDFS1/X-test-HDFS1-block.pickle')
    y_test = np.load('../../data/processed/HDFS1/y-test-HDFS1-block.npy')

    results = evaluate_tcnn(X_train, X_test, y_test)
    print(results)

    results = evaluate_cnn1d(X_train, X_test, y_test)
    print(results)

    results = evaluate_cnn2d(X_train, X_test, y_test)
    print(results)

    results = evaluate_tcnn_cnn1d(X_train, X_test, y_test)
    print(results)

    results = evaluate_aetcnn(X_train, X_test, y_test)
    print(results)

    results = evaluate_aecnn1d(X_train, X_test, y_test)
    print(results)

    results = evaluate_sa_cnn1d(X_train, X_test, y_test)
    print(results)

    results = evaluate_sa_cnn2d(X_train, X_test, y_test)
    print(results)

    ################################ HYBRID MODELS #####################################################################

    # train_path = '../../data/processed/HDFS1/X-train-HDFS1-interim-features.npy'
    # val_path = '../../data/processed/HDFS1/X-val-HDFS1-interim-features.npy'
    #
    # if os.path.exists(train_path) and os.path.exists(val_path):
    #     X_train = np.load(train_path)
    #     X_val = np.load(val_path)
    # else:
    #     X_train, X_val = get_extracted_features(X_train, X_val, y_train, y_val)
    #
    # # # apply ReLU
    # # X_train[X_train < 0] = 0
    # # X_val[X_val < 0] = 0
    #
    # results = train_hybrid_model_if(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)
