from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import uuid
import os
from typing import List, Dict, Union
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.hdfs import load_labels
from src.features.feature_extractor import FeatureExtractor
from src.models.autoencoder import AutoEncoder
from src.models.vanilla_tcnn import VanillaTCN
from src.models.autoencoder_tcnn import AETCN
from src.models.autoencoder_cnn1d import AECNN1D
from src.models.cnn1d import CNN1D
from src.models.cnn2d import CNN2D
from src.models.tcnn_cnn1d import TCNCNN1D
from src.models.sa_cnn1d import SACNN1D
from src.models.sa_cnn2d import SACNN2D
from src.models.train_hybrid_model import get_extracted_features
from src.models.metrics import metrics_report, get_metrics
from src.models.train_baseline_models import get_labels_from_csv
from src.models.utils import create_experiment_report, create_checkpoint, save_experiment, load_pickle_file, \
    find_optimal_threshold, classify, load_experiment, create_model_path, convert_predictions

SEED = 160121
np.random.seed(SEED)
torch.manual_seed(SEED)

DIR_TO_EXPERIMENTS = '../../models/ae_baseline'
EXPERIMENT_PATH = os.path.join(DIR_TO_EXPERIMENTS, 'experiments.json')


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


def train_tcnn(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = VanillaTCN()

    experiments = load_experiment('../../models/TCN-hyperparameters-embeddings-clipping-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_aetcnn(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = AETCN()

    experiments = load_experiment('../../models/AETCN-hyperparameters-embeddings-clipping-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_aecnn1d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = AECNN1D()

    experiments = load_experiment('../../models/AECNN1D-hyperparameters-embeddings-clipping-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_cnn1d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = CNN1D()

    experiments = load_experiment('../../models/CNN1D-inverse-bottleneck-hyperparameters-embeddings-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_cnn2d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = CNN2D()

    experiments = load_experiment('../../models/CNN2D-inverse-bottleneck-hyperparameters-embeddings-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_tcnn_cnn1d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = TCNCNN1D()

    experiments = load_experiment('../../models/TCNCNN1D-inverse-bottleneck-hyperparameters-embeddings-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_sa_cnn1d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = SACNN1D()

    experiments = load_experiment('../../models/SACNN1D-hyperparameters-embeddings-clipping-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_sa_cnn2d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = SACNN2D()

    experiments = load_experiment('../../models/SACNN2D-hyperparameters-embeddings-clipping-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


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


def train_autoencoder(x_train: Dict, x_test: Dict, y_train: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
    fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    y_train = get_labels_from_csv(y_train, x_train.keys())
    y_test = get_labels_from_csv(y_test, x_test.keys())
    x_train = fe.fit_transform(x_train)
    x_test = fe.transform(x_test)

    model = AutoEncoder()

    experiments = load_experiment('../../models/AE-hyperparameters-Drain3-HDFS1.json')
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, experiments)
    return evaluated_hyperparams


def train_lof(x_train: Dict, x_test: Dict,  y_train: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
    fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    y_test = get_labels_from_csv(y_test, x_test.keys())
    fe.fit_transform(x_train)
    x_test = fe.transform(x_test)

    clf = LocalOutlierFactor(n_jobs=os.cpu_count())

    experiments = load_experiment('../../models/LOF-hyperparameters-Drain3-HDFS1.json')
    evaluated_hyperparams = random_search_unsupervised((None, x_test, None, y_test), clf, experiments)
    return evaluated_hyperparams


def train_iso_forest(x_train: Dict, x_test: Dict,  y_train: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
    fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    y_test = get_labels_from_csv(y_test, x_test.keys())
    x_train = fe.fit_transform(x_train)
    x_test = fe.transform(x_test)

    clf = IsolationForest(bootstrap=True, n_jobs=os.cpu_count(), random_state=SEED)

    experiments = load_experiment('../../models/IF-hyperparameters-Drain3-HDFS1.json')
    evaluated_hyperparams = random_search_unsupervised((x_train, x_test, None, y_test), clf, experiments)
    return evaluated_hyperparams


def random_search(data_and_labels: tuple, model: Union[AutoEncoder, VanillaTCN, AETCN, AECNN1D, CNN1D, CNN2D, TCNCNN1D,
                                                       SACNN1D, SACNN2D], params: Dict) -> Dict:
    x_train, x_test, _, y_test = data_and_labels

    scores = []
    for experiment in params['experiments']:
        model.set_params(**experiment['hyperparameters'])

        print(f'Model current hyperparameters are: {experiment["hyperparameters"]}.')

        model.fit(x_train)
        y_pred = model.predict(x_test)  # return reconstruction errors

        theta, f1 = find_optimal_threshold(y_test, y_pred)
        y_pred = classify(y_pred, theta)
        metrics_report(y_test, y_pred)

        model_path = create_model_path(DIR_TO_EXPERIMENTS, str(uuid.uuid4()))
        torch.save(model, model_path)

        res = create_experiment_report(get_metrics(y_test, y_pred), experiment['hyperparameters'], theta, model_path)
        scores.append(res)
        create_checkpoint({'experiments': scores}, EXPERIMENT_PATH)
    return {
        'experiments': scores
    }


def random_search_unsupervised(data_and_labels: tuple, model: Union[LocalOutlierFactor, IsolationForest],
                               params: Dict) -> Dict:
    x_train, x_test, _, y_test = data_and_labels

    scores = []
    for experiment in params['experiments']:
        model.set_params(**experiment['hyperparameters'])

        print(f'Model current hyperparameters are: {experiment["hyperparameters"]}.')

        if isinstance(model, LocalOutlierFactor):
            y_pred = model.fit_predict(x_test)  # return labels
        else:
            model.fit(x_train)
            y_pred = model.predict(x_test)  # return labels

        y_pred = convert_predictions(y_pred)
        metrics_report(y_test, y_pred)

        model_path = create_model_path(DIR_TO_EXPERIMENTS, str(uuid.uuid4()))
        torch.save(model, model_path)

        res = create_experiment_report(get_metrics(y_test, y_pred), experiment['hyperparameters'], file_path=model_path)
        scores.append(res)
        create_checkpoint({'experiments': scores}, EXPERIMENT_PATH)
    return {
        'experiments': scores
    }


if __name__ == '__main__':
    X_train = load_pickle_file('../../data/processed/HDFS1/X-train-HDFS1-cv1-1-block.pickle')
    X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.pickle')
    y_train = np.load('../../data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')

    # results = train_tcnn(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_cnn1d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_cnn2d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_tcnn_cnn1d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_aetcnn(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_aecnn1d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_sa_cnn1d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_sa_cnn2d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

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

    ########################### BASELINE METHODS #######################################################################

    X_train = load_pickle_file('../../data/interim/HDFS1/train-data-Drain3-HDFS1-cv1-1.binlog')
    y_train = load_labels('../../data/interim/HDFS1/train-labels-HDFS1-cv1-1.csv')
    X_val = load_pickle_file('../../data/interim/HDFS1/val-data-Drain3-HDFS1-cv1-1.binlog')
    y_val = load_labels('../../data/interim/HDFS1/val-labels-HDFS1-cv1-1.csv')

    results = train_autoencoder(X_train, X_val, y_train, y_val)
    save_experiment(results, EXPERIMENT_PATH)

    # results = train_lof(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_iso_forest(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)
