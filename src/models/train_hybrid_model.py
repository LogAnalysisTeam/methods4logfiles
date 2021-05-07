from __future__ import annotations

import numpy as np
import torch
import os
from typing import List, Dict, Union
from sklearn.ensemble import IsolationForest

from src.models.metrics import metrics_report, get_metrics
from src.models.autoencoder import AutoEncoder
from src.models.datasets import CustomMinMaxScaler
from src.models.utils import load_pickle_file, find_optimal_threshold, classify, create_checkpoint, \
    create_experiment_report, save_experiment

SEED = 160121
np.random.seed(SEED)

EXPERIMENT_PATH = '../../models/IF-AETCN-hybrid-hyperparameters-HDFS1.json'


def generate_layer_settings(n_experiments: int) -> List:
    ret = []
    for i in range(n_experiments):
        layers = []

        n_encoder = np.random.randint(1, 6)
        layers_encoder = np.random.randint(10, 501, size=n_encoder)
        layers_encoder.sort(kind='mergesort')
        layers.extend(layers_encoder.tolist()[::-1])  # descending

        n_decoder = np.random.randint(1, 6)
        layers_decoder = np.random.randint(10, 501, size=n_decoder)
        layers_decoder.sort(kind='mergesort')
        layers.extend(layers_decoder.tolist())  # ascending

        ret.append(layers)
    return ret


def train_hybrid_model_ae(x_train: np.ndarray, x_test: np.ndarray, y_train: np.array, y_test: np.array) -> Dict:
    model = AutoEncoder()

    n_experiments = 100
    embeddings_dim = x_train.shape[1]

    params = {
        'epochs': np.random.choice(np.arange(1, 10), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -1), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_dim': [embeddings_dim] * n_experiments,
        'layers': generate_layer_settings(n_experiments),
        'dropout': np.random.uniform(0, 0.5, size=n_experiments).tolist()
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def train_hybrid_model_if(x_train: np.ndarray, x_test: np.ndarray, y_train: np.array, y_test: np.array) -> Dict:
    clf = IsolationForest(bootstrap=True, n_jobs=os.cpu_count(), random_state=SEED)

    n_experiments = 100
    embeddings_dim = x_train.shape[1]

    params = {
        'n_estimators': np.random.randint(100, 2000, size=n_experiments).tolist(),
        'max_samples': np.random.uniform(0.001, 1, size=n_experiments).tolist(),
        'max_features': np.random.randint(1, embeddings_dim + 1, size=n_experiments).tolist()
    }
    evaluated_hyperparams = random_search((x_train, x_test, None, y_test), clf, params)
    return evaluated_hyperparams


def random_search(data_and_labels: tuple, model: Union[AutoEncoder, IsolationForest], params: Dict) -> Dict:
    x_train, x_test, _, y_test = data_and_labels

    scores = []
    for conf in zip(*params.values()):
        kwargs = {k: val for k, val in zip(params.keys(), conf)}

        model.set_params(**kwargs)

        print(f'Model current hyperparameters are: {kwargs}.')

        model.fit(x_train)
        y_pred = model.predict(x_test)  # return reconstruction errors

        theta, f1 = find_optimal_threshold(y_test, y_pred)
        y_pred = classify(y_pred, theta)
        metrics_report(y_test, y_pred)
        scores.append(create_experiment_report(get_metrics(y_test, y_pred), kwargs))
        create_checkpoint({'experiments': scores}, EXPERIMENT_PATH)
    return {
        'experiments': scores
    }


def get_extracted_features(x_train: List, x_val: List, x_test: List, y_val: np.array):
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_val = sc.transform(x_val)
    x_test = sc.transform(x_test)

    model = torch.load('../../models/aetcn/5d9ad591-6d3c-428f-894f-02af96ca1930.pt')

    y_pred = model.predict(x_val)  # return reconstruction errors
    train_features = model.extract_features(x_train).astype(dtype=np.float32)
    val_features = model.extract_features(x_val).astype(dtype=np.float32)
    test_features = model.extract_features(x_test).astype(dtype=np.float32)

    theta, f1 = find_optimal_threshold(y_val, y_pred)
    y_pred = classify(y_pred, theta)
    metrics_report(y_val, y_pred)
    return train_features, val_features, test_features


if __name__ == '__main__':
    X_train = load_pickle_file('../../data/processed/HDFS1/X-train-HDFS1-cv1-1-block.pickle')
    X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.pickle')
    X_test = load_pickle_file('../../data/processed/HDFS1/X-test-HDFS1-block.pickle')
    y_train = np.load('../../data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')
    y_test = np.load('../../data/processed/HDFS1/y-test-HDFS1-block.npy')

    train_path = '../../data/processed/HDFS1/X-train-HDFS1-interim-features.npy'
    val_path = '../../data/processed/HDFS1/X-val-HDFS1-interim-features.npy'
    test_path = '../../data/processed/HDFS1/X-test-HDFS1-interim-features.npy'

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        X_train = np.load(train_path)
        X_val = np.load(val_path)
    else:
        X_train, X_val, X_test = get_extracted_features(X_train, X_val, X_test, y_val)
        np.save(train_path, X_train)
        np.save(val_path, X_val)
        np.save(test_path, X_test)

    results = train_hybrid_model_if(X_train, X_val, y_train, y_val)
    save_experiment(results, EXPERIMENT_PATH)
