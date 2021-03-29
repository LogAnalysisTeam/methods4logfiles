from __future__ import annotations

import numpy as np
from typing import List, Dict
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.models.autoencoder import AutoEncoder
from src.models.metrics import metrics_report, get_metrics
from src.visualization.visualization import visualize_distribution_with_labels
from src.data.hdfs import load_labels
from src.features.feature_extractor import FeatureExtractor
from src.models.train_baseline_models import get_labels_from_csv
from src.models.utils import load_pickle_file, find_optimal_threshold, classify, create_checkpoint, \
    create_experiment_report, save_experiment

SEED = 160121
np.random.seed(SEED)

EXPERIMENT_PATH = '../../models/AE-hyperparameters-Drain3-HDFS1.json'


def generate_layer_settings(size: int) -> List:
    ret = []
    for i in range(size):
        layers = []

        n_encoder = np.random.randint(1, 6)
        layers_encoder = np.random.randint(8, 201, size=n_encoder)
        layers_encoder.sort(kind='mergesort')
        layers.extend(layers_encoder.tolist()[::-1])  # descending

        n_decoder = np.random.randint(1, 6)
        layers_decoder = np.random.randint(8, 201, size=n_decoder)
        layers_decoder.sort(kind='mergesort')
        layers.extend(layers_decoder.tolist())  # ascending

        ret.append(layers)
    return ret


def train_autoencoder(x_train: Dict, x_test: Dict, y_train: np.array, y_test: np.array) -> Dict:
    fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    y_train = get_labels_from_csv(y_train, x_train.keys())
    y_test = get_labels_from_csv(y_test, x_test.keys())
    x_train = fe.fit_transform(x_train)
    x_test = fe.transform(x_test)

    model = AutoEncoder()
    n_experiments = 100
    params = {
        'epochs': np.random.choice(np.arange(1, 10), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.1), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_dim': [48] * n_experiments,
        'layers': generate_layer_settings(n_experiments),
        'dropout': np.random.uniform(0, 0.5, size=n_experiments).tolist()
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def random_search(data_and_labels: tuple, model: AutoEncoder, params: Dict) -> Dict:
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


if __name__ == '__main__':
    # X_train = load_pickle_file('/home/martin/bdip25/data/processed/HDFS1/X-train-HDFS1-cv1-1-block.npy')
    # X_val = load_pickle_file('/home/martin/bdip25/data/processed/HDFS1/X-val-HDFS1-cv1-1-block.npy')
    # y_train = np.load('/home/martin/bdip25/data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    # y_val = np.load('/home/martin/bdip25/data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')
    X_train = load_pickle_file('../../data/interim/HDFS1/train-data-Drain3-HDFS1-cv1-1.binlog')
    y_train = load_labels('../../data/interim/HDFS1/train-labels-HDFS1-cv1-1.csv')
    X_val = load_pickle_file('../../data/interim/HDFS1/val-data-Drain3-HDFS1-cv1-1.binlog')
    y_val = load_labels('../../data/interim/HDFS1/val-labels-HDFS1-cv1-1.csv')

    results = train_autoencoder(X_train, X_val, y_train, y_val)
    save_experiment(results, EXPERIMENT_PATH)

    #
    # train_autoencoder(X_val, X_val, y_val, y_val)
    #
    # # y_train = get_labels_from_csv(y_train, X_train.keys())
    # y_val = get_labels_from_csv(y_val, X_val.keys())
    #
    # # do standardization, normalization here!!!!
    #
    # # sc = CustomMinMaxScaler()
    # # X_train = sc.fit_transform(X_train)
    # # X_val = sc.transform(X_val)
    #
    # fe = FeatureExtractor(method='tf-idf', preprocessing='mean')
    # # X_train = fe.fit_transform(X_train).astype(np.float32)
    # X_val = fe.fit_transform(X_val).astype(np.float32)
    #
    # X = X_val[np.random.randint(0, 50000, size=45000)]
    #
    # model = AutoEncoder(epochs=5, learning_rate=0.0001)
    # model._initialize_model(39, [128, 64, 8, 32], 0.3)
    # model.fit(X)
    # # torch.save(model, 'model.pth')
    #
    # test_indices = np.random.randint(50000, 51000, size=500)
    # y_pred = model.predict(X_val[test_indices])
    #
    # for th in sorted(y_pred[y_val[test_indices] == 1]):
    #     tmp = np.zeros(shape=y_pred.shape)
    #     tmp[y_pred > th] = 1
    #     print('Threshold:', th)
    #     metrics_report(y_val[test_indices], tmp)
    #
    # visualize_distribution_with_labels(y_pred, y_val[test_indices], to_file=False)

    # results = train_autoencoder(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # model = VanillaTCN(epochs=15, learning_rate=0.0001)
    # model.fit(X)
    #
    # test_indices = np.random.randint(0, 50000, size=500)
    # y_pred = model.predict(X_val[test_indices])
    # visualize_distribution_with_labels(y_pred, y_val[test_indices], to_file=True)
