from __future__ import annotations

import torch
import numpy as np
import uuid
import os
from typing import List, Dict, Union
from sklearn.preprocessing import MinMaxScaler

from src.models.vanilla_tcnn import VanillaTCN
from src.models.autoencoder_tcnn import AETCN
from src.models.cnn1d import CNN1D
from src.models.cnn2d import CNN2D
from src.models.tcnn_cnn1d import TCNCNN1D
from src.models.sa_cnn1d import SACNN1D
from src.models.sa_cnn2d import SACNN2D
from src.models.metrics import metrics_report, get_metrics
from src.models.utils import create_experiment_report, create_checkpoint, save_experiment, load_pickle_file, \
    find_optimal_threshold, convert_predictions, get_encoder_size, generate_layer_settings, get_1d_window_size, \
    get_2d_kernels, get_2d_window_size, get_encoder_heads, get_decoder_heads, get_bottleneck_dim, load_experiment, \
    create_model_path

SEED = 160121
np.random.seed(SEED)

DIR_TO_EXPERIMENTS = '../../models/aetcn'
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
    n_experiments = 100
    embeddings_dim = x_train[0].shape[1]

    encoder_kernel_sizes = get_2d_kernels([2 * i + 1 for i in range(1, 5)], [2 * i + 1 for i in range(1, 4)],
                                          n_experiments)
    decoder_kernel_sizes = get_2d_kernels([2 * i + 1 for i in range(2, 8)], [2 * i + 1 for i in range(1, 5)],
                                          n_experiments)
    layers = generate_layer_settings(embeddings_dim, n_experiments)
    params = {
        'epochs': np.random.choice(np.arange(1, 6), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.5), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_shape': [embeddings_dim] * n_experiments,
        'layers': layers,
        'encoder_kernel_size': encoder_kernel_sizes,
        'decoder_kernel_size': decoder_kernel_sizes,
        'window': get_2d_window_size(encoder_kernel_sizes, layers)
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def train_tcnn_cnn1d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = TCNCNN1D()
    n_experiments = 100
    embeddings_dim = x_train[0].shape[1]

    encoder_kernel_sizes = np.random.choice([2 * i + 1 for i in range(1, 4)], size=n_experiments).tolist()
    layers = generate_layer_settings(embeddings_dim, n_experiments)
    params = {
        'epochs': np.random.choice(np.arange(1, 10), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.5), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_shape': [embeddings_dim] * n_experiments,
        'layers': layers,
        'encoder_kernel_size': encoder_kernel_sizes,
        'tcn_kernel_size': np.random.choice([2 * i + 1 for i in range(1, 4)], size=n_experiments).tolist(),
        'decoder_kernel_size': np.random.choice([2 * i + 1 for i in range(2, 7)], size=n_experiments).tolist(),
        'window': get_1d_window_size(encoder_kernel_sizes, layers, lambda x: len(x[0])),
        'dropout': np.random.uniform(0, 0.5, size=n_experiments).tolist()
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def train_sa_cnn1d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = SACNN1D()
    n_experiments = 100
    embeddings_dim = x_train[0].shape[1]

    encoder_kernel_sizes = np.random.choice([2 * i + 1 for i in range(1, 4)], size=n_experiments).tolist()
    layers = generate_layer_settings(embeddings_dim, n_experiments)
    params = {
        'epochs': np.random.choice(np.arange(1, 10), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.5), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_shape': [embeddings_dim] * n_experiments,
        'layers': layers,
        'encoder_kernel_size': encoder_kernel_sizes,
        'decoder_kernel_size': np.random.choice([2 * i + 1 for i in range(2, 7)], size=n_experiments).tolist(),
        'encoder_heads': get_encoder_heads(layers),
        'decoder_heads': get_decoder_heads(layers),
        'window': get_1d_window_size(encoder_kernel_sizes, layers, get_encoder_size),
        'dropout': np.random.uniform(0, 0.3, size=n_experiments).tolist()
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def train_sa_cnn2d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = SACNN2D()
    n_experiments = 100
    embeddings_dim = x_train[0].shape[1]

    encoder_kernel_sizes = get_2d_kernels([2 * i + 1 for i in range(1, 5)], [2 * i + 1 for i in range(1, 4)],
                                          n_experiments)
    decoder_kernel_sizes = get_2d_kernels([2 * i + 1 for i in range(2, 8)], [2 * i + 1 for i in range(1, 5)],
                                          n_experiments)
    layers = generate_layer_settings(embeddings_dim, n_experiments)
    params = {
        'epochs': np.random.choice(np.arange(1, 6), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.5), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_shape': [embeddings_dim] * n_experiments,
        'layers': layers,
        'encoder_kernel_size': encoder_kernel_sizes,
        'decoder_kernel_size': decoder_kernel_sizes,
        'bottleneck_dim': get_bottleneck_dim(layers),
        'window': get_2d_window_size(encoder_kernel_sizes, layers)
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def random_search(data_and_labels: tuple, model: Union[VanillaTCN, AETCN, CNN1D, CNN2D, TCNCNN1D, SACNN1D, SACNN2D],
                  params: Dict) -> Dict:
    x_train, x_test, _, y_test = data_and_labels

    scores = []
    for experiment in params['experiments']:
        model.set_params(**experiment['hyperparameters'])

        print(f'Model current hyperparameters are: {experiment["hyperparameters"]}.')

        model.fit(x_train)
        y_pred = model.predict(x_test)  # return reconstruction errors

        theta, f1 = find_optimal_threshold(y_test, y_pred)
        y_pred = convert_predictions(y_pred, theta)
        metrics_report(y_test, y_pred)

        print('old:', experiment['metrics'], 'new:', get_metrics(y_test, y_pred))

        model_path = create_model_path(DIR_TO_EXPERIMENTS, str(uuid.uuid4()))
        torch.save(model, model_path)

        res = create_experiment_report(get_metrics(y_test, y_pred), experiment['hyperparameters'], theta, model_path)
        scores.append(res)
        create_checkpoint({'experiments': scores}, EXPERIMENT_PATH)
    return {
        'experiments': scores
    }


if __name__ == '__main__':
    debug = False
    if debug:
        X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.npy')
        y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')

        train_cnn1d(X_val[:1000], X_val[:500], y_val[:1000], y_val[:500])

    X_train = load_pickle_file('../../data/processed/HDFS1/X-train-HDFS1-cv1-1-block.pickle')
    X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.pickle')
    y_train = np.load('../../data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')

    # results = train_window(X_train, X_val, y_train, y_val)
    # save_experiment(results, '../../models/TCN-cropped-window-embeddings-HDFS1.json')

    # results = train_tcnn(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_cnn1d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_cnn2d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_tcnn_cnn1d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    results = train_aetcnn(X_train, X_val, y_train, y_val)
    save_experiment(results, EXPERIMENT_PATH)

    # results = train_sa_cnn1d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_sa_cnn2d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)
