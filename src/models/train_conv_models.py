from __future__ import annotations

import numpy as np
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.models.vanilla_tcnn import VanillaTCN
from src.models.autoencoder_cnn import CNN1D
from src.visualization.visualization import visualize_distribution_with_labels
from src.models.metrics import metrics_report, get_metrics
from src.models.utils import create_experiment_report, save_experiment, create_checkpoint, load_pickle_file, \
    find_optimal_threshold, convert_predictions, get_encoder_size

SEED = 160121
np.random.seed(SEED)

EXPERIMENT_PATH = '../../models/CNN1D-bottleneck-hyperparameters-embeddings-HDFS1.json'


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


def generate_layer_settings(input_dim: int, size: int) -> List:
    ret = []
    for i in range(size):
        layers = []

        n_encoder = np.random.randint(1, 4)
        layers_encoder = np.random.randint(16, 301, size=n_encoder)
        layers_encoder.sort(kind='mergesort')
        layers.extend(layers_encoder.tolist()[::-1])  # descending

        n_decoder = np.random.randint(1, 4)
        layers_decoder = np.random.randint(16, 301, size=n_decoder)
        layers_decoder.sort(kind='mergesort')
        layers.extend(layers_decoder.tolist())  # ascending
        layers[-1] = input_dim

        ret.append(layers)
    return ret


def get_min_window_size(kernel: int, maxpool: int, n_encoder_layers: int):
    # time complexity might be improved with binary search O(log(n)) instead of O(n)
    for input_dim in range(1, 500):
        output_dim = input_dim
        for _ in range(n_encoder_layers):
            output_dim = output_dim - kernel + 1  # Conv1d
            output_dim //= maxpool  # MaxPool1d

        if output_dim > 0:
            return input_dim


def get_window_size(encoder_kernels: List, layers: List) -> List:
    maxpool = 2  # fixed also in the PyTorch model

    windows = []
    for i, kernel in enumerate(encoder_kernels):
        n_encoder_layers = get_encoder_size(layers[i])
        min_window_size = get_min_window_size(kernel, maxpool, n_encoder_layers)
        windows.append(np.random.randint(min_window_size, min_window_size + 32))
    return windows


def train_tcnn(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = VanillaTCN()
    n_experiments = 100
    params = {
        'epochs': np.random.choice(np.arange(1, 10), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.1), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_shape': [100] * n_experiments,
        'layers': generate_layer_settings(100, n_experiments),
        'kernel_size': np.random.choice([2 * i + 1 for i in range(1, 8)], size=n_experiments).tolist(),
        'dropout': np.random.uniform(0, 0.5, size=n_experiments).tolist()
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def train_cnn1d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = CNN1D()
    n_experiments = 100
    embeddings_dim = 100

    encoder_kernel_sizes = np.random.choice([2 * i + 1 for i in range(1, 4)], size=n_experiments).tolist()
    layers = generate_layer_settings(embeddings_dim, n_experiments)
    params = {
        'epochs': np.random.choice(np.arange(1, 10), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.1), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_shape': [embeddings_dim] * n_experiments,
        'layers': layers,
        'encoder_kernel_size': encoder_kernel_sizes,
        'decoder_kernel_size': np.random.choice([2 * i + 1 for i in range(2, 7)], size=n_experiments).tolist(),
        'window': get_window_size(encoder_kernel_sizes, layers)
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def random_search(data_and_labels: tuple, model: Union[VanillaTCN, CNN1D], params: Dict) -> Dict:
    x_train, x_test, _, y_test = data_and_labels

    scores = []
    for conf in zip(*params.values()):
        kwargs = {k: val for k, val in zip(params.keys(), conf)}

        model.set_params(**kwargs)

        print(f'Model current hyperparameters are: {kwargs}.')

        model.fit(x_train)
        y_pred = model.predict(x_test)  # return reconstruction errors

        theta, f1 = find_optimal_threshold(y_test, y_pred)
        y_pred = convert_predictions(y_pred, theta)
        metrics_report(y_test, y_pred)
        scores.append(create_experiment_report(get_metrics(y_test, y_pred), kwargs))
        create_checkpoint({'experiments': scores}, EXPERIMENT_PATH)
    return {
        'experiments': scores
    }


def train_window(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    scores = []
    for w in range(1, 50, 2):
        print('Window:', w)
        model = VanillaTCN(epochs=1, window=w)

        model.fit(x_train[y_train == 0])
        y_pred = model.predict(x_test)  # return reconstruction errors

        theta, f1 = find_optimal_threshold(y_test, y_pred)
        y_pred = convert_predictions(y_pred, theta)
        metrics_report(y_test, y_pred)
        scores.append(create_experiment_report(get_metrics(y_test, y_pred), {'window': w}))
        create_checkpoint({'experiments': scores}, '../../models/TCN-cropped-window-embeddings-HDFS1.json')
    return {
        'experiments': scores
    }


if __name__ == '__main__':
    debug = False
    if debug:
        X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.npy')
        y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')

        # train_window(X_val[:45000], X_val[45000:], y_val[:45000], y_val[45000:])

        # train_tcnn(X_val[:1000], X_val[:500], y_val[:1000], y_val[:500])
        # exit()

        train_cnn1d(None, None, None, None)

        sc = CustomMinMaxScaler()
        X_train = sc.fit_transform(X_val)
        # X_train = np.asarray(X_val)

        # from src.models.vanilla_tcnn import TrimmedDataset
        # x = TrimmedDataset(X_train)
        #
        # n, counts = np.unique([x.shape[0] for x in X_train], return_counts=True)
        # print(sorted(zip(n, counts), key=lambda x: -x[1]))
        #
        # sh = (25, 100)
        # y_val = np.asarray([y for x, y in zip(X_train, y_val) if x.shape == sh])
        # X_train = np.asarray([x for x in X_train if x.shape == sh])
        #
        # X = X_train[y_val == 0][:2000]
        X = X_train[y_val == 0][:40000]

        # model = VanillaTCN(epochs=1, learning_rate=0.00001)
        model = CNN1D(epochs=3, learning_rate=0.001)
        # model._initialize_model(100, [64, 32, 64], 3, 6)
        model.fit(X)

        # test_indices = list(range(2000, len(X_train))) + [i for i in range(len(X_train)) if y_val[i] == 1 and i < 2000]
        test_indices = np.random.randint(45000, 51000, size=500)
        y_pred = model.predict(X_train[test_indices])

        for th in sorted(y_pred[y_val[test_indices] == 1]):
            tmp = np.zeros(shape=y_pred.shape)
            tmp[y_pred > th] = 1
            print('Threshold:', th)
            metrics_report(y_val[test_indices], tmp)

        visualize_distribution_with_labels(y_pred, y_val[test_indices], to_file=False)
        exit()

    X_train = load_pickle_file('../../data/processed/HDFS1/X-train-HDFS1-cv1-1-block.npy')
    X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.npy')
    y_train = np.load('../../data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')

    # results = train_window(X_train, X_val, y_train, y_val)
    # save_experiment(results, '../../models/TCN-cropped-window-embeddings-HDFS1.json')

    # results = train_tcnn(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    results = train_cnn1d(X_train, X_val, y_train, y_val)
    save_experiment(results, EXPERIMENT_PATH)
