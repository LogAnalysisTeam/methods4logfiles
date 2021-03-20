from __future__ import annotations

import numpy as np
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.models.vanilla_tcnn import VanillaTCN
from src.models.autoencoder_tcnn import AETCN
from src.models.cnn1d import CNN1D
from src.models.cnn2d import CNN2D
from src.models.tcnn_cnn1d import TCNCNN1D
from src.models.sa_cnn1d import SACNN1D
from src.models.sa_cnn2d import SACNN2D
from src.visualization.visualization import visualize_distribution_with_labels
from src.models.metrics import metrics_report, get_metrics
from src.models.utils import create_experiment_report, create_checkpoint, save_experiment, load_pickle_file, \
    find_optimal_threshold, convert_predictions, get_encoder_size, generate_layer_settings, get_1d_window_size, \
    get_2d_kernels, get_2d_window_size, get_encoder_heads, get_decoder_heads, get_bottleneck_dim

SEED = 160121
np.random.seed(SEED)

EXPERIMENT_PATH = '../../models/SACNN2D-hyperparameters-embeddings-clipping-HDFS1.json'


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


class CustomStandardScaler(StandardScaler):
    def __init__(self):
        super().__init__()
        self.std = None
        self.mean = None

    @staticmethod
    def _flatten_dataset(dataset: List) -> np.array:
        ret = np.array([embedding for block in dataset for embedding in block])
        return ret

    def fit(self, X: List, y=None) -> CustomStandardScaler:
        data = self._flatten_dataset(X)
        self.std = data.std(axis=0)
        self.mean = data.mean(axis=0)
        return self

    def fit_transform(self, X: List, y: np.array = None, **fit_params) -> np.array:
        return self.fit(X).transform(X)

    def transform(self, X: List, copy=None) -> np.array:
        # (X - X.mean(axis=0)) / X.std(axis=0)
        return np.asarray([(x - self.mean) / self.std for x in X], dtype='object')


def train_tcnn(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = VanillaTCN()
    n_experiments = 100
    embeddings_dim = x_train[0].shape[1]

    params = {
        'epochs': np.random.choice(np.arange(1, 10), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.5), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_shape': [embeddings_dim] * n_experiments,
        'layers': generate_layer_settings(embeddings_dim, n_experiments),
        'kernel_size': np.random.choice([2 * i + 1 for i in range(1, 6)], size=n_experiments).tolist(),
        'window': np.random.randint(10, 100, size=n_experiments).tolist(),
        'dropout': np.random.uniform(0, 0.5, size=n_experiments).tolist()
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def train_aetcnn(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = AETCN()
    n_experiments = 100
    embeddings_dim = x_train[0].shape[1]

    params = {
        'epochs': np.random.choice(np.arange(1, 10), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.5), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_shape': [embeddings_dim] * n_experiments,
        'layers': generate_layer_settings(embeddings_dim, n_experiments),
        'kernel_size': np.random.choice([2 * i + 1 for i in range(1, 6)], size=n_experiments).tolist(),
        'window': np.random.randint(10, 100, size=n_experiments).tolist(),
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
        'window': get_1d_window_size(encoder_kernel_sizes, layers, get_encoder_size)
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
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

        train_sa_cnn2d(X_val[:1000], X_val[:500], y_val[:1000], y_val[:500])
        # exit()

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

        # from torchsummary import summary
        model = SACNN2D(epochs=1, learning_rate=0.001)
        # model = TCNCNN1DPyTorch(100, 35, [], 0, 0)
        # print(summary(model, (100, 35)))
        # model._initialize_model(100, [16, 32, 64, 32, 16], 3, 7)
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

    # results = train_aetcnn(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    # results = train_sa_cnn1d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    results = train_sa_cnn2d(X_train, X_val, y_train, y_val)
    save_experiment(results, EXPERIMENT_PATH)
