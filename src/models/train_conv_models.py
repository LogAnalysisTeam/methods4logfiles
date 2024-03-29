from __future__ import annotations

import numpy as np
from typing import List, Dict, Union

from src.models.vanilla_tcnn import VanillaTCN
from src.models.autoencoder_tcnn import AETCN
from src.models.autoencoder_cnn1d import AECNN1D
from src.models.cnn1d import CNN1D
from src.models.cnn2d import CNN2D
from src.models.tcnn_cnn1d import TCNCNN1D
from src.models.sa_cnn1d import SACNN1D
from src.models.sa_cnn2d import SACNN2D
from src.models.datasets import CustomMinMaxScaler
from src.visualization.visualization import visualize_distribution_with_labels
from src.models.metrics import metrics_report, get_metrics
from src.models.utils import create_experiment_report, create_checkpoint, save_experiment, load_pickle_file, \
    find_optimal_threshold, classify, get_encoder_size, generate_layer_settings, get_1d_window_size, \
    get_2d_kernels, get_2d_window_size, get_encoder_heads, get_decoder_heads, get_bottleneck_dim

SEED = 160121
np.random.seed(SEED)

EXPERIMENT_PATH = '../../models/AECNN1D-hyperparameters-embeddings-clipping-HDFS1.json'


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


def train_aecnn1d(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = AECNN1D()
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
        'bottleneck_dim': np.random.randint(100, 2001, size=n_experiments).tolist(),
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


def random_search(data_and_labels: tuple, model: Union[VanillaTCN, AETCN, CNN1D, AECNN1D, CNN2D, TCNCNN1D, SACNN1D,
                                                       SACNN2D], params: Dict) -> Dict:
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
        y_pred = classify(y_pred, theta)
        metrics_report(y_test, y_pred)
        scores.append(create_experiment_report(get_metrics(y_test, y_pred), {'window': w}))
        create_checkpoint({'experiments': scores}, '../../models/TCN-cropped-window-embeddings-HDFS1.json')
    return {
        'experiments': scores
    }


if __name__ == '__main__':
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

    # results = train_sa_cnn2d(X_train, X_val, y_train, y_val)
    # save_experiment(results, EXPERIMENT_PATH)

    results = train_aecnn1d(X_train, X_val, y_train, y_val)
    save_experiment(results, EXPERIMENT_PATH)
