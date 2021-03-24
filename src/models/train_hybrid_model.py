from __future__ import annotations

import numpy as np
import torch
from typing import List, Dict

from src.models.metrics import metrics_report, get_metrics
from src.features.feature_extractor import FeatureExtractor
from src.models.train_conv_models import CustomMinMaxScaler
from src.models.train_baseline_models import get_labels_from_csv
from src.models.utils import load_pickle_file, find_optimal_threshold, convert_predictions, create_checkpoint, \
    create_experiment_report, save_experiment

SEED = 160121
np.random.seed(SEED)

EXPERIMENT_PATH = '../../models/IF-AETCN-hybrid-hyperparameters-HDFS1.json'


def train_hybrid_model(x_train: Dict, x_test: Dict, y_train: np.array, y_test: np.array) -> Dict:
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
        y_pred = convert_predictions(y_pred, theta)
        metrics_report(y_test, y_pred)
        scores.append(create_experiment_report(get_metrics(y_test, y_pred), kwargs))
        create_checkpoint({'experiments': scores}, EXPERIMENT_PATH)
    return {
        'experiments': scores
    }


def get_extracted_features(x_train: List, x_test: List, y_train: np.array, y_test: np.array):
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = torch.load('../../models/aetcn/4f5f4682-1ca5-400a-a340-6243716690c0.pt')

    y_pred = model.predict(x_test)  # return reconstruction errors
    train_features = model.extract_features(x_train)
    test_features = model.extract_features(x_test)

    theta, f1 = find_optimal_threshold(y_test, y_pred)
    print(theta)
    y_pred = convert_predictions(y_pred, theta)
    metrics_report(y_test, y_pred)
    return train_features, test_features


if __name__ == '__main__':
    X_train = load_pickle_file('../../data/processed/HDFS1/X-train-HDFS1-cv1-1-block.pickle')
    X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.pickle')
    y_train = np.load('../../data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')

    X_train, X_val = get_extracted_features(X_train, X_val, y_train, y_val)

    np.save('../../data/processed/HDFS1/X-train-HDFS1-interim-features.npy', X_train)
    np.save('../../data/processed/HDFS1/X-train-HDFS1-interim-features.npy', X_val)

    # results = train_hybrid_model(X_val[:1000], X_val[:500], y_val[:1000], y_val[:500])
    # save_experiment(results, EXPERIMENT_PATH)
