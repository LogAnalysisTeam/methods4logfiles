from __future__ import annotations

import numpy as np
from typing import List, Dict

from src.models.metrics import metrics_report, get_metrics
from src.models.transformer import TransformerAutoEncoder
from src.models.datasets import CustomMinMaxScaler
from src.models.utils import load_pickle_file, find_optimal_threshold, classify, create_checkpoint, \
    create_experiment_report, save_experiment, get_all_divisors, get_normal_dist

SEED = 160121
np.random.seed(SEED)

EXPERIMENT_PATH = '../../models/TAE-hyperparameters-embeddings-HDFS1.json'


def train_transformer(x_train: List, x_test: List, y_train: np.array, y_test: np.array) -> Dict:
    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = TransformerAutoEncoder()
    n_experiments = 100
    embeddings_dim = x_train[0].shape[1]

    divisors = get_all_divisors(embeddings_dim)
    params = {
        'epochs': np.random.choice(np.arange(1, 5), size=n_experiments).tolist(),
        'learning_rate': np.random.choice(10 ** np.linspace(-4, -0.5), size=n_experiments).tolist(),
        'batch_size': np.random.choice([2 ** i for i in range(3, 8)], size=n_experiments).tolist(),
        'input_dim': [embeddings_dim] * n_experiments,
        'heads': np.random.choice(divisors, size=n_experiments, p=get_normal_dist(divisors)).tolist(),
        'n_encoders': np.random.randint(1, 5, size=n_experiments).tolist(),
        'n_decoders': np.random.randint(1, 5, size=n_experiments).tolist(),
        'dim_feedforward': np.random.randint(100, 2000, size=n_experiments).tolist(),
        'window': np.random.randint(10, 100, size=n_experiments).tolist(),
        'dropout': np.random.uniform(0, 0.5, size=n_experiments).tolist()
    }
    evaluated_hyperparams = random_search((x_train[y_train == 0], x_test, None, y_test), model, params)
    return evaluated_hyperparams


def random_search(data_and_labels: tuple, model: TransformerAutoEncoder, params: Dict) -> Dict:
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
        # visualize_distribution_with_labels(y_pred, y_test, to_file=False)
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_pred))
        create_checkpoint({'experiments': scores}, EXPERIMENT_PATH)
    return {
        'experiments': scores
    }


if __name__ == '__main__':
    X_train = load_pickle_file('../../data/processed/HDFS1/X-train-HDFS1-cv1-1-block.pickle')
    X_val = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.pickle')
    y_train = np.load('../../data/processed/HDFS1/y-train-HDFS1-cv1-1-block.npy')
    y_val = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')

    results = train_transformer(X_train, X_val, y_train, y_val)
    save_experiment(results, EXPERIMENT_PATH)
