from time import time_ns
from typing import Callable, Iterable, Dict, List, Union
import json
import pickle
import numpy as np

from src.models.metrics import get_metrics


def time_decorator(function: Callable):
    def wrapper(*arg, **kw):
        t1 = time_ns()
        ret = function(*arg, **kw)
        t2 = time_ns()
        return ret, (t2 - t1) * 1e-9

    return wrapper


def save_experiment(data: Iterable, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def create_experiment_report(metrics: Dict, hyperparameters: Dict) -> Dict:
    return {
        'metrics': metrics,
        'hyperparameters': hyperparameters
    }


def create_checkpoint(data: Iterable, file_path: str):
    save_experiment(data, file_path)


def load_pickle_file(file_path: str) -> Union[List, Dict]:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def find_optimal_threshold(y_true: np.array, y_pred: np.array) -> tuple:
    ret = {}
    for th in set(y_pred[y_true == 1]):
        tmp = convert_predictions(y_pred, th)
        f1 = get_metrics(y_true, tmp)['f1_score']
        ret[th] = f1
    return max(ret.items(), key=lambda x: x[1])


def convert_predictions(y_pred: np.array, theta: float) -> np.array:
    ret = np.zeros(shape=y_pred.shape)
    ret[y_pred >= theta] = 1
    return ret
