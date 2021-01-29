from time import time_ns
from typing import Callable, Iterable
import pickle


def time_decorator(function: Callable):
    def wrapper(*arg, **kw):
        t1 = time_ns()
        ret = function(*arg, **kw)
        t2 = time_ns()
        return ret, (t2 - t1) * 1e-9

    return wrapper


def save_experiment(data: Iterable, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
