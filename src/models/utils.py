from time import time_ns
from typing import Callable, Iterable, List
import json
import numpy as np


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


def get_json_serializable_array(array: np.array, f_type: Callable) -> List:
    return [f_type(x) for x in array]
