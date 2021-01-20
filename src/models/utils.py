from time import time_ns
from typing import Callable


def time_decorator(function: Callable):
    def wrapper(*arg, **kw):
        t1 = time_ns()
        ret = function(*arg, **kw)
        t2 = time_ns()
        return ret, (t2 - t1) * 1e-9

    return wrapper
