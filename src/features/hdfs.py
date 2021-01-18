import numpy as np
import fasttext
import os
import pandas as pd
from typing import Generator, Iterable
from collections import defaultdict

from src.data.hdfs import SEED, load_data, load_labels


def load_fold_pairs(data_dir: str, n_folds: int, fold: str) -> Generator:
    for i in range(1, n_folds + 1):
        data_filename = f'{fold}-data-HDFS1-cv-{i}-{n_folds}.log'
        labels_filename = f'{fold}-labels-HDFS1-cv-{i}-{n_folds}.csv'
        yield load_data(os.path.join(data_dir, data_filename)), load_labels(os.path.join(data_dir, labels_filename))


def get_number_of_splits(filename: str) -> int:
    return int(os.path.splitext(filename)[0][-1])


def get_embeddings_of_logs(data: defaultdict, model: fasttext.FastText) -> np.ndarray:
    embeddings = [model.get_sentence_vector(log) for logs in data.values() for log in logs]
    return np.asarray(embeddings)


def get_labels_from_keys(data: defaultdict, labels: pd.DataFrame) -> np.array:
    size = sum(len(logs) for logs in data.values())
    ground_truth = np.zeros(shape=size, dtype=np.float32)
    idx = 0
    for block_id, is_anomalous in labels.iterrows():
        if is_anomalous:
            ground_truth[idx:idx + len(data[block_id])] = 1  # mark all affected logs belonging to the trace as anomaly
        idx += len(data[block_id])
    return ground_truth


def check_order(keys: Iterable, block_ids: pd.Series):
    if not all(x == y for x, y in zip(keys, block_ids.tolist())):
        raise AssertionError('Data keys and block ids are not in the same order')


def create_embeddings(data_dir: str, output_dir: str, fasttext_model_path: str):
    n_folds = get_number_of_splits(fasttext_model_path)
    model = fasttext.load_model(fasttext_model_path)

    for fold in ['val']:  # ['train', 'val']:
        for data, labels in load_fold_pairs(data_dir, n_folds, fold):
            # check data.keys() and labels['BlockId'] are in the same order
            check_order(data.keys(), labels['BlockId'])
            pass
