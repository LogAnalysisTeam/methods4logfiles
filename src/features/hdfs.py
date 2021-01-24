import numpy as np
import fasttext
import os
import pandas as pd
from typing import Generator, Iterable
from collections import defaultdict

from src.data.hdfs import load_data, load_labels


def load_fold_pairs(data_dir: str, n_folds: int, fold: str) -> Generator:
    for i in range(1, n_folds + 1):
        data_filename = f'{fold}-data-HDFS1-cv{i}-{n_folds}.log'
        labels_filename = f'{fold}-labels-HDFS1-cv{i}-{n_folds}.csv'
        yield load_data(os.path.join(data_dir, data_filename)), load_labels(os.path.join(data_dir, labels_filename))


def get_number_of_splits(filename: str) -> int:
    return int(os.path.splitext(filename)[0][-1])


def get_embeddings_per_log(data: defaultdict, model: fasttext.FastText) -> np.ndarray:
    # create embeddings per log but at first remove '\n' (newline character) from the end
    embeddings = [model.get_sentence_vector(log.rstrip()) for logs in data.values() for log in logs]
    return np.asarray(embeddings)


def get_embeddings_per_block(data: defaultdict, model: fasttext.FastText) -> np.ndarray:
    # create embeddings per block but at first remove '\n' (newline character) from the end
    embeddings = [np.asarray([model.get_sentence_vector(log.rstrip()) for log in logs], dtype='object')
                  for logs in data.values()]
    return np.asarray(embeddings)


def get_labels_from_keys_per_log(data: defaultdict, labels: pd.DataFrame) -> np.array:
    size = sum(len(logs) for logs in data.values())
    ground_truth = np.zeros(shape=size, dtype=np.int8)
    idx = 0
    for row in labels.itertuples(index=False):
        block_id, is_anomalous = row
        block_len = len(data[block_id])

        if is_anomalous:
            ground_truth[idx:idx + block_len] = 1  # mark all affected logs belonging to the trace as anomaly
        idx += block_len
    return ground_truth


def get_labels_from_keys_per_block(labels: pd.DataFrame) -> np.array:
    return labels['Label'].to_numpy(dtype=np.int8)


def check_order(keys: Iterable, block_ids: pd.Series):
    if not all(x == y for x, y in zip(keys, block_ids.tolist())):
        raise AssertionError('Data keys and block ids are not in the same order')


def create_embeddings(data_dir: str, output_dir: str, fasttext_model_path: str, per_block: bool):
    n_folds = get_number_of_splits(fasttext_model_path)
    model = fasttext.load_model(fasttext_model_path)

    for fold in ['train', 'val']:
        for idx, (data, labels) in enumerate(load_fold_pairs(data_dir, n_folds, fold), start=1):
            # temporal check
            # check data.keys() and labels['BlockId'] are in the same order
            check_order(data.keys(), labels['BlockId'])

            if per_block:
                embeddings = get_embeddings_per_block(data, model)
                ground_truth = get_labels_from_keys_per_block(labels)
            else:
                embeddings = get_embeddings_per_log(data, model)
                ground_truth = get_labels_from_keys_per_log(data, labels)

            method = 'block' if per_block else 'log'

            np.save(os.path.join(output_dir, f'X-{fold}-HDFS1-cv{idx}-{n_folds}-{method}.npy'), embeddings)
            np.save(os.path.join(output_dir, f'y-{fold}-HDFS1-cv{idx}-{n_folds}-{method}.npy'), ground_truth)
