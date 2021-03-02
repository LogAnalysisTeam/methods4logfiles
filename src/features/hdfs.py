import numpy as np
import fasttext
import os
import re
import pandas as pd
import pickle
from datetime import datetime
from typing import Generator, Iterable, List
from collections import defaultdict

from src.data.hdfs import load_data, load_labels


def save_to_file(data: List, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_fold_pairs(data_dir: str, n_folds: int, fold: str) -> Generator:
    for i in range(1, n_folds + 1):
        data_filename = f'{fold}-data-HDFS1-cv{i}-{n_folds}.log'
        labels_filename = f'{fold}-labels-HDFS1-cv{i}-{n_folds}.csv'
        yield load_data(os.path.join(data_dir, data_filename)), load_labels(os.path.join(data_dir, labels_filename))


def get_number_of_splits(filename: str) -> int:
    return int(os.path.splitext(filename)[0][-1])


def extract_datetime_from_line(regex: re.Pattern, line: str) -> str:
    res = regex.search(line)
    if not res:  # temporal check!!
        raise AssertionError('Nothing find!!!!!')
    return res.group()


def get_timestamp(timestamp: str) -> float:
    datetime_object = datetime.strptime(timestamp, '%y%m%d %H%M%S')
    return datetime_object.timestamp()


def get_timedeltas(timestamps: np.array) -> np.array:
    timedeltas = np.zeros(shape=timestamps.shape, dtype=np.float32)
    timedeltas[1:] = timestamps[1:] - timestamps[:-1]
    timedeltas[timedeltas == 0] = 1  # due to undefined behaviour of log10
    timedeltas = np.log10(timedeltas)  # decrease importance of large time differences
    return timedeltas


def get_timestamps(block_of_logs: List) -> np.array:
    regex = re.compile(r'(^\d{6} \d{6})')

    timestamps = np.zeros(shape=(len(block_of_logs),), dtype=np.float32)
    for i, log in enumerate(block_of_logs):
        str_timestamp = extract_datetime_from_line(regex, log)
        timestamps[i] = get_timestamp(str_timestamp)

    timedeltas = get_timedeltas(timestamps)
    return timedeltas


def get_embeddings_per_block_with_timestamps(data: defaultdict, model: fasttext.FastText) -> List:
    embeddings = []
    for logs in data.values():
        numpy_block = np.zeros(shape=(len(logs), model.get_dimension() + 1), dtype=np.float32)

        for i, log in enumerate(logs):
            numpy_block[i, 1:] = model.get_sentence_vector(log.rstrip())
        numpy_block[:, 0] = get_timestamps(logs)

        embeddings.append(numpy_block)
    return embeddings


def get_embeddings_per_log(data: defaultdict, model: fasttext.FastText) -> np.ndarray:
    # create embeddings per log but at first remove '\n' (newline character) from the end
    embeddings = [model.get_sentence_vector(log.rstrip()) for logs in data.values() for log in logs]
    return np.asarray(embeddings)


def get_embeddings_per_block(data: defaultdict, model: fasttext.FastText, with_timestamp: bool) -> List:
    # create embeddings per block but at first remove '\n' (newline character) from the end
    if with_timestamp:
        embeddings = get_embeddings_per_block_with_timestamps(data, model)
    else:
        embeddings = [np.asarray([model.get_sentence_vector(log.rstrip()) for log in logs]) for logs in data.values()]
    return embeddings


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
        raise AssertionError('Data keys and block ids are not in the same order!')


def create_embeddings(data_dir: str, output_dir: str, fasttext_model_path: str, per_block: bool, with_timestamp: bool):
    n_folds = get_number_of_splits(fasttext_model_path)
    model = fasttext.load_model(fasttext_model_path)

    for fold in ['val']:  # ['train', 'val']:
        for idx, (data, labels) in enumerate(load_fold_pairs(data_dir, n_folds, fold), start=1):
            # check data.keys() and labels['BlockId'] are in the same order
            check_order(data.keys(), labels['BlockId'])

            if per_block:
                print(data[list(data.keys())[0]])
                embeddings = get_embeddings_per_block(data, model, with_timestamp)
                ground_truth = get_labels_from_keys_per_block(labels)
                save_to_file(embeddings, os.path.join(output_dir, f'X-{fold}-HDFS1-cv{idx}-{n_folds}-block.pickle'))
                np.save(os.path.join(output_dir, f'y-{fold}-HDFS1-cv{idx}-{n_folds}-block.npy'), ground_truth)
            else:
                embeddings = get_embeddings_per_log(data, model)
                ground_truth = get_labels_from_keys_per_log(data, labels)
                np.save(os.path.join(output_dir, f'X-{fold}-HDFS1-cv{idx}-{n_folds}-log.npy'), embeddings)
                np.save(os.path.join(output_dir, f'y-{fold}-HDFS1-cv{idx}-{n_folds}-log.npy'), ground_truth)
