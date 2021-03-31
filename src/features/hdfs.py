import numpy as np
import fasttext
import os
import re
import pandas as pd
import pickle
from datetime import datetime
from typing import Generator, Iterable, List, Dict
from collections import defaultdict

from src.data.hdfs import load_data, load_labels


def save_to_file(data: List, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_from_file(file_path: str) -> Dict:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_fold_pairs(data_dir: str, n_folds: int, fold: str) -> Generator:
    for i in range(1, n_folds + 1):
        data_filename = f'{fold}-data-HDFS1-cv{i}-{n_folds}.log'
        labels_filename = f'{fold}-labels-HDFS1-cv{i}-{n_folds}.csv'
        yield load_data(os.path.join(data_dir, data_filename)), load_labels(os.path.join(data_dir, labels_filename))


def get_number_of_splits(filename: str) -> int:
    return int(os.path.splitext(filename)[0][-1])


def search(regex: re.Pattern, line: str) -> str:
    res = regex.search(line)
    if not res:  # temporal check!!
        raise AssertionError('Nothing find!!!!!')
    return res.group(1)


def get_datetime(timestamp: str) -> datetime:
    datetime_object = datetime.strptime(timestamp, '%y%m%d %H%M%S')
    return datetime_object


def to_seconds(timedelta: np.array) -> np.array:
    return np.vectorize(lambda x: int(x.total_seconds()))(timedelta)


def calculate_timedeltas_from_timestamps(timestamps: np.array) -> np.array:
    # timedeltas = np.zeros(shape=timestamps.shape, dtype=np.int32)
    # timedeltas[1:] = to_seconds(timestamps[1:] - timestamps[:-1])
    # timedeltas[timedeltas == 0] = 1  # due to undefined behaviour of log10
    # # timedeltas += 1 # we don't lose the information about difference 1
    # timedeltas = np.log10(timedeltas)  # decrease importance of large time differences
    # return timedeltas
    timedeltas = np.ones(shape=timestamps.shape, dtype=np.int32)  # init as 1 since log10(0) is undefined
    timedeltas[1:] += to_seconds(timestamps[1:] - timestamps[:-1])  # we don't lose the information if the delta is 1
    timedeltas = np.log10(timedeltas)  # decrease importance of large time differences
    return timedeltas


def get_timedeltas(block_of_logs: List) -> np.array:
    datetime_from_line = re.compile(r'(^\d{6} \d{6})')

    timestamps = np.empty(shape=(len(block_of_logs),), dtype=np.object)
    for i, log in enumerate(block_of_logs):
        str_timestamp = search(datetime_from_line, log)
        timestamps[i] = get_datetime(str_timestamp)

    timedeltas = calculate_timedeltas_from_timestamps(timestamps)
    return timedeltas


def get_embeddings_with_timedeltas_per_block(data: defaultdict, model: fasttext.FastText) -> List:
    embeddings = []
    for logs in data.values():
        numpy_block = np.zeros(shape=(len(logs), model.get_dimension() + 1), dtype=np.float32)

        for i, log in enumerate(logs):
            numpy_block[i, 1:] = model.get_sentence_vector(log.rstrip())
        numpy_block[:, 0] = get_timedeltas(logs)

        embeddings.append(numpy_block)
    return embeddings


def get_embeddings_per_log(data: defaultdict, model: fasttext.FastText) -> np.ndarray:
    # create embeddings per log but at first remove '\n' (newline character) from the end
    embeddings = [model.get_sentence_vector(log.rstrip()) for logs in data.values() for log in logs]
    return np.asarray(embeddings)


def get_embeddings_per_block(data: defaultdict, model: fasttext.FastText, with_timedelta: bool) -> List:
    # create embeddings per block but at first remove '\n' (newline character) from the end
    if with_timedelta:
        embeddings = get_embeddings_with_timedeltas_per_block(data, model)
    else:
        embeddings = [np.asarray([model.get_sentence_vector(log.rstrip()) for log in logs]) for logs in data.values()]
    return embeddings


def get_contextual_embeddings_per_block(data: defaultdict, embedding_mapping: Dict) -> List:
    # create embeddings per block but at first remove '\n' (newline character) from the end, a timestamp and a PID from
    # the beginning
    log_without_timestamp_and_pid = re.compile(r'^\d{6} \d{6} \d+ (.*)')

    embeddings = [np.asarray([embedding_mapping[search(log_without_timestamp_and_pid, log.rstrip())] for log in logs])
                  for logs in data.values()]
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


def create_embeddings(data_dir: str, output_dir: str, fasttext_model_path: str, per_block: bool, with_timedelta: bool):
    n_folds = get_number_of_splits(fasttext_model_path)
    model = fasttext.load_model(fasttext_model_path)

    for fold in ['train', 'val']:
        for idx, (data, labels) in enumerate(load_fold_pairs(data_dir, n_folds, fold), start=1):
            # check data.keys() and labels['BlockId'] are in the same order
            check_order(data.keys(), labels['BlockId'])

            if per_block:
                embeddings = get_embeddings_per_block(data, model, with_timedelta)
                ground_truth = get_labels_from_keys_per_block(labels)
                save_to_file(embeddings, os.path.join(output_dir, f'X-{fold}-HDFS1-cv{idx}-{n_folds}-block.pickle'))
                np.save(os.path.join(output_dir, f'y-{fold}-HDFS1-cv{idx}-{n_folds}-block.npy'), ground_truth)
            else:
                embeddings = get_embeddings_per_log(data, model)
                ground_truth = get_labels_from_keys_per_log(data, labels)
                np.save(os.path.join(output_dir, f'X-{fold}-HDFS1-cv{idx}-{n_folds}-log.npy'), embeddings)
                np.save(os.path.join(output_dir, f'y-{fold}-HDFS1-cv{idx}-{n_folds}-log.npy'), ground_truth)


def create_contextual_embeddings(data_dir: str, output_dir: str, model_path: str):
    print('Currently works only for train, val split and --per-block argument!')
    n_folds = 1  # hardcoded (currently only train-val split)
    model = load_from_file(model_path)

    for fold in ['train', 'val']:
        for idx, (data, labels) in enumerate(load_fold_pairs(data_dir, n_folds, fold), start=1):
            # check data.keys() and labels['BlockId'] are in the same order
            check_order(data.keys(), labels['BlockId'])

            embeddings = get_contextual_embeddings_per_block(data, model)
            ground_truth = get_labels_from_keys_per_block(labels)
            save_to_file(embeddings, os.path.join(output_dir, f'X-{fold}-HDFS1-cv{idx}-{n_folds}-context-block.pickle'))
            np.save(os.path.join(output_dir, f'y-{fold}-HDFS1-cv{idx}-{n_folds}-context-block.npy'), ground_truth)
