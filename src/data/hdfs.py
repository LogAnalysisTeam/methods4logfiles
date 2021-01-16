import pandas as pd
import os
from collections import defaultdict
import re
from typing import Dict

from utils import train_test_split

DATA_DIR = '../../data/raw/HDFS1/'
OUTPUT_DIR = '../../data/interim/HDFS1'


def load_labels(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, converters={'Label': lambda x: True if x == 'Anomaly' else False})
    return df


def load_data(file_path: str) -> defaultdict:
    traces = defaultdict(list)

    regex = re.compile(r'(blk_-?\d+)')  # pattern is eg. blk_-1608999687919862906

    with open(file_path, 'r') as f:
        for line in f:
            block_id = find_block_id_in_log(regex, line)
            traces[block_id].append(line)
    return traces


def find_block_id_in_log(regex: re.Pattern, line: str) -> str:
    res = regex.search(line)
    return res.group()


def save_logs_to_file(data: Dict, file_path: str):
    with open(file_path, 'w') as f:
        for val in data.values():
            logs = '\n'.join(val)
            f.write(logs)


def process_hdfs(save_to_file: bool = True, test_size: float = 0.1) -> tuple:
    """
    The logs are sliced into traces according to block ids. Then each trace associated with a specific block id is
    assigned a ground-truth label.
    :return:
    """
    labels = load_labels(os.path.join(DATA_DIR, 'anomaly_label.csv'))
    data = load_data(os.path.join(DATA_DIR, 'HDFS.log'))

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, seed=160121, test_size=test_size)

    if save_to_file:
        save_logs_to_file(train_data, os.path.join(OUTPUT_DIR, 'train-data-HDFS1.log'))
        save_logs_to_file(test_data, os.path.join(OUTPUT_DIR, 'test-data-HDFS1.log'))
        train_labels.to_csv(os.path.join(OUTPUT_DIR, 'train-labels-HDFS1.csv'), index=False)
        test_labels.to_csv(os.path.join(OUTPUT_DIR, 'test-labels-HDFS1.csv'), index=False)
    return train_data, test_data, train_labels, test_labels


if __name__ == '__main__':
    # print(load_labels(os.path.join(DATA_DIR, 'anomaly_label.csv'))['Label'].value_counts())
    process_hdfs()
