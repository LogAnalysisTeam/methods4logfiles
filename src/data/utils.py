import pandas as pd
from sklearn import model_selection
from collections import defaultdict
from typing import Dict, Union


def get_data_by_indices(data: Union[defaultdict, Dict], labels: pd.DataFrame) -> Dict:
    ret = {block_id: data[block_id] for block_id in labels['BlockId']}
    return ret


def train_test_split(data: Union[defaultdict, Dict], labels: pd.DataFrame, test_size: float, seed: int) -> tuple:
    # assumes that one block is one label, otherwise it would generate more data
    train_labels, test_labels = model_selection.train_test_split(labels, stratify=labels['Label'], test_size=test_size,
                                                                 random_state=seed)

    train_data = get_data_by_indices(data, train_labels)
    test_data = get_data_by_indices(data, test_labels)
    return train_data, test_data, train_labels, test_labels
