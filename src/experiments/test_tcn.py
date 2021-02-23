from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from math import gcd

from src.models.vanilla_tcnn import VanillaTCN
from src.visualization.visualization import visualize_distribution_with_labels, visualize_distribution
from src.models.train_conv_models import CustomMinMaxScaler
from src.models.metrics import metrics_report, get_metrics

np.random.seed(1)


def fill_array(new_data, new_labels, data, labels, val, window, start):
    idx = start - 1
    for j, ex in enumerate(data[labels == val]):
        if j % window == 0:
            idx += 1

        new_data[idx, j % window, :] = ex
        new_labels[idx] = val


def create_dataset(data, labels):
    window = 17  # gcd
    new_data = np.zeros(shape=(len(data) // window, window, data.shape[1]), dtype=np.float32)
    new_labels = np.zeros(shape=(len(labels) // window,))

    fill_array(new_data, new_labels, data, labels, 0, window, 0)
    fill_array(new_data, new_labels, data, labels, 1, window, len(new_data) - sum(labels == 1) // window)

    return new_data, new_labels


if __name__ == '__main__':
    x, y = make_classification(44047, weights=[0.974], class_sep=25, random_state=1)  # imbalanced classification
    print('counts', np.unique(y, return_counts=True))
    x, y = create_dataset(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

    # x = np.zeros(shape=(50000, 20, 100), dtype=np.float32)
    # y = np.zeros(shape=(50000,))
    # org = np.random.randn(20, 100)
    # for i in range(49000):
    #     noise = np.random.normal(0, 1, 2000).reshape((20, 100))
    #     x[i, ...] = org + noise
    #
    # org = np.random.randn(20, 100)
    # for i in range(49000, 50000):
    #     noise = np.random.exponential(1, 2000).reshape((20, 100))
    #     x[i, ...] = org + noise
    #     y[i] = 1
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)

    sc = CustomMinMaxScaler()
    x_train = sc.fit_transform(x_train).astype(np.float32)
    x_test = sc.transform(x_test).astype(np.float32)

    model = VanillaTCN(epochs=1, learning_rate=0.0001)
    model._initialize_model(20, [20], 7, 0.0)
    model.fit(x_train[y_train == 0])

    y_pred = model.predict(x_test)

    f1_max = 0
    for th in sorted(y_pred[y_test == 1]):
        tmp = np.zeros(shape=y_pred.shape)
        tmp[y_pred > th] = 1
        f1_max = max(f1_max, get_metrics(y_test, tmp)['f1_score'])
    print(f1_max)

    visualize_distribution_with_labels(y_pred, y_test)
    print(min(y_pred[y_test == 1]), max(y_pred[y_test == 1]))
