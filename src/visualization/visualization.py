from matplotlib import pyplot as plt
import numpy as np
from typing import List

from src.data.hdfs import SEED

np.random.seed(SEED)


def get_number_of_bins(data: np.array) -> int:
    q25, q75 = np.quantile(data, [0.25, 0.75])
    bin_width = 2 * (q75 - q25) * len(data) ** (-1 / 3)
    return round((data.max() - data.min()) / bin_width)


def visualize_distribution(y_pred: np.array, to_file: bool = False):
    fig, ax = plt.subplots()

    ax.hist(y_pred, density=True, bins=get_number_of_bins(y_pred), edgecolor='black', linewidth=1)

    ax.grid(linestyle='dashed')
    ax.set_axisbelow(True)

    ax.set_title('Reconstruction Error Distribution')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Reconstruction Error (MSE)')
    fig.tight_layout()

    if not to_file:
        plt.show()


def visualize_distribution_with_labels(y_pred: np.array, labels: np.array, to_file: bool = False):
    fig, ax = plt.subplots()

    n_bins = get_number_of_bins(y_pred)
    ax.hist([y_pred[labels == 0], y_pred[labels == 1]], bins=n_bins, edgecolor='black', linewidth=1,
            label=['Normal', 'Anomalous'])

    ax.grid(linestyle='dashed')
    ax.set_axisbelow(True)

    ax.set_title('Reconstruction Error Histogram')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.legend()
    fig.tight_layout()

    if not to_file:
        plt.show()
    else:
        plt.savefig('../../reports/figures/hist.pdf')


def visualize_windows(experiments: List, to_file: bool = False):
    fig, ax = plt.subplots()

    windows = [x['hyperparameters']['window'] for x in experiments]
    f1_scores = [x['metrics']['f1_score'] for x in experiments]

    bar = ax.bar(windows, f1_scores, edgecolor='black')
    bar[np.argmax(f1_scores)].set_facecolor('tab:orange')

    ax.set_xticks(windows)

    ax.grid(linestyle='dashed')
    ax.set_axisbelow(True)

    ax.set_title('F1 Scores by Window Size')
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Window Size')
    fig.tight_layout()

    for i, v in zip(windows, f1_scores):
        ax.text(i, v + 0.01, f'{v:.3f}', color='black', ha='center')

    if not to_file:
        plt.show()
    else:
        plt.savefig('../../reports/figures/windows.pdf')


def visualize_shapes(to_file: bool = False):
    train_counts = [(19, 243465), (13, 78063), (25, 24189), (22, 21518), (20, 20242), (23, 17316), (28, 10633),
                    (31, 10623), (24, 9568), (26, 5774), (14, 4511), (21, 3602), (4, 2593), (2, 2397), (27, 2204),
                    (29, 1633), (32, 1388), (30, 1200), (35, 1181), (36, 1096), (37, 813), (38, 537), (41, 296),
                    (33, 263), (42, 143), (15, 114), (39, 84), (34, 78), (16, 69), (40, 54), (43, 40), (44, 21),
                    (222, 12), (8, 9), (17, 9), (45, 9), (269, 9), (46, 8), (48, 5), (3, 4), (61, 3), (12, 2), (49, 2),
                    (53, 2), (229, 2), (273, 2), (50, 1), (51, 1), (52, 1), (54, 1), (55, 1), (56, 1), (57, 1),
                    (223, 1), (274, 1), (278, 1), (280, 1), (284, 1)]
    val_counts = [(19, 27074), (13, 8729), (25, 2695), (22, 2424), (20, 2256), (23, 1886), (28, 1209), (31, 1134),
                  (24, 1044), (26, 615), (14, 507), (21, 408), (4, 300), (2, 274), (27, 220), (29, 159), (32, 152),
                  (30, 141), (35, 141), (36, 105), (37, 77), (38, 61), (33, 40), (41, 30), (42, 16), (34, 10), (39, 10),
                  (15, 9), (16, 6), (43, 5), (44, 4), (3, 2), (40, 2), (222, 2), (17, 1), (45, 1), (49, 1), (229, 1),
                  (230, 1), (269, 1), (270, 1), (277, 1), (298, 1)]

    fig, ax = plt.subplots()

    x, y = zip(*train_counts)
    ax.bar(x, y, edgecolor='black', label='Training Data')
    x, y = zip(*val_counts)
    ax.bar(x, y, edgecolor='black', label='Validation Data')

    ax.grid(linestyle='dashed')
    ax.set_axisbelow(True)

    ax.set_title('Counts of Block Shapes by Data Split')
    ax.set_ylabel('Count')
    ax.set_xlabel('Block Shape')
    ax.legend()
    fig.tight_layout()

    if not to_file:
        plt.show()
    else:
        plt.savefig('../../reports/figures/shapes.pdf')


if __name__ == '__main__':
    import json
    with open('../../models/TCN-cropped-window-embeddings-HDFS1.json', 'r') as f:
        d = json.load(f)
    visualize_windows(d['experiments'])
    visualize_shapes()
