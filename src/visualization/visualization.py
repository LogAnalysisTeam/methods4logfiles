from matplotlib import pyplot as plt
import numpy as np

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


if __name__ == '__main__':
    tmp = np.random.randint(0, 20, size=500)

    visualize_distribution(tmp)
