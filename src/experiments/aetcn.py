import numpy as np
from sklearn.metrics import confusion_matrix

from src.models.datasets import CustomMinMaxScaler
from src.models.utils import load_pickle_file, find_optimal_threshold, classify
from src.models.metrics import metrics_report
from src.models.autoencoder_tcnn import AETCN

config = {
    "hyperparameters": {
        "batch_size": 8,
        "dropout": 0.3238513794447296,
        "epochs": 4,
        "input_shape": 100,
        "kernel_size": 3,
        "layers": [
            [
                142
            ],
            1246,
            [
                100  # output_shape == input_shape
            ]
        ],
        "learning_rate": 0.0016378937069540646,
        "window": 45
    },
    # not used currently
    "model_path": "../../models/aetcn/4f5f4682-1ca5-400a-a340-6243716690c0.pt",
    "threshold": 0.00331703620031476
}

X = load_pickle_file('../../data/processed/HDFS1/X-val-HDFS1-cv1-1-block.npy')[:1000]
y = np.load('../../data/processed/HDFS1/y-val-HDFS1-cv1-1-block.npy')[:1000]

# list of matrices, a matrix == blk_id -> NumPy [[005515, ...], ...] (n_logs x 100)

# F1 = (2 * r * p) / (r + p)

n_examples = 700

sc = CustomMinMaxScaler()  # range 0 -- 1
x_train = sc.fit_transform(X[:n_examples])
y_train = y[:n_examples]
x_test = sc.transform(X[n_examples:])
y_test = y[n_examples:]

model = AETCN()
model.set_params(**config['hyperparameters'])

model.fit(x_train[y_train == 0])  # 0 -> normal, 1 -> anomaly
y_pred = model.predict(x_test)  # return reconstruction errors

theta, f1 = find_optimal_threshold(y_test, y_pred)
y_pred = classify(y_pred, theta)
metrics_report(y_test, y_pred)
confusion_matrix(y_test, y_pred)
