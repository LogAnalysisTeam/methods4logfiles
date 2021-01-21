import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.models.autoencoder import AutoEncoder

np.random.seed(160121)

if __name__ == '__main__':
    # change to X_train -> this is only debugging
    X_val = np.load('/home/martin/bdip25/data/processed/HDFS1/X-val-HDFS1-cv1-1.npy')
    y_val = np.load('/home/martin/bdip25/data/processed/HDFS1/y-val-HDFS1-cv1-1.npy')

    # do standardization, normalization here!!!!

    sc = StandardScaler()
    X = sc.fit_transform(X_val)
    X = X[y_val == 0, :][np.random.randint(0, 900000, size=100000)]  # get only normal training examples

    model = AutoEncoder(epochs=10)
    model.fit(X)
    print(model.predict(X_val[y_val == 1, :]))
