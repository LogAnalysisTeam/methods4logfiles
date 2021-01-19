from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import sklearn
from typing import Callable

SEED = 160121

np.random.seed(SEED)
torch.manual_seed(SEED)


class AutoEncoderPyTorch(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=input_dim, out_features=32)
        self.encoder_output_layer = nn.Linear(in_features=32, out_features=8)
        self.decoder_hidden_layer = nn.Linear(in_features=8, out_features=32)
        self.decoder_output_layer = nn.Linear(in_features=32, out_features=input_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.encoder_hidden_layer(x))
        x = F.relu(self.encoder_output_layer(x))
        x = F.relu(self.decoder_hidden_layer(x))
        x = self.decoder_output_layer(x)
        return x


class AutoEncoder(sklearn.base.OutlierMixin):
    def __init__(self, epochs: int = 1, batch_size: int = 32, optimizer: str = 'adam',
                 loss: str = 'mean_squared_error'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        # internal representation of a torch model
        self._model = None

    def fit(self, X: np.ndarray) -> AutoEncoder:
        # 1. convert to torch Tensor
        # 2. initialize model
        for epoch in range(self.epochs):
            for i in range((n - 1) // self.batch_size + 1):
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                xb = X[start_i:end_i]
                yb = y_train[start_i:end_i]
                pred = self._model(xb)
                loss = loss_func(pred, yb)

                loss.backward()
                with torch.no_grad():
                    for p in model.parameters():
                        p -= p.grad * lr
                    model.zero_grad()
        return self

    def predict(self, X: np.ndarray) -> np.array:
        pass

    def fit_predict(self, X: np.ndarray, y: np.array = None) -> np.array:
        # Returns -1 for outliers and 1 for inliers.
        return self.fit(X).predict(X)

    def _initialize_model(self, input_shape: np.ndarray):
        pass

    def _get_loss_function(self) -> Callable:
        if self.loss == 'mean_squared_error':
            return nn.MSELoss
        else:
            raise NotImplementedError(f'"{self.loss}" is not implemented.')

    def _get_optimizer(self) -> Callable:
        if self.optimizer == 'adam':
            return torch.optim.Adam
        else:
            raise NotImplementedError(f'"{self.optimizer}" is not implemented.')
