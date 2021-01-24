from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sklearn
from tqdm import tqdm
import sys

from src.models.utils import time_decorator

SEED = 160121

np.random.seed(SEED)
torch.manual_seed(SEED)


class AutoEncoderPyTorch(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.l1 = nn.Linear(in_features=input_dim, out_features=16)
        self.l2 = nn.Linear(in_features=16, out_features=8)
        self.l3 = nn.Linear(in_features=8, out_features=16)
        self.l4 = nn.Linear(in_features=16, out_features=input_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


class AutoEncoder(sklearn.base.OutlierMixin):
    def __init__(self, epochs: int = 1, batch_size: int = 32, optimizer: str = 'adam',
                 loss: str = 'mean_squared_error', learning_rate: float = 0.001, verbose: int = True):
        # add dictionary with architecture of the model i.e., number of layers, hidden units per layer etc.
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.verbose = verbose

        # internal representation of a torch model
        self._model = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X: np.ndarray) -> AutoEncoder:
        # 1. convert to torch Tensor
        train_dl = self._numpy_to_tensors(X, batch_size=self.batch_size)

        # 2. initialize model
        if not self._model:
            self._initialize_model(X.shape)

        loss_function = self._get_loss_function()
        opt = self._get_optimizer()

        for epoch in range(self.epochs):
            self._model.train()
            loss, execution_time = self._train_epoch(train_dl, opt, loss_function)

            if self.verbose:
                digits = int(np.log10(self.epochs)) + 1
                print(f'Epoch: {epoch + 1:{digits}}/{self.epochs}, loss: {loss / len(X):.5f}, '
                      f'time: {execution_time:.5f} s')
        return self

    def predict(self, X: np.ndarray) -> np.array:
        test_dl = self._numpy_to_tensors(X, batch_size=1)

        loss_function = self._get_loss_function()

        self._model.eval()
        with torch.no_grad():
            return sum(loss_function(self._model(e), e).item() for (e,) in test_dl) / len(X)

    # def fit_predict(self, X: np.ndarray, y: np.array = None) -> np.array:
    #     # Returns -1 for outliers and 1 for inliers.
    #     # train only using normal data examples
    #     return self.fit(X[y == 0, :]).predict(X)

    def _initialize_model(self, input_shape: tuple):
        self._model = AutoEncoderPyTorch(input_dim=input_shape[1])
        self._model.to(self._device)

    def _get_loss_function(self) -> nn.Module:
        if self.loss == 'mean_squared_error':
            return nn.MSELoss(reduction='sum')  # avoid division by number of examples in mini batch
        elif self.loss == 'kullback_leibler_divergence':
            return nn.KLDivLoss(reduction='sum')  # avoid division by number of examples in mini batch
        else:
            raise NotImplementedError(f'"{self.loss}" is not implemented.')

    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer == 'adam':
            return torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f'"{self.optimizer}" is not implemented.')

    def _numpy_to_tensors(self, X: np.ndarray, batch_size: int) -> DataLoader:
        X_tensor = torch.from_numpy(X).to(self._device)
        train_ds = TensorDataset(X_tensor)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        return train_dl

    @time_decorator
    def _train_epoch(self, train_dl: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        loss = 0
        train_dl = tqdm(train_dl, file=sys.stdout, ascii=True, unit='batch')
        for idx, (batch,) in enumerate(train_dl, start=1):
            optimizer.zero_grad()

            pred = self._model(batch)
            batch_loss = criterion(pred, batch)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            train_dl.set_postfix({'loss': loss / (idx * self.batch_size)})
        return loss
