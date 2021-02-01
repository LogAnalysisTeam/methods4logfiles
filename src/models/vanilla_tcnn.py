from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sklearn
from tqdm import tqdm
from typing import List
import sys

from src.models.utils import time_decorator
from src.models.tcnn import TemporalConvNet

SEED = 160121

np.random.seed(SEED)
torch.manual_seed(SEED)


class VanillaTCNPyTorch(nn.Module):
    def __init__(self, input_dim: int, layers: List, kernel_size: int, dropout: float):
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, layers, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = self.tcn(x)
        return x


class EmbeddingDataset(Dataset):
    def __init__(self, data: np.ndarray, to: str):
        self.data = data
        self.device = to

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # transpose each example in order to be consistent with TCN implementation
        tensor = torch.from_numpy(self.data[idx].T)
        tensor = torch.unsqueeze(tensor, 0)  # add one dimension, only for batch size = 1
        return tensor.to(self.device)


class VanillaTCN(sklearn.base.OutlierMixin):
    def __init__(self, epochs: int = 1, batch_size: int = 32, optimizer: str = 'adam',
                 loss: str = 'mean_squared_error', learning_rate: float = 0.001, verbose: int = True):
        # add dictionary with architecture of the model i.e., number of layers, hidden units per layer etc.
        self.epochs = epochs
        self.batch_size = 1  # batch_size # each example has different dimensions
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.verbose = verbose

        # internal representation of a torch model
        self._model = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X: np.ndarray) -> VanillaTCN:
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
            return np.asarray([loss_function(self._model(e), e).item() for (e,) in test_dl])

    def _initialize_model(self, input_shape: tuple):
        self._model = VanillaTCNPyTorch(100, [10, 20, 100], 5, 0)
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
        train_ds = EmbeddingDataset(X, to=self._device)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
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
