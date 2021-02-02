from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sklearn
from tqdm import tqdm
from typing import List, DefaultDict
from collections import defaultdict
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
    def __init__(self, data: np.ndarray, to: str = 'cpu', batch_size: int = 8):
        self.data = data
        self.device = to
        self.batch_size = batch_size

        self.batches = self._prepare_data()

    def _get_occurrences(self) -> DefaultDict:
        ret = defaultdict(list)
        for x in self.data:
            ret[x.shape].append(x)
        return ret

    def _prepare_data(self) -> List:
        occurrences = self._get_occurrences()

        tensors = []
        for logs in occurrences.values():
            for i in range(0, len(logs), self.batch_size):
                batch = np.asarray(logs[i:i + self.batch_size])
                tensor = torch.from_numpy(batch).permute(0, 2, 1)  # transpose each example in the batch
                tensors.append(tensor.to(self.device))
        return tensors

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.batches[idx]


class VanillaTCN(sklearn.base.OutlierMixin):
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
            return np.asarray([loss_function(self._model(e), e).item() for e in test_dl])

    def _initialize_model(self, input_shape: tuple):
        self._model = VanillaTCNPyTorch(100, [100], 5, 0.2)
        self._model.to(self._device)

    def _get_loss_function(self) -> nn.Module:
        if self.loss == 'mean_squared_error':
            return nn.MSELoss()
        elif self.loss == 'kullback_leibler_divergence':
            return nn.KLDivLoss()
        else:
            raise NotImplementedError(f'"{self.loss}" is not implemented.')

    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer == 'adam':
            return torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f'"{self.optimizer}" is not implemented.')

    @staticmethod
    def custom_collate(data: List):
        # randomly shuffle data within a batch
        tensor = data[0]
        indexes = torch.randperm(tensor.shape[0])
        return tensor[indexes]

    def _numpy_to_tensors(self, X: np.ndarray, batch_size: int) -> DataLoader:
        train_ds = EmbeddingDataset(X, to=self._device, batch_size=batch_size)
        train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=self.custom_collate)
        return train_dl

    @time_decorator
    def _train_epoch(self, train_dl: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        loss = 0
        train_dl = tqdm(train_dl, file=sys.stdout, ascii=True, unit='batch')
        for idx, batch in enumerate(train_dl, start=1):
            optimizer.zero_grad()

            pred = self._model(batch)
            batch_loss = criterion(pred, batch)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            train_dl.set_postfix({'loss': loss / idx})
        return loss
