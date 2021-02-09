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
        assert kernel_size % 2 == 1 and kernel_size > 1
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

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.batches[idx]


class TrimmedDataset(Dataset):
    def __init__(self, data: np.ndarray, to: str = 'cpu', trimmed_size: int = 25):
        self.data = data
        self.device = to
        self.trimmed_size = trimmed_size

        self.tensor = self._prepare_data()

    def _prepare_data(self) -> torch.Tensor:
        dims = len(self.data), self.data[0].shape[1], self.trimmed_size
        tensors = torch.zeros(*dims, dtype=torch.float32, device=self.device)

        for i in range(len(self.data)):
            block = self.data[i]
            used_size = self.trimmed_size if len(block) > self.trimmed_size else len(block)
            tensors[i, :, :used_size] = torch.from_numpy(block[:used_size, :].T)
        return tensors

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx) -> List:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return [self.tensor[idx]]


class VanillaTCN(sklearn.base.OutlierMixin):
    def __init__(self, epochs: int = 1, batch_size: int = 32, optimizer: str = 'adam',
                 loss: str = 'mean_squared_error', learning_rate: float = 0.001, dataset_type: str = 'cropped',
                 verbose: int = True):
        # add dictionary with architecture of the model i.e., number of layers, hidden units per layer etc.
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.dataset_type = dataset_type
        self.verbose = verbose

        # internal representation of a torch model
        self._model = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X: np.ndarray) -> VanillaTCN:
        # 1. convert to torch Tensor
        train_dl = self._numpy_to_tensors(X, batch_size=self.batch_size, shuffle=True)

        # 2. initialize model
        if not self._model:
            self._initialize_model(X[0].shape[-1], [X[0].shape[-1]], 3, 0.2)

        loss_function = self._get_loss_function()
        opt = self._get_optimizer()

        for epoch in range(self.epochs):
            self._model.train()
            loss, execution_time = self._train_epoch(train_dl, opt, loss_function)

            if self.verbose:
                digits = int(np.log10(self.epochs)) + 1
                print(f'Epoch: {epoch + 1:{digits}}/{self.epochs}, loss: {loss:.5f}, time: {execution_time:.5f} s')
        return self

    def predict(self, X: np.ndarray) -> np.array:
        test_dl = self._numpy_to_tensors(X, batch_size=1, shuffle=False)

        loss_function = self._get_loss_function()

        self._model.eval()
        with torch.no_grad():
            return np.asarray([loss_function(self._model(e), e).item() for (e,) in test_dl])

    def set_params(self, **kwargs):
        self._initialize_model(kwargs['input_shape'], kwargs['layers'], kwargs['kernel_size'], kwargs['dropout'])
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']

    def _initialize_model(self, input_shape: int, layers_out: List, kernel_size: int, dropout: float):
        self._model = VanillaTCNPyTorch(input_shape, layers_out, kernel_size, dropout)
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
        return [tensor[indexes]]  # must stay persistent with PyTorch API

    def _numpy_to_tensors(self, X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        if self.dataset_type == 'variable_sized':
            train_ds = EmbeddingDataset(X, to=self._device, batch_size=batch_size)
            collate_fn = self.custom_collate if shuffle else None
            train_dl = DataLoader(train_ds, batch_size=1, shuffle=shuffle, collate_fn=collate_fn)
        elif self.dataset_type == 'cropped':
            train_ds = TrimmedDataset(X, to=self._device)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        else:
            raise NotImplementedError('This dataset preprocessing is not implemented yet.')
        return train_dl

    @time_decorator
    def _train_epoch(self, train_dl: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        loss = 0
        n_seen_examples = 0
        train_dl = tqdm(train_dl, file=sys.stdout, ascii=True, unit='batch')
        for (batch,) in train_dl:
            optimizer.zero_grad()

            pred = self._model(batch)
            batch_loss = criterion(pred, batch)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item() * batch.size(0)
            n_seen_examples += batch.size(0)

            train_dl.set_postfix({'loss': loss / n_seen_examples, 'curr_loss': batch_loss.item()})
        return loss / n_seen_examples
