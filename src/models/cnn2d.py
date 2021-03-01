from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sklearn
from tqdm import tqdm
from typing import List
import sys

from src.models.utils import time_decorator, get_encoder_size
from src.models.datasets import EmbeddingDataset, CroppedDataset2D

SEED = 160121

np.random.seed(SEED)
torch.manual_seed(SEED)


class CNN2DPyTorch(nn.Module):
    def __init__(self, input_dim: int, window: int, layer_configurations: List, encoder_kernel: tuple,
                 decoder_kernel: tuple, maxpool: int = 2, upsampling_factor: int = 2):
        super().__init__()

        n_encoder_layers = get_encoder_size(layer_configurations)

        layers = [nn.Conv2d(1, layer_configurations[0], kernel_size=encoder_kernel), nn.ReLU(), nn.MaxPool2d(maxpool)]
        for i in range(1, n_encoder_layers):
            layers.append(nn.Conv2d(layer_configurations[i - 1], layer_configurations[i], kernel_size=encoder_kernel))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(maxpool))

        for i in range(n_encoder_layers, len(layer_configurations)):
            layers.append(
                nn.ConvTranspose2d(layer_configurations[i - 1], layer_configurations[i], kernel_size=decoder_kernel))
            layers.append(nn.ReLU())
            layers.append(nn.Upsample(scale_factor=upsampling_factor))

        layers.append(nn.ConvTranspose2d(layer_configurations[-1], 1, kernel_size=decoder_kernel))
        # it works also reversely if Tensor is greater than the window!
        layers.append(nn.Upsample(size=(input_dim, window)))

        self.net = nn.Sequential(*layers)

        # encoder = [
        #     nn.Conv2d(1, 32, kernel_size=(5, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(maxpool),
        #     nn.Conv2d(32, 64, kernel_size=(5, 5)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(maxpool)
        # ]
        #
        # decoder = [
        #     nn.ConvTranspose2d(64, 32, kernel_size=(9, 5)),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=upsampling_factor),
        #     nn.ConvTranspose2d(32, 1, kernel_size=(7, 7)),
        #     nn.ReLU(),
        #     nn.Upsample((input_dim, window))
        # ]
        #
        # self.encoder = nn.Sequential(*encoder)
        # self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor):
        # # relu = nn.ReLU()
        #
        # x = self.encoder(x)
        #
        # # this bottleneck does not have positive impact on F1 score!!!
        # # org_shape = x.size()
        # # x = torch.flatten(x, start_dim=1)
        # # flat_shape = x.size()
        # # min_dim = 2048
        # # bottleneck = nn.Linear(flat_shape[1], min_dim)
        # # x = relu(bottleneck(x))
        # # fc = nn.Linear(min_dim, flat_shape[1])
        # # x = relu(fc(x))
        # # x = torch.reshape(x, shape=org_shape)
        #
        # x = self.decoder(x)
        # return x
        x = self.net(x)
        return x


class CNN2D(sklearn.base.OutlierMixin):
    def __init__(self, epochs: int = 1, batch_size: int = 32, optimizer: str = 'adam',
                 loss: str = 'mean_squared_error', learning_rate: float = 0.001, dataset_type: str = 'cropped',
                 window: int = 25, verbose: int = True):
        # add dictionary with architecture of the model i.e., number of layers, hidden units per layer etc.
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.dataset_type = dataset_type
        self.window = window
        self.verbose = verbose

        # internal representation of a torch model
        self._model = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X: np.ndarray) -> CNN2D:
        # 1. convert to torch Tensor
        train_dl = self._numpy_to_tensors(X, batch_size=self.batch_size, shuffle=True)

        # 2. initialize model
        if not self._model:
            self._initialize_model(X[0].shape[-1], [32, 64, 32], (12, 3), (7, 5))

        loss_function = self._get_loss_function()
        opt = self._get_optimizer()

        for epoch in range(self.epochs):
            self._model.train()
            loss, execution_time = self._train_epoch(train_dl, opt, loss_function)

            if self.verbose:
                digits = int(np.log10(self.epochs)) + 1
                print(f'Epoch: {epoch + 1:{digits}}/{self.epochs}, loss: {loss:.5f}, time: {execution_time:.5f} s')
        del train_dl  # free the memory
        return self

    def predict(self, X: np.ndarray) -> np.array:
        test_dl = self._numpy_to_tensors(X, batch_size=64, shuffle=False)

        loss_function = self._get_loss_function(reduction='none')

        self._model.eval()
        with torch.no_grad():
            ret = []
            for (batch,) in test_dl:
                batch = batch.to(self._device)
                ret.extend(torch.mean(loss_function(self._model(batch), batch), (1, 2, 3)).tolist())
            del test_dl  # free the memory
            return np.asarray(ret)

    def set_params(self, **kwargs):
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.window = kwargs['window']
        self._initialize_model(kwargs['input_shape'], kwargs['layers'], kwargs['encoder_kernel_size'],
                               kwargs['decoder_kernel_size'])

    def _initialize_model(self, input_shape: int, layers_out: List, encoder_kernel: tuple, decoder_kernel: tuple):
        self._model = CNN2DPyTorch(input_shape, self.window, layers_out, encoder_kernel, decoder_kernel)
        self._model.to(self._device)

    def _get_loss_function(self, reduction: str = 'mean') -> nn.Module:
        if self.loss == 'mean_squared_error':
            return nn.MSELoss(reduction=reduction)
        elif self.loss == 'kullback_leibler_divergence':
            return nn.KLDivLoss(reduction=reduction)
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
            train_ds = EmbeddingDataset(X, batch_size=batch_size)
            collate_fn = self.custom_collate if shuffle else None
            train_dl = DataLoader(train_ds, batch_size=1, shuffle=shuffle, collate_fn=collate_fn)
        elif self.dataset_type == 'cropped':
            train_ds = CroppedDataset2D(X, window=self.window)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        else:
            raise NotImplementedError('This dataset preprocessing is not implemented yet.')
        del train_ds  # free the memory
        return train_dl

    @time_decorator
    def _train_epoch(self, train_dl: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        loss = 0
        n_seen_examples = 0
        train_dl = tqdm(train_dl, file=sys.stdout, ascii=True, unit='batch')
        for (batch,) in train_dl:
            batch = batch.to(self._device)

            optimizer.zero_grad()

            pred = self._model(batch)
            batch_loss = criterion(pred, batch)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item() * batch.size(0)
            n_seen_examples += batch.size(0)

            train_dl.set_postfix({'loss': loss / n_seen_examples, 'curr_loss': batch_loss.item()})
        return loss / n_seen_examples