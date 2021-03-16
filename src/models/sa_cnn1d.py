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
from src.models.datasets import EmbeddingDataset, CroppedDataset1D

SEED = 160121

np.random.seed(SEED)
torch.manual_seed(SEED)


class SACNN1DPyTorch(nn.Module):
    def __init__(self, input_dim: int, window: int, layer_configurations: List, encoder_kernel_size: int,
                 decoder_kernel_size: int, n_encoder_heads: int, n_decoder_heads: int, dropout: float, maxpool: int = 2,
                 upsampling_factor: int = 2):
        super().__init__()

        n_encoder_layers = get_encoder_size(layer_configurations)

        encoder_layers = [nn.Conv1d(input_dim, layer_configurations[0], kernel_size=encoder_kernel_size), nn.ReLU(),
                          nn.MaxPool1d(maxpool)]
        for i in range(1, min(n_encoder_layers, 2)):  # two CNN layers followed by self-attention
            encoder_layers.append(
                nn.Conv1d(layer_configurations[i - 1], layer_configurations[i], kernel_size=encoder_kernel_size))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.MaxPool1d(maxpool))

        # remove ReLU before self-attention, for more details see: M. Li, W. Hsu, X. Xie, J. Cong and W. Gao,
        # "SACNN: Self-Attention Convolutional Neural Network for Low-Dose CT Denoising With Self-Supervised Perceptual
        # Loss Network," in IEEE Transactions on Medical Imaging, vol. 39, no. 7, pp. 2289-2301, July 2020,
        # doi: 10.1109/TMI.2020.2968472.
        del encoder_layers[-2]
        self.encoder = nn.Sequential(*encoder_layers)

        attention_dim = layer_configurations[min(n_encoder_layers, 2) - 1]
        self.encoder_self_attention = nn.MultiheadAttention(attention_dim, n_encoder_heads, dropout=dropout)
        self.norm_layer1 = nn.LayerNorm(attention_dim)

        self.cnn = None
        if n_encoder_layers > 2:
            self.cnn = nn.Sequential(nn.Conv1d(attention_dim, layer_configurations[2], kernel_size=encoder_kernel_size),
                                     nn.ReLU(), nn.MaxPool1d(maxpool))

        decoder_layers = []
        for i in range(n_encoder_layers, len(layer_configurations)):
            decoder_layers.append(nn.ConvTranspose1d(layer_configurations[i - 1], layer_configurations[i],
                                                     kernel_size=decoder_kernel_size))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Upsample(scale_factor=upsampling_factor))

        self.decoder_self_attention = None
        if n_encoder_layers < len(layer_configurations):
            del decoder_layers[-2]  # see explanation above
            self.decoder = nn.Sequential(*decoder_layers)

            self.decoder_self_attention = nn.MultiheadAttention(layer_configurations[-1], n_decoder_heads,
                                                                dropout=dropout)
            self.norm_layer2 = nn.LayerNorm(layer_configurations[-1])

        self.final_layers = nn.Sequential(
            nn.ConvTranspose1d(layer_configurations[-1], input_dim, kernel_size=decoder_kernel_size),
            nn.Upsample(size=window))  # it works also reversely if Tensor is greater than the window!

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)

        x = x.permute(2, 0, 1)
        att_out, _ = self.encoder_self_attention(x, x, x)
        x = self.norm_layer1(x + att_out)
        x = x.permute(1, 2, 0)

        x = self.cnn(x) if self.cnn else x

        if self.decoder_self_attention:
            x = self.decoder(x)

            x = x.permute(2, 0, 1)
            att_out, _ = self.decoder_self_attention(x, x, x)
            x = self.norm_layer2(x + att_out)
            x = x.permute(1, 2, 0)

        x = self.final_layers(x)
        return x


class SACNN1D(sklearn.base.OutlierMixin):
    def __init__(self, epochs: int = 1, batch_size: int = 32, optimizer: str = 'adam',
                 loss: str = 'mean_squared_error', learning_rate: float = 0.001, dataset_type: str = 'cropped',
                 window: int = 35, verbose: int = True):
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

    def fit(self, X: np.ndarray) -> SACNN1D:
        # 1. convert to torch Tensor
        train_dl = self._numpy_to_tensors(X, batch_size=self.batch_size, shuffle=True)

        # 2. initialize model
        if not self._model:
            self._initialize_model(X[0].shape[-1], [X[0].shape[-1]], 3, 5, 1, 1, 0.2)

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
        test_dl = self._numpy_to_tensors(X, batch_size=128, shuffle=False)

        loss_function = self._get_loss_function(reduction='none')

        self._model.eval()
        with torch.no_grad():
            ret = []
            for (batch,) in test_dl:
                batch = batch.to(self._device)
                ret.extend(torch.mean(loss_function(self._model(batch), batch), (1, 2)).tolist())
            del test_dl  # free the memory
            return np.asarray(ret)

    def set_params(self, **kwargs):
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.window = kwargs['window']
        self._initialize_model(kwargs['input_shape'], kwargs['layers'], kwargs['encoder_kernel_size'],
                               kwargs['decoder_kernel_size'], kwargs['encoder_heads'], kwargs['decoder_heads'],
                               kwargs['dropout'])

    def _initialize_model(self, input_shape: int, layers_out: List, encoder_kernel_size: int, decoder_kernel_size: int,
                          n_encoder_heads: int, n_decoder_heads: int, dropout: float):
        self._model = SACNN1DPyTorch(input_shape, self.window, layers_out, encoder_kernel_size, decoder_kernel_size,
                                     n_encoder_heads, n_decoder_heads, dropout)
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
            train_ds = CroppedDataset1D(X, window=self.window)
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
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
            optimizer.step()

            loss += batch_loss.item() * batch.size(0)
            n_seen_examples += batch.size(0)

            train_dl.set_postfix({'loss': loss / n_seen_examples, 'curr_loss': batch_loss.item()})
        return loss / n_seen_examples
