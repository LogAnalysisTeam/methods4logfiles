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


class SelfAttention2D(nn.Module):
    # implemented according to: M. Li, W. Hsu, X. Xie, J. Cong and W. Gao, "SACNN: Self-Attention Convolutional Neural
    # Network for Low-Dose CT Denoising With Self-Supervised Perceptual Loss Network," in IEEE Transactions on Medical
    # Imaging, vol. 39, no. 7, pp. 2289-2301, July 2020, doi: 10.1109/TMI.2020.2968472 and: X. Wang, R. Girshick,
    # A. Gupta and K. He, "Non-local Neural Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern
    # Recognition, Salt Lake City, UT, USA, 2018, pp. 7794-7803, doi: 10.1109/CVPR.2018.00813.
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()

        # try to add ReLU after each CNN layer!!

        self.bottleneck_dim = bottleneck_dim
        self.keys = nn.Conv2d(input_dim, bottleneck_dim, kernel_size=(1, 1))
        self.values = nn.Conv2d(input_dim, bottleneck_dim, kernel_size=(1, 1))
        self.queries = nn.Conv2d(input_dim, bottleneck_dim, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=2)
        self.upscale = nn.Conv2d(bottleneck_dim, input_dim, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        x_k = self.keys(x)
        batch_size, channels, height, width = x_k.size()
        x_k = x_k.reshape((batch_size, self.bottleneck_dim, -1))
        x_q = self.queries(x).reshape((batch_size, self.bottleneck_dim, -1)).permute(0, 2, 1)

        x_qk = torch.einsum('bij,bjk->bik', x_q, x_k)
        soft_qk = self.softmax(x_qk)

        x_v = self.values(x).reshape((batch_size, self.bottleneck_dim, -1)).permute(0, 2, 1)
        x_qkv = torch.einsum('bij,bjk->bik', soft_qk, x_v)
        x_qkv = x_qkv.permute(0, 2, 1).reshape((batch_size, self.bottleneck_dim, height, width))
        
        y = self.upscale(x_qkv)
        return x + y


class SACNN2DPyTorch(nn.Module):
    def __init__(self, input_dim: int, window: int, layer_configurations: List, encoder_kernel: tuple,
                 decoder_kernel: tuple, maxpool: int = 2, upsampling_factor: int = 2):
        super().__init__()

        encoder = [
            nn.Conv2d(1, 32, kernel_size=(5, 3)),
            nn.ReLU(),
            nn.MaxPool2d(maxpool),
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(maxpool)
        ]

        self.attn = SelfAttention2D(64, 50)

        decoder = [
            nn.ConvTranspose2d(64, 32, kernel_size=(9, 5)),
            nn.ReLU(),
            nn.Upsample(scale_factor=upsampling_factor),
            nn.ConvTranspose2d(32, 1, kernel_size=(7, 7)),
            nn.ReLU(),
            nn.Upsample((input_dim, window))
        ]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor):
        # # relu = nn.ReLU()
        #
        x = self.encoder(x)

        x = self.attn(x)
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
        x = self.decoder(x)
        return x


class SACNN2D(sklearn.base.OutlierMixin):
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

    def fit(self, X: np.ndarray) -> SACNN2D:
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
        test_dl = self._numpy_to_tensors(X, batch_size=128, shuffle=False)

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
        self._model = SACNN2DPyTorch(input_shape, self.window, layers_out, encoder_kernel, decoder_kernel)
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
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1)
            optimizer.step()

            loss += batch_loss.item() * batch.size(0)
            n_seen_examples += batch.size(0)

            train_dl.set_postfix({'loss': loss / n_seen_examples, 'curr_loss': batch_loss.item()})
        return loss / n_seen_examples
