from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np
import sklearn
from tqdm import tqdm
from typing import List
import sys

from src.models.utils import time_decorator
from src.models.datasets import EmbeddingDataset, CroppedDataset1D

SEED = 160121

np.random.seed(SEED)
torch.manual_seed(SEED)


class TransformerAutoEncoder(sklearn.base.OutlierMixin):
    def __init__(self, epochs: int = 1, batch_size: int = 32, optimizer: str = 'adam',
                 loss: str = 'mean_squared_error', learning_rate: float = 0.001,
                 dataset_type: str = 'cropped', window: int = 35, verbose: int = True):
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

    def fit(self, X: np.ndarray) -> TransformerAutoEncoder:
        # 1. convert to torch Tensor
        train_dl = self._numpy_to_tensors(X, batch_size=self.batch_size, shuffle=True)

        # 2. initialize model
        if not self._model:
            self._initialize_model(X[0].shape[-1], 1, 1, 1, 256, 0.2)

        loss_function = self._get_loss_function()
        opt = self._get_optimizer()
        total_steps = len(train_dl) * self.epochs
        scheduler = self._get_warmup_optimizer(opt, int(0.2 * total_steps), total_steps)

        for epoch in range(self.epochs):
            self._model.train()
            loss, execution_time = self._train_epoch(train_dl, opt, loss_function, scheduler)

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
                batch = batch.permute(2, 0, 1)

                ret.extend(torch.mean(loss_function(self._model(batch, batch), batch), (0, 2)).tolist())
            del test_dl  # free the memory
            return np.asarray(ret)

    def set_params(self, **kwargs):
        self.epochs = kwargs['epochs']
        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.window = kwargs['window']
        self._initialize_model(kwargs['input_dim'], kwargs['heads'], kwargs['n_encoders'], kwargs['n_decoders'],
                               kwargs['dim_feedforward'], kwargs['dropout'])

    def _initialize_model(self, input_dim: int, n_heads: int, n_encoders: int, n_decoders: int, fc_dim: int,
                          dropout: float):
        self._model = nn.Transformer(input_dim, n_heads, n_encoders, n_decoders, fc_dim, dropout)
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
    def _get_warmup_optimizer(optimizer: torch.optim.Optimizer, n_warmup_steps: int, n_total_steps: int,
                              sch_type: str = 'cosine') -> LambdaLR:
        # inspired on website: https://huggingface.co/transformers/_modules/transformers/optimization.html
        if sch_type == 'constant':
            return LambdaLR(optimizer, lambda step: (step / max(1, n_warmup_steps)) if step < n_warmup_steps else 1)
        elif sch_type == 'cosine':
            def lr_lambda(step):
                if step < n_warmup_steps:
                    return step / max(1, n_warmup_steps)
                progress = float(step - n_warmup_steps) / float(max(1, n_total_steps - n_warmup_steps))
                return max(0, 0.5 * (1 + np.cos(np.pi * progress)))

            return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def custom_collate(data: List) -> List:
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
    def _train_epoch(self, train_dl: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                     scheduler: LambdaLR) -> float:
        loss = 0
        n_seen_examples = 0
        train_dl = tqdm(train_dl, file=sys.stdout, ascii=True, unit='batch')
        for (batch,) in train_dl:
            batch = batch.to(self._device)
            batch = batch.permute(2, 0, 1)

            optimizer.zero_grad()

            pred = self._model(batch, batch)
            batch_loss = criterion(pred, batch)

            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            # print(scheduler.state_dict())

            loss += batch_loss.item() * batch.size(0)
            n_seen_examples += batch.size(0)

            train_dl.set_postfix({'loss': loss / n_seen_examples, 'curr_loss': batch_loss.item()})
        return loss / n_seen_examples
