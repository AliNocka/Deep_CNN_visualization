import torch
import math
import sys
from sklearn import metrics
from abc import ABC, abstractmethod
import numpy as np
from sklearn.utils import random

from dnn.functions import *
from dnn.layers import *


class DeepCNNBase(ABC):
    ACTIVASIONS = {
        'equality': EqualityFunction(),
        'relu': torch.nn.ReLU(),
        'sigmoid': torch.sigmoid,
        'hyperbolic_tg': torch.tanh,
        'softmax': torch.nn.LogSoftmax(dim=1),
        'leakyReLU': torch.nn.LeakyReLU(0.1)
    }

    def __init__(self, n_layers=50, activation='equality'):
        self._n_layers = n_layers
        if activation not in self.ACTIVASIONS:
            raise RuntimeError('''Invalid activation function name. 
      Use one of: {}'''.format(','.join(self.ACTIVASIONS.keys())))
        self._activation = self.ACTIVASIONS[activation]
        self._layers = []

    @abstractmethod
    def fit(self, X, y, X_test=None, y_test=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class DeepCNNClassifier(DeepCNNBase):
    METRIC_FUNCTIONS = {
        'accuracy': metrics.accuracy_score,
        'f1': metrics.f1_score,
        'precision': metrics.precision_score,
        'recall': metrics.recall_score,
    }

    OPTIMIZERS = {
        'rmsprop': torch.optim.RMSprop,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'adamw': torch.optim.AdamW,
        'adadelta': torch.optim.Adadelta,
    }

    METHODS = {
        'euler': HiddenEulerLayer,
        'runge': HiddenRungeLayer
    }

    def __init__(self, n_layers=3, n_features=None, activation='equality', metric='accuracy',
                 max_iter=10, batch_size=-1, max_opt_iter=100,
                 optimizer='adam', lr=0.1, l2=1e-3, h=1,
                 method='euler', verbose=False, log_fds=[sys.stdout]):
        super().__init__(n_layers, activation)
        if metric not in self.METRIC_FUNCTIONS:
            raise RuntimeError('''Invalid metric function name. 
      Use one of: {}'''.format(','.join(self.METRIC_FUNCTIONS.keys())))

        if optimizer not in self.OPTIMIZERS:
            raise RuntimeError('''Invalid optimizer name. 
      Use one of: {}'''.format(','.join(self.OPTIMIZERS.keys())))

        if method not in self.METHODS:
            raise RuntimeError('''Invalid discretization method name. 
      Use one of: {}'''.format(','.join(self.METHODS.keys())))

        self._metric = self.METRIC_FUNCTIONS[metric]
        self._optimizer = self.OPTIMIZERS[optimizer]
        self._layerclass = self.METHODS[method]

        self._max_iter = max_iter
        self._verbose = verbose
        self._batch_size = batch_size
        self._max_opt_iter = max_opt_iter
        self._lr = lr
        self._l2 = l2
        self._n_features = n_features
        self._h = h
        self.log_fds = log_fds

        self.metrics = []
        self.errors = []

        self.X = None
        self.X_test = None
        self.y = None
        self.y_test = None
        # Train params
        self.temp_epoch_count = None
        self.temp_iter_count = None
        self.temp_layer_idx = None
        self.optimizer = None
        self.scheduler = None


    def fit(self, X, y, X_test=None, y_test=None):
        self.X = torch.Tensor(X).clone()
        if X_test is not None:
            self.X_test = torch.Tensor(X_test).clone()
        if y_test is not None:
            self.y_test = torch.Tensor(y_test).type(torch.LongTensor).clone()

        classes_count = np.unique(y).shape[0]
        self.y = torch.Tensor(y.reshape((y.shape[0]))).type(torch.LongTensor).clone()

        self._layers = []
        input_features_count = X.shape[1]
        features_count = self._n_features or X.shape[1]
        # Создаем слои нейронной сети
        self._layers.append(
            ConvolutionalLayer(input_features_count, features_count, self.ACTIVASIONS['equality'], h=self._h))
        for _ in range(self._n_layers):
            self._layers.append(self._layerclass(features_count, self._activation, h=self._h))
        # Выходной слой
        self._layers.append(ConvolutionalLayer(features_count, classes_count, self.ACTIVASIONS['softmax'], h=self._h))

        # Инитим начальные параметры обучения
        self.temp_epoch_count = 0
        self.temp_iter_count = 0
        self.temp_layer_idx = len(self._layers) - 1
        self._layers[self.temp_layer_idx].W.requires_grad = True
        self.optimizer = self._optimizer([self._layers[self.temp_layer_idx].W], lr=self.__get_lr(self.temp_epoch_count),
                                    weight_decay=self._l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)

    def train(self, iterations_on_layer=10):
        iterations_on_layer = min(self._max_opt_iter, iterations_on_layer)
        for _ in range(round(self._max_opt_iter / iterations_on_layer)):
            if self.temp_iter_count == self._max_opt_iter:
                self.temp_iter_count = 0
                self._layers[self.temp_layer_idx].W.requires_grad = False
                self.temp_layer_idx -= 1
                self._layers[self.temp_layer_idx].W.requires_grad = True
                self.optimizer = self._optimizer([self._layers[self.temp_layer_idx].W],
                                                 lr=self.__get_lr(self.temp_epoch_count),
                                                 weight_decay=self._l2)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.75)
            if self.temp_layer_idx < 0:
                self.temp_layer_idx = len(self._layers) - 1
                self.temp_epoch_count += 1
            if self.temp_epoch_count == self._max_iter:
                return True, -1

            self.optimizer.zero_grad(set_to_none=True)
            if self._batch_size >= self.X.shape[0] or self._batch_size == -1:
                indicies = np.arange(self.X.shape[0])
            else:
                indicies = random.sample_without_replacement(n_population=self.X.shape[0],
                                                             n_samples=self._batch_size)
            X_copy = self.X[indicies].clone().detach()
            X_copy = self.__forward(X_copy)
            # Кросс-энтропия
            loss = torch.nn.NLLLoss()
            error = loss(X_copy, self.y[indicies])
            error.backward()

            self.optimizer.step()
            self.scheduler.step()

            self.temp_iter_count += 1
        return False, self.temp_layer_idx

    def predict(self, X):
        X_copy = torch.Tensor(X)
        X_copy = self.__forward(X_copy)
        return self.__out_to_labels(X_copy)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self._metric(y, y_pred)

    def __forward(self, X, from_i=None, to_i=None):
        if from_i is None:
            from_i = 0
        if to_i is None:
            to_i = len(self._layers)
        for i in range(from_i, to_i):
            X = self._layers[i].forward(X)
        return X

    def __get_lr(self, iter):
        drop = 0.8
        epochs_drop = 8.0
        return self._lr * math.pow(drop, math.floor((1 + iter) / epochs_drop))

    def __out_to_labels(self, X_out):
        X_out = X_out.detach().numpy()
        return np.argmax(X_out, axis=1)
