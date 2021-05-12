import torch
from abc import ABC, abstractmethod


class LayerBase(ABC):
    def __init__(self, activation):
        self._activation = activation

    @abstractmethod
    def forward(self, X):
        pass


class HiddenEulerLayer(LayerBase):
    def __init__(self, n_neurons, activation, h=1e-3):
        super().__init__(activation)
        self.W = torch.empty(n_neurons, n_neurons)
        self._normalizator = torch.nn.BatchNorm1d(n_neurons, track_running_stats=False, affine=False)
        # TODO: Тут неплохо инициализировать в зависимости от activation
        torch.nn.init.kaiming_uniform_(self.W, mode='fan_in', nonlinearity='relu')
        self._h = h

    def forward(self, X):
        X = self._normalizator(X)
        return X + self._h * self._activation(X @ self.W)


class HiddenEulerLayerTest(HiddenEulerLayer):
    def forward(self, X):
        return X + self._h * self._activation(X @ self.W)


class HiddenRungeLayer(LayerBase):
    def __init__(self, n_neurons, activation, h=1e-3):
        super().__init__(activation)
        self.W = torch.empty(n_neurons, n_neurons)
        self._normalizator = torch.nn.BatchNorm1d(n_neurons, track_running_stats=False, affine=False)
        torch.nn.init.kaiming_uniform_(self.W, mode='fan_in', nonlinearity='relu')
        self._h = h

    def forward(self, X):
        X = self._normalizator(X)
        k1 = self._activation(X @ self.W)
        k2 = self._activation((X + self._h / 2 * k1) @ self.W)
        k3 = self._activation((X + self._h / 2 * k2) @ self.W)
        k4 = self._activation((X + self._h * k3) @ self.W)

        return X + self._h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class ConvolutionalLayer(LayerBase):
    def __init__(self, n_neurons_in, n_neurons_out, activation, h=1e-3):
        super().__init__(activation)
        self.W = torch.empty(n_neurons_in, n_neurons_out)
        torch.nn.init.kaiming_uniform_(self.W, mode='fan_in', nonlinearity='relu')
        self._h = h

    def forward(self, X):
        return self._h * self._activation(X @ self.W)
