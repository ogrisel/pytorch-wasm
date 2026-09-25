"""Minimal optimizers mirroring ``torch.optim`` (SGD with optional momentum)."""
from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.momentum:
                self._velocity[i] = self.momentum * self._velocity[i] + p.grad
                p.data -= self.lr * self._velocity[i]
            else:
                p.data -= self.lr * p.grad
