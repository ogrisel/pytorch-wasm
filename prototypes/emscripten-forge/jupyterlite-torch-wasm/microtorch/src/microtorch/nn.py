"""A tiny ``torch.nn``-style module system built on :class:`microtorch.Tensor`.

Reduced reimplementation, not upstream PyTorch. Weight convention: ``Linear`` stores
its weight with shape ``(in_features, out_features)`` and computes ``x @ W + b`` (note
this differs from torch's ``(out, in)`` + ``x @ W.T``; kept simple on purpose).
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Module:
    def parameters(self):
        params = []
        for _name, value in vars(self).items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, Module):
                        params.extend(v.parameters())
                    elif isinstance(v, Tensor) and v.requires_grad:
                        params.append(v)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - overridden
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming-ish init for ReLU nets.
        scale = np.sqrt(2.0 / in_features)
        w = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if not isinstance(target, Tensor):
            target = Tensor(target)
        diff = pred - target
        return (diff * diff).mean()
