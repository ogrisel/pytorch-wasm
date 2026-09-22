"""microtorch: a minimal, pure-Python, numpy-backed autograd engine with a
PyTorch-compatible API subset, runnable inside a WebAssembly (xeus-python) kernel.

This is NOT upstream PyTorch. It reimplements just enough of the ``torch`` /
``torch.nn`` / ``torch.optim`` surface to run a basic MLP training loop in the
browser. See RESULTS.md / the demo notebook for scope and limitations.
"""
from __future__ import annotations

import numpy as np

from . import nn, optim
from .tensor import Tensor, _NoGrad, is_grad_enabled

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "tensor",
    "randn",
    "zeros",
    "ones",
    "manual_seed",
    "no_grad",
    "nn",
    "optim",
    "is_grad_enabled",
    "__version__",
]


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def manual_seed(seed: int) -> None:
    np.random.seed(seed)


def no_grad():
    return _NoGrad()
