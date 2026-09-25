"""Reverse-mode autograd over numpy arrays.

This is a *reduced reimplementation* of a tiny slice of the PyTorch tensor API. It
is intentionally small and pure-Python (numpy-backed) so it can be installed into a
WebAssembly xeus-python kernel with no compiled extension. It is NOT upstream
PyTorch and only implements the handful of ops needed for a small MLP training loop.
"""
from __future__ import annotations

import numpy as np

_DTYPE = np.float32

# Global autograd switch, toggled by `microtorch.no_grad()`.
_GRAD_ENABLED = True


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


class _NoGrad:
    """Context manager mirroring ``torch.no_grad()``."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` so its shape matches ``shape`` (reverse of numpy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A minimal autograd tensor. Mirrors a small subset of ``torch.Tensor``."""

    def __init__(self, data, requires_grad: bool = False, _children=(), _op: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=_DTYPE)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        # Only track the graph when grad is enabled and something needs it.
        track = _GRAD_ENABLED and (requires_grad or any(c.requires_grad for c in _children))
        self._prev = set(_children) if track else set()
        self._tracked = track
        self._op = _op

    # -- construction helpers -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def item(self):
        return float(self.data.reshape(-1)[0])

    def numpy(self):
        return self.data.copy()

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def _wrap(self, data, children, op):
        out = Tensor(data, _children=children, _op=op)
        # A result requires grad iff any tracked child does.
        out.requires_grad = any(c.requires_grad for c in children) and _GRAD_ENABLED
        return out

    # -- elementwise ops ------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._wrap(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._wrap(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._wrap(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only scalar powers supported"
        out = self._wrap(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = self._wrap(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = self._wrap(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        out = self._wrap(self.data.mean(), (self,), "mean")

        def _backward():
            self.grad += (np.ones_like(self.data) / n) * out.grad

        out._backward = _backward
        return out

    # -- derived ops ----------------------------------------------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    # -- autograd -------------------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)

        build(self)
        for v in topo:
            v.grad = np.zeros_like(v.data)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            # Skip nodes that were not tracked for autograd (e.g. constants folded
            # into the graph as children of a tracked op); their _backward closure
            # would otherwise touch a child whose grad was never allocated.
            if v._tracked:
                v._backward()

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"microtorch.Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
