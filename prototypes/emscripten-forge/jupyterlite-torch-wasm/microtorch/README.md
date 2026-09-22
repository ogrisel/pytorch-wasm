# microtorch

A minimal, **pure-Python, numpy-backed reverse-mode autograd engine** exposing a
small **PyTorch-compatible API** (`microtorch.tensor`, `.backward()`,
`microtorch.nn.Linear/ReLU/Sequential/MSELoss`, `microtorch.optim.SGD`,
`microtorch.no_grad`).

**This is NOT upstream PyTorch.** It exists so a *basic PyTorch-style program*
(defining and **training** a small MLP) can run inside a WebAssembly `xeus-python`
kernel in the browser (JupyterLite), where the real compiled `torch` is not
available. It implements only the handful of ops needed for an MLP training loop and
is validated numerically against real PyTorch on the host (see the repo's
`RESULTS.md`). For *real* PyTorch operators compiled to WASM (inference only), see the
ExecuTorch WASM path in `../../executorch-wasm/`.

```python
import microtorch as torch
from microtorch import nn, optim

model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
opt = optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

pred = model(X)
loss = loss_fn(pred, y)
loss.backward()
opt.step()
```

Weight convention: `Linear` stores `weight` as `(in_features, out_features)` and
computes `x @ W + b` (differs from torch's `(out, in)` + `x @ W.T`; simplified).
