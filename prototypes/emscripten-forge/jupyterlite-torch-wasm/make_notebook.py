"""Generate the JupyterLite demo notebook (mlp_training_demo.ipynb).

The code cells are also exercised on the host (see logs/05) to guarantee they run
before being shipped into the WASM kernel.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Training a PyTorch-style MLP **in the browser** (WebAssembly)\n"
    "\n"
    "This notebook runs entirely in a **WebAssembly `xeus-python` kernel** (from the\n"
    "[emscripten-forge](https://emscripten-forge.org) channel) inside JupyterLite — no\n"
    "server, no local Python. It fits a small multi-layer perceptron (MLP) on a\n"
    "synthetic dataset and shows the training loss decreasing over epochs.\n"
    "\n"
    "> **What is `microtorch`?** Upstream PyTorch does not build for `wasm32-emscripten`,\n"
    "> so this demo uses **`microtorch`**: a tiny, pure-Python, numpy-backed reverse-mode\n"
    "> autograd engine that reimplements a *small PyTorch-compatible API subset*\n"
    "> (`tensor`, `.backward()`, `nn.Linear/ReLU/Sequential/MSELoss`, `optim.SGD`).\n"
    "> **It is not upstream PyTorch.** Its gradients are validated numerically against\n"
    "> real PyTorch on the host (see the repo's `RESULTS.md`). For *real* PyTorch\n"
    "> operators compiled to WASM (inference only), see the ExecuTorch WASM path.\n"
))

cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import microtorch as torch          # reduced, PyTorch-compatible API\n"
    "from microtorch import nn, optim\n"
    "print('microtorch', torch.__version__, '| numpy', np.__version__)\n"
    "import platform; print('running on', platform.machine(), 'python', platform.python_version())"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 1. Synthetic dataset\n"
    "A nonlinear target: `y = relu(X @ w_true) + small noise`, so a linear model cannot\n"
    "fit it well but a 1-hidden-layer MLP can."
))

cells.append(nbf.v4.new_code_cell(
    "torch.manual_seed(0)\n"
    "rng = np.random.RandomState(1)\n"
    "N, D = 256, 4\n"
    "X_np = rng.randn(N, D).astype(np.float32)\n"
    "w_true = np.array([[1.5], [-2.0], [0.5], [3.0]], dtype=np.float32)\n"
    "y_np = np.maximum(X_np @ w_true, 0.0) + 0.1 * rng.randn(N, 1).astype(np.float32)\n"
    "X, y = torch.tensor(X_np), torch.tensor(y_np)\n"
    "print('X', X.shape, '| y', y.shape)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. Define the MLP, loss and optimizer\n"
    "Idiomatic PyTorch-style code — `nn.Sequential`, `nn.Linear`, `nn.ReLU`,\n"
    "`nn.MSELoss`, `optim.SGD`."
))

cells.append(nbf.v4.new_code_cell(
    "model = nn.Sequential(\n"
    "    nn.Linear(D, 32),\n"
    "    nn.ReLU(),\n"
    "    nn.Linear(32, 1),\n"
    ")\n"
    "loss_fn = nn.MSELoss()\n"
    "opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)\n"
    "n_params = sum(int(np.prod(p.shape)) for p in model.parameters())\n"
    "print('trainable tensors:', len(model.parameters()), '| scalar params:', n_params)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. Training loop\n"
    "Full-batch gradient descent for 200 epochs; the loss should fall by ~2-3 orders\n"
    "of magnitude."
))

cells.append(nbf.v4.new_code_cell(
    "losses = []\n"
    "for epoch in range(200):\n"
    "    opt.zero_grad()\n"
    "    pred = model(X)\n"
    "    loss = loss_fn(pred, y)\n"
    "    loss.backward()\n"
    "    opt.step()\n"
    "    losses.append(loss.item())\n"
    "    if epoch % 25 == 0 or epoch == 199:\n"
    "        print(f'epoch {epoch:3d}  loss = {loss.item():.5f}')\n"
    "print(f'\\nfinal loss {losses[-1]:.5f}  (started at {losses[0]:.5f}, '\n"
    "      f'{losses[0]/losses[-1]:.0f}x reduction)')\n"
    "assert losses[-1] < 0.1 * losses[0], 'loss did not decrease enough'"
))

cells.append(nbf.v4.new_markdown_cell("## 4. Plot the loss curve"))

cells.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "plt.figure(figsize=(6, 4))\n"
    "plt.semilogy(losses)\n"
    "plt.xlabel('epoch'); plt.ylabel('MSE loss (log scale)')\n"
    "plt.title('MLP training loss in WebAssembly (microtorch)')\n"
    "plt.grid(True, which='both', alpha=0.3)\n"
    "plt.tight_layout(); plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 5. Sanity check: predictions track targets\n"
    "Correlation between predictions and targets should be high (~0.99)."
))

cells.append(nbf.v4.new_code_cell(
    "with torch.no_grad():\n"
    "    p = model(X).numpy().ravel()\n"
    "t = y.numpy().ravel()\n"
    "corr = float(np.corrcoef(p, t)[0, 1])\n"
    "print(f'pred/target correlation: {corr:.4f}')\n"
    "assert corr > 0.95"
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "xpython", "display_name": "Python (XPython)", "language": "python"},
    "language_info": {"name": "python"},
}

with open("content/mlp_training_demo.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote content/mlp_training_demo.ipynb with", len(cells), "cells")
