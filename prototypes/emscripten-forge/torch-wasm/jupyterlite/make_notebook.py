#!/usr/bin/env python3
"""Generate the real-torch MLP training demo notebook for JupyterLite."""
import json, os

cells = []

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None, "source": src})

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

md([
    "# Real PyTorch (wasm32-emscripten) in the browser\n",
    "\n",
    "This notebook imports the **real upstream `torch`** (reduced, CPU-only, single-threaded,\n",
    "built for `wasm32-emscripten` via emscripten-forge) and trains a small MLP on a\n",
    "synthetic dataset. Everything runs client-side in the xeus-python WASM kernel.",
])

code([
    "import sys, platform, torch\n",
    "print('python', sys.version.split()[0], '| platform', platform.system())\n",
    "print('torch', torch.__version__)\n",
    "print('default dtype', torch.get_default_dtype())",
])

code([
    "import torch\n",
    "torch.manual_seed(0)\n",
    "# Synthetic regression dataset: y = Xw* + b* + noise\n",
    "N, D = 256, 4\n",
    "X = torch.randn(N, D)\n",
    "true_w = torch.tensor([2.0, -3.0, 1.5, 0.5])\n",
    "true_b = 0.7\n",
    "y = X @ true_w + true_b + 0.1 * torch.randn(N)\n",
    "y = y.unsqueeze(1)\n",
    "X.shape, y.shape",
])

code([
    "import torch.nn as nn\n",
    "model = nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, 1))\n",
    "loss_fn = nn.MSELoss()\n",
    "opt = torch.optim.SGD(model.parameters(), lr=0.05)\n",
    "model",
])

code([
    "losses = []\n",
    "for epoch in range(200):\n",
    "    opt.zero_grad()\n",
    "    pred = model(X)\n",
    "    loss = loss_fn(pred, y)\n",
    "    loss.backward()\n",
    "    opt.step()\n",
    "    losses.append(float(loss))\n",
    "    if epoch % 40 == 0 or epoch == 199:\n",
    "        print(f'epoch {epoch:3d}  loss {float(loss):.4f}')\n",
    "print('first loss', round(losses[0], 4), '-> last loss', round(losses[-1], 4))\n",
    "assert losses[-1] < losses[0], 'loss should decrease'\n",
    "print('OK: real torch trained an MLP in WASM; loss decreased %.1fx' % (losses[0]/losses[-1]))",
])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "xpython", "display_name": "Python (XPython)",
                        "language": "python"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "content", "torch_mlp_demo.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
