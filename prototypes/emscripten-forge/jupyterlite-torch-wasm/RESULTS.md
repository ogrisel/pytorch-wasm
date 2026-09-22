# emscripten-forge / xeus-python (JupyterLite) — results

_Follow-up target: **emscripten-forge ecosystem only**. Goal: a working packaging that
lets a basic PyTorch program run in the browser, with a JupyterLite reviewer demo that
trains a small MLP (loss decreasing). Hard 3-hour budget._

## TL;DR — what works

A **PyTorch-style MLP trains end-to-end inside a WebAssembly Jupyter kernel in the
browser**, with the loss decreasing ~1148× over 200 epochs and a rendered loss curve.

- **Packaging path:** JupyterLite + `jupyterlite-xeus` builds the **emscripten-forge
  `xeus-python`** kernel (CPython 3.13 compiled to `wasm32-emscripten`) and packs
  `numpy` + `matplotlib` (from emscripten-forge) plus our pure-Python **`microtorch`**
  package into the site. No server, no local Python — the notebook runs in the tab.
- **`microtorch`:** a minimal, pure-Python, numpy-backed **reverse-mode autograd
  engine** exposing a **PyTorch-compatible API subset** (`tensor`, `.backward()`,
  `nn.Linear/ReLU/Sequential/MSELoss`, `optim.SGD`, `no_grad`). It is **NOT upstream
  PyTorch** — see "What is disabled / not real torch" below.
- **Verified in a real (headless) browser:** Playwright drove Chromium against the
  built site, ran the notebook in the xeus-python kernel, and captured the training
  output + plot (screenshots + log committed).

### Demo result (captured in-browser)

```
epoch   0  loss = 9.95450
epoch  25  loss = 0.32793
epoch  50  loss = 0.04428
epoch 100  loss = 0.01047
epoch 199  loss = 0.00867
final loss 0.00867  (started at 9.95450, 1148x reduction)
pred/target correlation: 0.9992
```

## How to run it

### Option A — local static server (works exactly like GitHub Pages)
```bash
python3 -m http.server -d docs 8000
# open http://localhost:8000/lab/index.html?path=mlp_training_demo.ipynb
# then Run ▸ Run All Cells
```
The prebuilt site is committed under `docs/`. It uses the JupyterLite/xeus **COI
service worker**, so `SharedArrayBuffer` works without any custom COOP/COEP headers
(i.e. plain static hosting like GitHub Pages is sufficient).

### Option B — GitHub Pages
The site is committed to `docs/`. A maintainer can enable **Settings ▸ Pages ▸ Deploy
from branch ▸ `/docs`**; the demo is then at
`https://<owner>.github.io/pytorch-wasm/lab/index.html?path=mlp_training_demo.ipynb`.
(This agent has read-only GitHub access and cannot toggle the Pages setting itself.)

### Option C — rebuild from scratch (one command)
```bash
python3 -m venv .venv-jlite && source .venv-jlite/bin/activate
pip install "jupyterlite-core==0.6.*" "jupyterlite-xeus==4.*" jupyter_server
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$VIRTUAL_ENV" bin/micromamba
cd prototypes/emscripten-forge/jupyterlite-torch-wasm && ./build_site.sh   # -> ../../../docs
```

## Numerical fidelity (microtorch vs real PyTorch)

`microtorch/test_vs_pytorch.py` runs on the host (where real `torch` is installed) and
asserts microtorch's forward/backward matches PyTorch:

```
[OK] training loss 9.9022 -> 0.0086
[OK] microtorch gradients match PyTorch (loss=29.509943 vs 29.509941)   # max|Δgrad| < 1e-4
```

## What is disabled / not real torch (honest scope)

- **Upstream `torch` is NOT built for WASM.** It does not compile for
  `wasm32-emscripten`, and building it was out of scope/budget here (consistent with
  `docs/research-why-no-pytorch-wasm.md` and the Pyodide sibling branch, which got a
  from-source `torch_cpu` ~89% compiled before hitting c10/protobuf/SymInt ILP32
  walls). So there is **no importable upstream `torch`** in the browser.
- **ExecuTorch is inference-only** (no autograd/training), so it cannot drive a
  "loss decreasing" training demo. The ExecuTorch WASM **inference** path (real
  PyTorch operators lowered to `.pte` and run by the ExecuTorch C++ runtime compiled
  to wasm) is the complementary "real torch ops" story — see
  `../executorch-wasm/RESULTS.md` (add_mul → `[3,3,3,3]`, MobileNetV2 → 1×1000 logits
  under Node). That path is documented + reproducible via
  `../executorch-wasm/build_executorch_wasm.sh`.
- **`microtorch` implements only a small subset** needed for a dense MLP:
  ops `add/sub/mul/matmul/pow/relu/sum/mean`, layers `Linear/ReLU/Sequential`, loss
  `MSELoss`, optimizer `SGD(+momentum)`, `no_grad`. **Not implemented:** conv/RNN/
  attention, most tensor ops, autodiff of unimplemented ops, dtypes other than
  float32, GPU, threading, XNNPACK, quantization, distributed, JIT/`torch.compile`.
  `Linear` weight is stored `(in, out)` (computes `x@W+b`), differing from torch's
  `(out, in)`.

## Artifact sizes

| Item | Size |
|------|------|
| Built site `docs/` (total) | ~126 MB |
| `docs/xeus/xeus-torch-wasm/bin/xpython.wasm` | 15.4 MB |
| `matplotlib-base` kernel package | 6.9 MB |
| `python-3.13.1` kernel package | 6.2 MB |
| `microtorch-0.1.0-pip.tar.gz` (our package) | ~4 KB |
| Largest single file | 15.4 MB (well under GitHub's 100 MB limit) |

## Remaining blockers / next steps

1. **Real `torch` in emscripten-forge.** The high-value target is a `wasm32-emscripten`
   recipe for a reduced-but-real `torch` (or `executorch` Python bindings) so
   `import torch` works in xeus-python. Blockers are the ones in the research doc:
   c10/protobuf/SymInt ILP32 portability, XNNPACK WASM kernels, consistent
   `-pthread/-matomics/-mbulk-memory` across the graph, and binary size.
2. **Training with real ops via ExecuTorch.** ExecuTorch has an experimental training
   extension; wiring that into the WASM runtime + Python bindings would give real-op
   training in the browser (replacing microtorch).
3. **Grow `microtorch`** (conv2d, softmax/cross-entropy, Adam, minibatching) if a
   pure-Python fallback remains useful for teaching/demos.

## Files

- `microtorch/` — the reduced-torch package (`src/microtorch/{tensor,nn,optim}.py`) +
  `test_vs_pytorch.py` (host cross-check).
- `environment.yml` — emscripten-forge env: `xeus-python`, `numpy`, `matplotlib`,
  `pip: ./microtorch`.
- `content/mlp_training_demo.ipynb` — the reviewer notebook (generated by
  `make_notebook.py`).
- `build_site.sh` — one-command JupyterLite build (preserves the research doc).
- `playwright_run.py` — headless-browser end-to-end test.
- `logs/` — pip install, smoke build, microtorch-vs-pytorch, notebook host run, site
  build, and the Playwright in-browser run.
- `../../../docs/` — the committed, servable JupyterLite site.
