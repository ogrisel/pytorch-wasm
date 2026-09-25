> **⚠️ SUPERSEDED / THROWAWAY PLACEHOLDER — NOT THE DELIVERABLE.**
> This document describes a pure-Python `microtorch` reimplementation. Per the
> corrected task, the deliverable is the **real upstream PyTorch** build for
> `wasm32-emscripten`. See
> [`../torch-wasm/PR_DESCRIPTION.md`](../torch-wasm/PR_DESCRIPTION.md) and
> [`../torch-wasm/RESULTS.md`](../torch-wasm/RESULTS.md). `microtorch` is kept
> only as a labelled placeholder and does **not** represent the achievement.

# (superseded) emscripten-forge: PyTorch-style MLP training in the browser (JupyterLite + xeus-python)

Base: `main` · Branch: `cursor/emscripten-forge-torch-wasm-6b2e`

This PR owns the **emscripten-forge / xeus-python (WASM Jupyter)** target. It delivers a
working packaging + a JupyterLite reviewer demo in which a **PyTorch-style MLP trains
in the browser** (loss decreasing ~1148× over 200 epochs) — verified in a real headless
browser. It builds on (and is isolated from) the earlier prototype work; the shared
PR #1 is closed and not touched.

## What works

- **Packaging:** JupyterLite + `jupyterlite-xeus` builds the **emscripten-forge
  `xeus-python`** kernel (CPython 3.13 → `wasm32-emscripten`) and packs `numpy` +
  `matplotlib` (emscripten-forge) plus our pure-Python **`microtorch`** into the site.
- **`microtorch`:** a minimal, numpy-backed **reverse-mode autograd** engine with a
  **PyTorch-compatible API subset** (`tensor`, `.backward()`,
  `nn.Linear/ReLU/Sequential/MSELoss`, `optim.SGD`, `no_grad`). Its gradients are
  validated against **real PyTorch 2.14** on the host (max|Δgrad| < 1e-4).
- **Demo notebook** trains a 4→32→1 MLP on synthetic nonlinear data; loss
  9.95 → 0.00867 (1148×), pred/target corr 0.9992, with a log-scale loss curve.
- **In-browser verification:** Playwright drove Chromium against the built site, ran
  the notebook in the xeus-python WASM kernel, and captured the output + plot (see
  `media/mlp_training_in_browser_wasm.png`).

## Reviewer demo — how to test

- **Local (same as Pages):** `python3 -m http.server -d docs 8000` → open
  `http://localhost:8000/lab/index.html?path=mlp_training_demo.ipynb` → **Run ▸ Run All
  Cells**. The prebuilt site is committed in `docs/`; a bundled COI service worker makes
  `SharedArrayBuffer` work on plain static hosting (no COOP/COEP headers needed).
- **GitHub Pages:** enable **Settings ▸ Pages ▸ Deploy from branch ▸ `/docs`**, then
  visit `https://<owner>.github.io/pytorch-wasm/lab/index.html?path=mlp_training_demo.ipynb`.
  (This agent has read-only GitHub access and cannot toggle Pages itself.)
- **Rebuild:** `prototypes/emscripten-forge/jupyterlite-torch-wasm/build_site.sh`.

## What is disabled / not real torch (honest)

- Upstream **`torch` is not built for WASM** here (out of budget; matches the research
  doc + the Pyodide sibling's from-source wall). There is no importable upstream `torch`
  in the browser.
- **ExecuTorch is inference-only** (no training), so the training demo uses `microtorch`.
  The real-operator ExecuTorch **inference** path (add_mul, MobileNetV2 under Node) is the
  complement — see `prototypes/emscripten-forge/executorch-wasm/RESULTS.md`.
- `microtorch` implements only the MLP subset (dense layers, relu, MSE, SGD; float32;
  no conv/attention, GPU, threading, XNNPACK, quantization, distributed, `torch.compile`).

## Artifact sizes

Site `docs/` ≈ 126 MB; largest file `xpython.wasm` 15.4 MB; `matplotlib` pkg 6.9 MB;
`python-3.13.1` pkg 6.2 MB; `microtorch` pip pkg ≈ 4 KB. No file exceeds GitHub's 100 MB.

## Remaining blockers / next steps

Real `torch`/`executorch` Python bindings as an emscripten-forge `wasm32` recipe
(c10/protobuf/SymInt ILP32, XNNPACK WASM kernels, consistent `-pthread/-matomics`, size);
ExecuTorch training extension for real-op in-browser training; growing `microtorch`
(conv2d, cross-entropy, Adam).

See `prototypes/emscripten-forge/jupyterlite-torch-wasm/RESULTS.md` for full detail.
