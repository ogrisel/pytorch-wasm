# emscripten-forge target: **real upstream PyTorch** imports & trains an MLP in `wasm32-emscripten`

Base: `main` · Branch: `cursor/emscripten-forge-torch-wasm-6b2e`

This PR owns the **emscripten-forge / xeus-python (WASM Jupyter)** target: the *real*
upstream PyTorch codebase (`github.com/pytorch/pytorch`) built as an emscripten-forge
package for `wasm32-emscripten`, so that `import torch` runs a basic program (fit a small
MLP) under the emscripten-forge `xeus-python` kernel in JupyterLite. **This is not the
pure-Python `microtorch` placeholder** — that earlier artifact is retained only as a
clearly-labelled throwaway and is *not* the deliverable.

## What works (executed in a real headless-Chrome browser run)

Reduced, CPU-only, single-threaded **upstream `torch` 2.8.0** built from source with the
emscripten-forge toolchain (**emscripten 3.1.73 / cross-python 3.13.1**, matching the
published `xeus-python` side-module ABI) **imports and trains a small MLP entirely
client-side in the xeus-python WASM kernel**:

```
python 3.13.1 | platform Emscripten
torch 2.8.0a0
epoch   0  loss 18.7851
epoch 199  loss 0.0354
OK: real torch trained an MLP in WASM; loss decreased 530.2x
```

`import torch`, tensor creation + `@` matmul, `nn.Sequential(Linear, ReLU, Linear)`,
`MSELoss`, autograd `loss.backward()`, and `torch.optim.SGD.step()` all execute in
WebAssembly; the loss drops ~530×.

Pipeline:

- ✅ Reduced **`torch_cpu` + `torch` + `torch_python`** compile and link to wasm32.
- ✅ **`BUILD_PYTHON` forced on** despite wasm CPython having no shared `libpython`.
- ✅ **Single `torch/_C.*.so` SIDE_MODULE** (~143 MB) statically links
  `libtorch_python` + `libtorch` + `libtorch_cpu` + deps via `--whole-archive`,
  side-stepping the cross-`.so` `GOT.func` relocation wall.
- ✅ Packaged as an **emscripten-wasm32 conda package** in a local channel, referenced
  from `environment.yml`, packed into a JupyterLite site by `jupyterlite-xeus`, executed
  in-browser.

## Evidence

- Video: `torch_wasm_mlp_training_in_jupyterlite.webm`
- Screenshot: `torch_wasm_mlp_training_output.png`
- `prototypes/emscripten-forge/torch-wasm/RESULTS.md` — full write-up + all 13 blockers.
- `prototypes/emscripten-forge/torch-wasm/logs/44-torch-run.log` — per-cell in-browser
  output with the `TORCH SUCCESS` marker.

## Key fixes beyond the initial `torch_cpu` compile

- **#8 BUILD_PYTHON on** (`WASM_PYTHON_INCLUDE_DIR` + `Python::Module` stub); bypass the
  `.pyi` stub codegen that needs host `_opcode`.
- **#10 single SIDE_MODULE** whole-archive link (keeps ATen op-registration static
  initializers; resolves cross-module C++ symbols internally).
- **#11 conda `paths.json`** needs `sha256` + `size_in_bytes` or libmamba won't extract.
- **#12 kernel boot**: no-op `pyodide-http` override (channel `pyjs-rt` `to_js` lacks the
  `dict_converter` kwarg `pyodide_http._streaming` passes).
- **#13 import chain**: compile `stub.c` as C (unmangled `initModule`); compile
  `cpuinfo_emscripten_init`; Emscripten guards for `_manager_path`,
  `multiprocessing.resource_tracker`, `_load_global_deps`; ship `torchgen`; restore the
  real `torch_version.py`; add `sympy`.

## TabICL classifier runs on a small dataset in-browser ✅

[TabICL](https://github.com/soda-inria/tabicl) 2.2.0 (sklearn-compatible tabular
in-context-learning classifier on PyTorch) does **`fit` + `predict` on a small
dataset entirely in the wasm32 kernel** against this reduced `torch`:

```
torch 2.8.0 | sklearn 1.8.0 | numpy 2.4.4
train (180, 6) test (60, 6) classes [0, 1, 2]
fit done; classes_ [0 1 2]
accuracy 0.967
TABICL SUCCESS: fit+predict ran in wasm; accuracy 0.967
```

`from tabicl import TabICLClassifier`, loading the 110 MB pretrained checkpoint
via `torch.load`, the transformer in-context forward pass, and the sklearn
`fit`/`predict` path all run in WebAssembly. What it took:

- **`tabicl-wasm` conda pkg** (`make_tabicl_pkg.py`): `tabicl`+`einops`+`tqdm`
  `--no-deps`, plus pure-python `psutil` + `huggingface_hub` stubs.
- **Partial scikit-learn** (`make_sklearn_pure_pkg.py`): the *full* compiled
  emscripten-forge sklearn (~69 `.so`) aborts kernel boot
  (`XKernel is already registered`); TabICL's import closure needs only 38
  extensions, so we keep exactly that closure (43 `.so`:
  utils/__check_build/_cyutility/_loss/decomposition/linear_model/metrics/
  neighbors/preprocessing/svm) and strip the rest — this boots cleanly and
  imports TabICL.
- **numpy-bridge shim** (`tabicl_wasm_shim.py`): `USE_NUMPY=0`, so
  `torch.from_numpy`/`Tensor.numpy` are monkeypatched via `.tolist()`.
- **Checkpoint bundled** at build time (`bundle_checkpoint.py`), git-ignored
  (>100 MB), loaded with `model_path=<local>` + `allow_auto_download=False`.

Evidence: `jupyterlite/content/tabicl_demo.ipynb`,
`logs/50-tabicl-run.log`, screenshot `logs/pw-08-tabicl.png`; harness
`jupyterlite/test/run_retry.js`. Full write-up: `RESULTS.md` → "TabICL".

## Upstream test suite vs. the wasm build

Added `tests/`: upstream PyTorch 2.8.0 core test files (`test_torch/autograd/nn/
optim/type_promotion/ops`) driven against the reduced build, with a categorized,
machine-readable **wasm-inapplicable skip manifest** (`skip_manifest.json` +
`conftest.py`, capability-gated so the same policy is correct under host torch
and the wasm kernel), a host runner (`run_tests.py`), and an in-wasm harness
(`run_pytest_wasm.js` + `wasm_pytest_driver.py`).

Honest status:

- **True in-wasm run: not performed here.** It needs the built `torch/_C.so`
  wasm module, absent on a fresh VM; the from-scratch rebuild does not fit the
  time budget (host `protoc` fails to build on the bare VM → ONNX/Caffe2 fall
  back to the un-runnable wasm `protoc.js`, exit 126; and `libtorch_cpu` ~368 MB
  on 4 CPUs is hours). Evidence: `tests/logs/... ` and `logs/50-build-repro-attempt.log`.
- **Skip classification:** 766 tests across the 6 files are inapplicable-by-
  environment (GPU 423, numpy-bridge 256, slow/large-mem 51, threads 19,
  disabled-backends 11, cpp-extension 2, distributed 2, mp 1, profiler 1) —
  `tests/logs/skip_classification.json`.
- **Reference-torch execution (proxy) validates the harness:** host baseline runs
  the upstream tests (e.g. `test_torch` 983 pass / 61 skip / 0 fail; `test_optim`
  820/146/0); the simulated-wasm applicable subset is **3711 passed / 920 skipped
  / 0 failed** across type_promotion+optim+torch+nn.
- **Failure ledger kept separate** from skips: the only executed failures were 63
  `test_autograd` logging tests, shown to be a single-process isolation artifact
  (pass in isolation), documented in `tests/logs/known_failures.md`.

See `tests/RUN_IN_WASM.md` and `RESULTS.md` → "Running the upstream test suite".

## Reproduce

See `RESULTS.md` → "How to reproduce": `build_iter.sh` (build + link `_C.so`) →
`assemble_payload.py` (payload guards) → `make_conda_pkg.py` + `make_pyodide_http_stub.py`
(package + channel) → `jupyter lite build` with `jupyterlite/environment.yml` → run
`jupyterlite/content/torch_mlp_demo.ipynb` (verify with `jupyterlite/test/run_torch.js`).
