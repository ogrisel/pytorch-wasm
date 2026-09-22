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

## Reproduce

See `RESULTS.md` → "How to reproduce": `build_iter.sh` (build + link `_C.so`) →
`assemble_payload.py` (payload guards) → `make_conda_pkg.py` + `make_pyodide_http_stub.py`
(package + channel) → `jupyter lite build` with `jupyterlite/environment.yml` → run
`jupyterlite/content/torch_mlp_demo.ipynb` (verify with `jupyterlite/test/run_torch.js`).
