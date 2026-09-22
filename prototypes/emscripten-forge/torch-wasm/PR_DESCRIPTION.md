# emscripten-forge target: build **real upstream PyTorch** for `wasm32-emscripten`

Base: `main` · Branch: `cursor/emscripten-forge-torch-wasm-6b2e`

This PR owns the **emscripten-forge / xeus-python (WASM Jupyter)** target. The goal is
the *real* upstream PyTorch codebase (`github.com/pytorch/pytorch`) built as an
emscripten-forge package for `wasm32-emscripten`, so that `import torch` can eventually
run a basic program (fit a small MLP) under the emscripten-forge `xeus-python` kernel in
JupyterLite. **This is not the pure-Python `microtorch` placeholder** — that earlier
artifact is retained only as a clearly-labelled throwaway and is *not* the deliverable.

## What actually works (executed, not aspirational)

Using the emscripten-forge build flow (`rattler-build` + the emscripten-forge channel),
pinned to the **published** toolchain **emscripten 3.1.73 / cross-python 3.13.1** (so the
side-module ABI matches the published `xeus-python`):

- ✅ **The complete reduced `torch_cpu` compiles _and_ links to `wasm32-emscripten`.**
  All 1175 steps of the `torch_cpu` target succeed →
  **`libtorch_cpu.a`, a 368 MB WebAssembly static archive** (object files verified as
  `WebAssembly (wasm) binary module version 0x1`). It contains eager **ATen CPU** ops,
  **autograd** (`VariableType_*`), the **JIT** runtime, **`torch::nn`** modules, and
  **`torch::optim`** (`sgd.cpp`) — the full C++ surface an `nn.Linear + ReLU + MSELoss +
  SGD` MLP needs.
- ✅ Also cross-built to wasm: `libc10.a`, `libonnx.a`, `libprotobuf-lite.a`,
  `libcpuinfo.a`.
- ✅ A real **emscripten-forge recipe** (`recipe/recipe.yaml` + `build.sh` + a portability
  patch + a CMake fixups file + a `variant-3173.yaml` toolchain pin).
- ✅ **Six build blockers** identified and fixed on the emscripten-forge toolchain
  (toolchain-version mismatch, `Python::Module` stub, `Caffe2Targets` export, `SymInt *
  size_t` ILP32, `__assert_fail` EH spec, `ssize_t`, host `protoc`).

## Honest boundary (not reached within the ~3h budget)

- ⚠️ **`BUILD_PYTHON` is auto-disabled** because wasm CPython exposes no
  `Development.Module`, so `libtorch_python` / the `torch._C` side module — the
  *importable* surface — are not yet built (configure log: `BUILD_PYTHON : OFF`). This is
  the Pyodide sibling's **blocker #8**.
- ⚠️ Consequently there is **no working in-browser `import torch` training demo yet**;
  presenting one would be dishonest. Beyond #8 lie the large `torch._C` side-module link,
  conda packaging into a local channel referenced from `environment.yml`, and the runtime
  dynamic-load wall the Pyodide sibling documented as **blocker #11** (an unresolved
  `GOT.func` function-pointer relocation in the huge module).

## Evidence

- `prototypes/emscripten-forge/torch-wasm/RESULTS.md` — full write-up, versions, blockers,
  artifact sizes.
- `prototypes/emscripten-forge/torch-wasm/logs/` — raw logs: `10-configure.log`
  (`CONFIGURE_RC=0`), `11-build-torch_cpu.log` (`BUILD_TORCH_CPU_RC=0`),
  `12-hostprotoc.log`, `03*-rattler-build*.log`.
- Recipe + driver: `recipe/`, `build_iter.sh`, `apply_patches.py`.

## Reproduce

```bash
micromamba create -n ef -f recipes/ci_env.yml
rattler-build build --recipe recipes/recipes_emscripten/pytorch/recipe.yaml \
  --target-platform=emscripten-wasm32 \
  -c https://prefix.dev/emscripten-forge-dev -c conda-forge -c microsoft \
  -m prototypes/emscripten-forge/torch-wasm/recipe/variant-3173.yaml --keep-build
```
