# ExecuTorch-on-WASM prototype — results

_Run date: 2026-09-22. Sandbox: Ubuntu 24.04 (x86_64), 4 vCPU, 15 GiB RAM._

## TL;DR

**It works, end-to-end.** Real PyTorch models were exported with ExecuTorch,
cross-compiled to `wasm32-emscripten`, and executed under Node.js:

| Rung | Model | Result |
|------|-------|--------|
| Tensor-kernel PoC (no PyTorch) | hand-written `linear+relu` | ✅ matches JS reference |
| ExecuTorch WASM | `add_mul` (toy `mm`+`add`) | ✅ `tensor([2,2], [3.,3.,3.,3.])` |
| ExecuTorch WASM | `mv2` (**MobileNetV2**) | ✅ `tensor([1,1000], ...)` in ~49 s |

Both ExecuTorch runs used the **unmodified upstream** `examples/wasm` target — no
source patches to ExecuTorch or PyTorch were needed. The only work was host
environment setup (documented below). This is "PyTorch on WASM" in the inference
sense: a PyTorch `nn.Module` lowered through the ExecuTorch export/AOT pipeline and
run by the ExecuTorch C++ runtime compiled to WebAssembly.

The honest caveat is **binary size**: even the trivial `add_mul` runner is a
**92 MB** `.wasm` and the MobileNetV2 runner is **107 MB**, because the Release build
statically links the entire portable-kernel library with no dead-code elimination
and embeds the model into MEMFS. mv2 inference is also slow (~49 s for one iteration)
because this build uses the reference portable kernels, not XNNPACK. These match the
size/perf concerns already noted in `docs/research-why-no-pytorch-wasm.md`.

## Versions / commits

| Component | Version |
|-----------|---------|
| Emscripten (emsdk) | **4.0.10** (`b7dc6e5747465580df5984e723b9d1f10d8e804b`) — upstream-pinned in `.ci/scripts/setup-emscripten.sh` |
| emsdk-provided Node | 24.19.0 (also verified with system Node v22.14.0) |
| ExecuTorch | commit **`8081eb8813b073c5e1b97ca0eb0b8da894a3bf04`** (`main`, shallow clone) |
| PyTorch (host) | 2.14.0+cpu |
| CMake | 3.28.3 (host) / 3.31 (pip, used by some subbuilds) |
| Ninja | 1.13.2 (pip, `~/.local/bin`) |
| Host compiler | gcc/g++ 13.3.0 |

The tensor-kernel PoC was built with emsdk **3.1.58** (the repo default in
`scripts/install-emsdk.sh`); emsdk 4.0.10 was installed additionally for ExecuTorch.

## Exact commands run

### 0. Prereqs (host, one-time)

```bash
# Working GCC toolchain: /usr/bin/c++ resolved to clang-18, which auto-selected an
# incomplete gcc-14 install and failed to link with "cannot find -lstdc++".
sudo update-alternatives --set c++ /usr/bin/g++
sudo update-alternatives --set cc  /usr/bin/gcc

sudo apt-get update && sudo apt-get install -y python3-dev   # pybind11 headers
pip install ninja                                            # no apt candidate
export PIP_BREAK_SYSTEM_PACKAGES=1                           # PEP-668

# Emscripten
EMSDK_VERSION=4.0.10 scripts/install-emsdk.sh
source .emsdk/emsdk_env.sh
```

### 1. Tensor-kernel PoC

```bash
source .emsdk/emsdk_env.sh                       # emsdk 3.1.58 is fine here
cd prototypes/emscripten-forge/tensor-kernel-wasm
./build.sh && node run_node.mjs
# -> OK: WASM tensor kernel matches reference (PyTorch-style dense layer runs in WebAssembly)
```

### 2. ExecuTorch host install + WASM build + run

```bash
cd prototypes/emscripten-forge/executorch-wasm/.et-work
git clone --depth 1 https://github.com/pytorch/executorch.git
cd executorch                                    # commit 8081eb8...
PYTHON_EXECUTABLE=python3 ./install_executorch.sh   # torch + submodules + executorch

# upstream examples/wasm/test_build_wasm.sh does exactly this for add_mul and mv2:
cmake -DCMAKE_INSTALL_PREFIX=cmake-out -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=python3 -Bcmake-out . \
  && cmake --build cmake-out -j9 --target install --config Release   # host libs/tools

mkdir -p models_test
python3 -m examples.portable.scripts.export --model_name=add_mul --output_dir=models_test/
emcmake cmake -DWASM_MODEL_DIR="$(realpath models_test)" -Bcmake-out/examples/wasm .
cmake --build cmake-out/examples/wasm -j5 --target executor_runner
$EMSDK_NODE cmake-out/examples/wasm/executor_runner.js --model_path=add_mul.pte
```

`emcmake` expands the configure step to (from the log):

```
cmake -DWASM_MODEL_DIR=.../models_test -Bcmake-out/examples/wasm . \
  -DCMAKE_TOOLCHAIN_FILE=.../.emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
  -DCMAKE_CROSSCOMPILING_EMULATOR=.../.emsdk/node/24.19.0_64bit/bin/node
```

The whole thing is wrapped in `build_executorch_wasm.sh` in this directory (verified
working recipe). Running the upstream `bash examples/wasm/test_build_wasm.sh`
directly also succeeds and covers both `add_mul` and `mv2`.

## What built / ran (captured output)

### `add_mul` under Node (wasm)

```
I executorch:executor_runner.cpp:565] Model file add_mul.pte is loaded.
I executorch:executor_runner.cpp:575] Using method forward
I executorch:executor_runner.cpp:718] Model loaded in 7.85 ms.
I executorch:executor_runner.cpp:880] Iteration 1 of 1: 1.62 ms
I executorch:executor_runner.cpp:889] Model executed successfully 1 time(s) in 1.62 ms.
I executorch:executor_runner.cpp:893] 1 outputs:
OutputX 0: tensor(sizes=[2, 2], [3., 3., 3., 3.])
```

### `mv2` / MobileNetV2 under Node (wasm)

```
I executorch:executor_runner.cpp:565] Model file mv2.pte is loaded.
I executorch:executor_runner.cpp:649] Setting up CPU planned buffer 0, size 9936896.
I executorch:executor_runner.cpp:718] Model loaded in 29.49 ms.
I executorch:executor_runner.cpp:880] Iteration 1 of 1: 49252.00 ms
I executorch:executor_runner.cpp:889] Model executed successfully 1 time(s) in 49252.00 ms.
OutputX 0: tensor(sizes=[1, 1000], [-0.50986, 0.300638, 0.0953863, ... ])   # 1000 logits
```

### Artifact sizes (Release, portable kernels, model embedded in MEMFS)

```
add_mul: executor_runner.wasm  =  92 MB   executor_runner.js = 165 KB
mv2:     executor_runner.wasm  = 107 MB   (includes the embedded model weights)
```

## What failed along the way (and the fixes)

Every failure was **host environment**, not the WASM/ExecuTorch code path:

1. **`git submodule` / host toolchain: `/usr/bin/ld: cannot find -lstdc++`.**
   `/usr/bin/c++` → `/etc/alternatives/c++` → **clang-18**, which selected an
   incomplete gcc-14 (`/usr/lib/gcc/x86_64-linux-gnu/14`, no `libstdc++`) while the
   only `libstdc++` present was under gcc-13. cmake's compiler check aborted with
   "The C++ compiler ... is not able to compile a simple test program".
   **Fix:** `update-alternatives --set c++ /usr/bin/g++` (+ `cc` → `gcc`).

2. **`pytorch_tokenizers` wheel: `pybind11::module includes non-existent path
   "/usr/include/python3.12"`.** No Python dev headers.
   **Fix:** `apt-get install python3-dev` (after `apt-get update`; the image's
   package lists were stale and initially reported "no installation candidate").

3. **`ninja` missing, no apt candidate.** **Fix:** `pip install ninja`.

4. **`pip install ... torch torchao ...` → `error: externally-managed-environment`
   (PEP-668).** **Fix:** `export PIP_BREAK_SYSTEM_PACKAGES=1`.

After those four fixes, `install_executorch.sh` completed (executorch wheel built
from source, ~10 min) and the WASM build+run succeeded on the first attempt with the
unmodified upstream target.

## Concrete next steps

- **Shrink the binary.** 92 MB for `add_mul` is dominated by statically-linked
  portable kernels. Options: build only the ops a given model needs (ExecuTorch
  selective build / kernel registration), `-Os`/`-flto`, `-sMODULARIZE`, and
  `--closure 1`. Target: single-digit MB for small models.
- **Enable XNNPACK WASM microkernels** for real speed (mv2 took ~49 s on portable
  kernels). This needs `-DEXECUTORCH_BUILD_XNNPACK=ON` with the XNNPACK WASM kernel
  target and a `-pthread -matomics -mbulk-memory`-consistent build across the graph
  (see research doc §1 and pytorch/pytorch#177983). Lower the model with the XNNPACK
  partitioner (`examples/xnnpack`) so ops actually offload.
- **Browser run.** `python3 -m http.server --directory cmake-out/examples/wasm` then
  open `executor_runner.html` (needs the model named `model.pte`). COOP/COEP headers
  are only required once pthreads/XNNPACK are enabled.
- **Threading.** Current build is single-threaded (`EXECUTORCH_PAL_DEFAULT=posix`).
  A threaded build requires the whole dep graph compiled with atomics/bulk-memory
  and `SharedArrayBuffer` (cross-origin isolation) at deploy time.
- **Python-importable package.** The realistic Pyodide/emscripten-forge target is
  `executorch` pybindings over this same runtime (see `prototypes/pyodide/`), not
  full eager `libtorch`.

## Relationship to the research doc

This validates the doc's core thesis: full eager `libtorch` on WASM is a
multi-subsystem port, but **ExecuTorch inference on WASM is achievable today** with
upstream code. Freshly-observed data points to fold back into the discussion:
ExecuTorch's `examples/wasm` builds cleanly against emsdk 4.0.10 with no ExecuTorch
patches; the practical wall is **binary size** (tens–>100 MB) and **portable-kernel
speed**, exactly as predicted — XNNPACK offload + selective build are the levers.
