# Real upstream PyTorch (`wasm32-emscripten`) via emscripten-forge — empirical results

_Cloud-sandbox run, 2026-09-22. Everything below was actually executed; raw
output is under [`logs/`](logs/). This targets the **emscripten-forge /
xeus-python** ecosystem (rattler-build + the emscripten-forge channel), the
sibling of the Pyodide path in [`../../../docs/research-why-no-pytorch-wasm.md`](../../../docs/research-why-no-pytorch-wasm.md)._

## TL;DR — it works

**The real, upstream `torch` (reduced, CPU-only, single-threaded, built from
source for `wasm32-emscripten` with the emscripten-forge toolchain) imports and
trains a small MLP entirely client-side in the xeus-python WASM kernel inside
JupyterLite.** Captured in a real headless-Chrome run of the demo notebook:

```
python 3.13.1 | platform Emscripten
torch 2.8.0a0
default dtype torch.float32
(torch.Size([256, 4]), torch.Size([256, 1]))
Sequential(
  (0): Linear(in_features=4, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
epoch   0  loss 18.7851
epoch  40  loss 0.1634
epoch  80  loss 0.1148
epoch 120  loss 0.0758
epoch 160  loss 0.0502
epoch 199  loss 0.0354
first loss 18.7851 -> last loss 0.0354
OK: real torch trained an MLP in WASM; loss decreased 530.2x
```

So `import torch`, tensor creation + `@` matmul, `nn.Sequential(Linear, ReLU,
Linear)`, `MSELoss`, autograd `loss.backward()`, and `torch.optim.SGD.step()`
all run in WebAssembly and the loss drops ~530×.

Pipeline that gets there:

- ✅ Reduced **`torch_cpu` + `torch` + `torch_python`** compile and link to
  wasm32 (eager ATen CPU, autograd, TorchScript/JIT, `torch::nn`,
  `torch::optim`).
- ✅ **`BUILD_PYTHON` forced on** despite wasm CPython having no shared
  `libpython` (blocker #8), so `libtorch_python` and the `torch._C` extension
  build.
- ✅ **Single `torch/_C.*.so` SIDE_MODULE** (~143 MB wasm) statically links
  `libtorch_python` + `libtorch` + `libtorch_cpu` + deps with
  `--whole-archive`, side-stepping cross-`.so` `GOT.func` relocation (the
  Pyodide sibling's blocker #10/#11).
- ✅ Packaged as an **emscripten-wasm32 conda package** in a local channel,
  referenced from `environment.yml`, packed into a JupyterLite site by
  `jupyterlite-xeus`, and **executed in-browser**.

## Evidence

- Video: `torch_wasm_mlp_training_in_jupyterlite.webm` (headless Chrome running
  the notebook to completion).
- Screenshot: `torch_wasm_mlp_training_output.png` (cell 4 training output).
- Console/exec log: [`logs/44-torch-run.log`](logs/44-torch-run.log) (per-cell
  outputs incl. the `TORCH SUCCESS` marker), captured by the Playwright harness
  [`jupyterlite/test/run_torch.js`](jupyterlite/test/run_torch.js).

## Environment / versions

| Component | Version |
| --- | --- |
| Host OS | Ubuntu 24.04 (x86_64), 4 CPU |
| emscripten-forge compiler (`emscripten_emscripten-wasm32`) | **3.1.73** |
| `cross-python_emscripten-wasm32` | **3.13.1** |
| Target wasm CPython (runtime, from emscripten-forge-dev) | 3.13.1 |
| Side-module ABI | `-fPIC -O2`, `-s WASM=1 -sWASM_BIGINT`, `-s SIDE_MODULE=1`, JS exceptions (`-fexceptions`), single-threaded (no `-pthread`) |
| PyTorch | 2.8.0 (GitHub release tarball, bundles `third_party/*`) |
| JupyterLite / jupyterlite-xeus | 0.8.3 / 5.1.0 |
| xeus-python / pyjs-rt (runtime kernel) | 0.17.8 / 3.2.0 |

Matching the *published* 3.1.73 toolchain matters: the runtime `xeus-python`
side-module ABI (`emscripten-abi 3.1.73`) must equal what `torch/_C.so` was
built against, or dynamic loading fails.

## How to reproduce

1. **Build the reduced torch wasm libs + link the single `_C.so`:**
   [`build_iter.sh`](build_iter.sh) drives `emcmake cmake` / `emmake ninja`
   (BUILD_PYTHON forced on) and links `torch/_C.so` (stub compiled as C so
   `PyInit__C -> initModule` resolves; `cpuinfo_emscripten_init` compiled in).
   Source edits are in [`apply_patches.py`](apply_patches.py).
2. **Assemble the importable payload:**
   [`assemble_payload.py`](assemble_payload.py) drops in `_C.so`, applies the
   Python-side Emscripten guards, restores the real `torch_version.py`, and adds
   the `torchgen` package (idempotent).
3. **Package + channel:** [`make_conda_pkg.py`](make_conda_pkg.py) emits
   `torch-2.8.0` for `emscripten-wasm32` with a valid `info/paths.json`;
   [`make_pyodide_http_stub.py`](make_pyodide_http_stub.py) publishes the
   kernel-boot `pyodide-http` override into the same local channel.
4. **Site + demo:** [`jupyterlite/environment.yml`](jupyterlite/environment.yml)
   (channels: local, emscripten-forge-dev, conda-forge; deps: `xeus-python`,
   `numpy`, `sympy`, `torch`) + `jupyter lite build`, then run
   [`jupyterlite/content/torch_mlp_demo.ipynb`](jupyterlite/content/torch_mlp_demo.ipynb).
   In-browser verification: [`jupyterlite/test/run_torch.js`](jupyterlite/test/run_torch.js)
   served with the COOP/COEP [`serve.py`](jupyterlite/test/serve.py).

## Reduced configuration (what is disabled)

`USE_CUDA/ROCM/XPU=0`, `USE_MKLDNN/FBGEMM/NNPACK/QNNPACK/PYTORCH_QNNPACK/
XNNPACK=0`, `USE_KINETO=0`, `USE_DISTRIBUTED/TENSORPIPE/GLOO/MPI=0`,
`USE_OPENMP=0` (single-threaded, `ATEN_THREADING=NATIVE`), `USE_NUMA=0`,
`USE_NUMPY=0`, `USE_MAGMA=0`, `USE_ITT=0`, `USE_MIMALLOC=0`, `USE_OBSERVERS=0`,
`USE_ONNX=0`, `BUILD_CAFFE2/CAFFE2_OPS=0`, `BUILD_TEST=0`, `BUILD_BINARY=0`,
`USE_LITE_PROTO=1`. **Kept:** eager ATen CPU ops, autograd, TorchScript/JIT,
`torch::nn`, `torch::optim`, and the Python bindings (`torch._C`).

## Running the upstream PyTorch test suite

Beyond the MLP demo, this prototype now exercises **upstream PyTorch 2.8.0's own
test files** against the reduced build and iteratively skips only the tests that
are *fundamentally inapplicable* to the wasm runtime. Everything lives under
[`tests/`](tests/) (runner + skip manifest + harness) and
[`tests/logs/`](tests/logs/) (results + ledger).

### Honest status of the in-wasm run

The **true in-wasm execution was NOT performed in this cloud run.** It requires
the built `torch/_C.so` wasm module, which is not present on a fresh VM, and the
from-scratch rebuild could not complete within the time budget: the reproduction
gets through toolchain env + source + full CMake configure (after supplying
`-DCMAKE_INSTALL_PREFIX`, which rattler-build sets for free), then stops early in
compilation because the version-matched **host `protoc` failed to build on the
bare VM**, so ONNX/Caffe2 fall back to the cross-compiled wasm `protoc.js` which
cannot run as a host tool (`Exec format error`, exit 126, ninja stops at
185/1514). Even past that, compiling `libtorch_cpu` (~368 MB of objects) on 4
CPUs far exceeds the budget. Full evidence:
[`logs/50-build-repro-attempt.log`](logs/50-build-repro-attempt.log). The in-wasm
runner ([`tests/run_pytest_wasm.js`](tests/run_pytest_wasm.js) +
[`tests/wasm_pytest_driver.py`](tests/wasm_pytest_driver.py)) is committed and
ready for when the module is packaged; under wasm `sys.platform=='emscripten'`
auto-activates the skips.

### What was executed (reference-torch proxy)

To validate the runner + skip manifest end-to-end and produce real numbers, the
same `conftest.py`/`skip_manifest.json` were run against a **full host CPU torch
2.8.0** (a proxy for eager-CPU correctness; it is *not* the reduced wasm build):

* **HOST baseline** (only host-absent capabilities skipped): proves the upstream
  tests actually execute through the harness —
  [`tests/logs/results.json`](tests/logs/results.json).

  | file | passed | skipped | failed |
  | --- | --- | --- | --- |
  | `test_type_promotion.py` | 423 | 0 | 0 |
  | `test_optim.py` | 820 | 146 | 0 |
  | `test_torch.py` | 983 | 61 | 0 |
  | `test_autograd.py` | 547 | 37 | 63† |

  †The 63 `test_autograd.py` failures are a **single-process harness/isolation
  artifact** (pt2 logging handlers accumulate → `assertLessEqual(handlers, 2)`
  trips); the same tests pass in isolation. Not a torch failure, not
  wasm-inapplicable — see [`tests/logs/known_failures.md`](tests/logs/known_failures.md).

* **SIMULATED-WASM applicable subset** (`TORCH_WASM_SIMULATE=1` forces the wasm
  capability values, skipping everything the reduced wasm runtime cannot run,
  and executes the remainder on the reference torch) —
  [`tests/logs/results-wasm-sim.json`](tests/logs/results-wasm-sim.json):

  | file | passed | skipped | failed |
  | --- | --- | --- | --- |
  | `test_type_promotion.py` | 302 | 121 | 0 |
  | `test_optim.py` | 820 | 146 | 0 |
  | `test_torch.py` | 944 | 100 | 0 |
  | `test_nn.py` | 1645 | 553 | 0 |
  | **total** | **3711** | **920** | **0** |

### Skip classification (wasm-inapplicable)

Across the six core files, **766 tests** are classified inapplicable-by-environment
(collect-only, so safe even for the 34,364-test `test_ops.py`). Machine-readable:
[`tests/logs/skip_classification.json`](tests/logs/skip_classification.json).

| category | count | why |
| --- | --- | --- |
| `gpu_cuda_rocm_xpu_mps` | 423 | no accelerator (USE_CUDA/ROCM/XPU=0) |
| `numpy_bridge` | 256 | USE_NUMPY=0 → `from_numpy`/`.numpy()` raise |
| `slow_large_memory` | 51 | OOM constrained wasm heap / too slow |
| `threading_threadpool` | 19 | single-threaded (USE_OPENMP=0, NATIVE) |
| `disabled_backends` | 11 | MKLDNN/FBGEMM/QNNPACK/XNNPACK/quant off |
| `cpp_extension_jit_compile` | 2 | no host compiler at runtime |
| `distributed_rpc_c10d` | 2 | USE_DISTRIBUTED=0 |
| `multiprocessing_fork_subprocess` | 1 | single process, no fork/exec |
| `profiler_kineto_itt` | 1 | USE_KINETO/ITT=0 |
| **total** | **766** | |

**Genuine reduced-build failures:** none can be measured until the module runs
under wasm; the ledger and predicted entries are kept separate in
[`tests/logs/known_failures.md`](tests/logs/known_failures.md) (never mixed into
the skip manifest). Reproduce: [`tests/RUN_IN_WASM.md`](tests/RUN_IN_WASM.md).

## Blockers found & fixed (in build/import order)

Fixes live in [`apply_patches.py`](apply_patches.py),
[`recipe/emscripten_fixups.cmake`](recipe/emscripten_fixups.cmake),
[`build_iter.sh`](build_iter.sh), and [`assemble_payload.py`](assemble_payload.py).

1. **`emscripten_emscripten-wasm32=4.0.9` not published** — the published
   compiler is **3.1.73**; pin the toolchain to it
   ([`recipe/variant-3173.yaml`](recipe/variant-3173.yaml)) to match the runtime
   `xeus-python` ABI.
2. **`Python::Module` target missing** — inject header-only `Python::Module` /
   `Python::Python` INTERFACE targets via `-DCMAKE_PROJECT_INCLUDE`
   ([`recipe/emscripten_fixups.cmake`](recipe/emscripten_fixups.cmake)).
3. **`install(EXPORT Caffe2Targets)` fails at generate time** — disabled
   (`CMakeLists.txt`, `if(NOT BUILD_LIBTORCHLESS)` → `if(FALSE)`).
4. **`SymInt * size_t` ambiguous overload (ILP32)** — extend the `__APPLE__`
   guard to `__EMSCRIPTEN__` in `c10/core/SymInt.h`.
5. **`__assert_fail` exception-spec mismatch** — skip the `NDEBUG` forward
   declaration in `c10/macros/Macros.h` on `__EMSCRIPTEN__`.
6. **`ssize_t` undeclared** in `c10/util/Enumerate.h` — `#include <sys/types.h>`
   on `__EMSCRIPTEN__`.
7. **Cross-compiled `protoc` can't run as a host tool (exit 126)** — build a
   version-matched host `protoc` (libprotoc 3.13.0) and point
   `CAFFE2_CUSTOM_PROTOC_EXECUTABLE` at it ([`logs/12`](logs/12-hostprotoc.log)).
8. **`BUILD_PYTHON` auto-disabled** — wasm CPython exposes no
   `Development.Module`, so `cmake/Dependencies.cmake` forces `BUILD_PYTHON OFF`.
   Fix: keep it on when `WASM_PYTHON_INCLUDE_DIR` is set (patch #5 in
   `apply_patches.py`) + the Python::Module stub. Configure now reports
   `BUILD_PYTHON : ON` ([`logs/10`](logs/10-configure.log)). Also disable the
   `torch_python_stubs` `.pyi` codegen dep (needs host `_opcode`).
9. **Cross-python codegen import failures** — copy `typing_extensions` into the
   codegen `PYTHONPATH`; bypass the `.pyi` stub target.
10. **Single-SIDE_MODULE strategy** — instead of one `.so` per lib (which hit
    the Pyodide sibling's cross-`.so` `GOT.func` relocation wall), statically
    link `libtorch_python.a` + `libtorch.a` + `libtorch_cpu.a` with
    `--whole-archive` into one `torch/_C.so` so op-registration static
    initializers survive and all C++ symbols resolve internally.
11. **`_C.so` conda package would not extract** — libmamba rejected the
    `info/paths.json` (`type must be number, but is null`). Fix: emit `sha256` +
    `size_in_bytes` per path ([`make_conda_pkg.py`](make_conda_pkg.py)).
12. **xeus-python kernel crashed on boot** (`XKernel is already registered`) —
    root cause (from the in-browser kernel log): `xeus_python_shell` calls
    `pyodide_http.patch_urllib()`, whose `_streaming` module calls
    `to_js(..., dict_converter=...)`, a kwarg the channel's `pyjs-rt 3.2.0` does
    not accept → `TypeError` → kernel never becomes ready. Fix: publish a no-op
    `pyodide-http` override in the local channel (selected first by strict
    channel priority) — [`make_pyodide_http_stub.py`](make_pyodide_http_stub.py).
13. **`import torch` runtime failures, fixed in order:**
    - `Dynamic linking error: cannot resolve symbol _Z10initModulev` — `stub.c`
      was compiled as C++ (`em++`), mangling its `initModule` reference while the
      definition is `extern "C"`. Fix: compile `stub.c` with `emcc` (C).
    - `RuntimeError: Unable to find torch_shm_manager` — guard `_manager_path()`
      on Emscripten (no shared-memory manager in single-process wasm).
    - `ModuleNotFoundError: torchgen` — ship the top-level `torchgen` package.
    - `ImportError: cannot import name 'TorchVersion'` — a build stub had
      replaced `torch/torch_version.py`; restore the real one.
    - `ModuleNotFoundError: _multiprocessing` — guard
      `torch/multiprocessing/__init__.py`'s `resource_tracker` import on
      Emscripten.
    - `Dynamic linking error: cannot resolve symbol cpuinfo_emscripten_init` —
      the vendored cpuinfo CMake never compiles `src/emscripten/init.c`; compile
      it (`-DCPUINFO_LOG_LEVEL=2`) and link it into `_C.so`.
    - `ModuleNotFoundError: sympy` — add `sympy` to `environment.yml` (a genuine
      torch dependency for `torch.fx`/dynamo, reached via `torch.optim.SGD`).

## Artifact sizes

| Artifact | Size |
| --- | --- |
| `torch/_C.*.so` (single wasm SIDE_MODULE) | ~143 MB |
| `libtorch_cpu.a` | 368 MB |
| `torch-2.8.0` conda package (`.tar.bz2`) | ~26 MB |
| `libonnx.a` | 12 MB |
| `libc10.a` | 2.2 MB |

## Notes / limitations

- The `torch._C` module is one large (~143 MB) wasm side module; first import in
  the browser instantiates it, which takes some seconds.
- `numpy` interop is off (`USE_NUMPY=0`); the demo stays within torch tensors.
- The demo emits an expected `UserWarning` about `float(loss)` on a
  `requires_grad=True` tensor; it is cosmetic and does not affect training.
