# Real upstream PyTorch (`wasm32-emscripten`) via emscripten-forge — empirical results

_Cloud-sandbox run, 2026-09-22. Everything below was actually executed; raw
output is under [`logs/`](logs/). This targets the **emscripten-forge /
xeus-python** ecosystem (rattler-build + the emscripten-forge channel), the
sibling of the Pyodide path in [`../../../docs/research-why-no-pytorch-wasm.md`](../../../docs/research-why-no-pytorch-wasm.md)._

## TL;DR

This drives a **from-source build of upstream PyTorch 2.8.0 for
`wasm32-emscripten` using the emscripten-forge toolchain** (emscripten 3.1.73 /
cross-python 3.13.1, the versions the published emscripten-forge `xeus-python`
is built against, so the side-module ABI matches).

- ✅ **The complete reduced `torch_cpu` compiles _and_ links to wasm32.**
  All 1175 build steps of the `torch_cpu` target succeed and produce
  **`libtorch_cpu.a` — a 368 MB WebAssembly static archive** (verified: object
  files are `WebAssembly (wasm) binary module version 0x1`). This includes:
  - eager **ATen CPU** kernels,
  - **autograd** (`torch/csrc/autograd/generated/VariableType_*.cpp`),
  - the **TorchScript/JIT** runtime,
  - the **`torch::nn`** C++ modules (`linear`, activations, containers, …),
  - **`torch::optim`** including **`sgd.cpp`**.

  i.e. every C++ ingredient an `nn.Linear + ReLU + MSELoss + SGD` MLP training
  loop needs is compiled to wasm.
- ✅ Also cross-compiled to wasm static libs on the way: `libc10.a` (2.2 MB),
  `libprotobuf-lite.a`, `libcpuinfo.a`, `libonnx.a` (12 MB).
- ✅ **Six independent build blockers found and fixed** on the emscripten-forge
  toolchain (details below). Configure (`cmake` generate) is clean
  (`CONFIGURE_RC=0`) and the `torch_cpu` build is clean (`BUILD_TORCH_CPU_RC=0`).
- ⚠️ **Remaining blocker toward an _importable_ `torch`: `BUILD_PYTHON` is
  auto-disabled**, so `libtorch_python` / the `torch._C` side module are not yet
  built. Because wasm CPython is statically linked (no shared `libpython`),
  `find_package(Python COMPONENTS Development.Module)` reports
  `Development.Module` *missing*, and `cmake/Dependencies.cmake` then forces
  `BUILD_PYTHON OFF` (configure log:
  `Found Python … missing components: Development.Module … BUILD_PYTHON : OFF`).
  This is exactly the Pyodide sibling's **blocker #8**. Forcing it back on, then
  linking the ~large `torch._C` side module, conda-packaging it, and loading it
  in xeus at runtime (the sibling's **blocker #11**, an unresolved `GOT.func`
  function-pointer relocation in the huge module) is the work left on the hard
  time budget.

**Net:** the *compile* of a reduced, CPU-only, single-threaded upstream
`torch_cpu` (eager ATen + autograd + `torch::nn` + `torch::optim`) for
`wasm32-emscripten` **is solved end-to-end with the emscripten-forge toolchain**;
the remaining gap is packaging/loading the Python extension, not compiling the
tensor/autograd core.

## Environment / versions

| Component | Version |
| --- | --- |
| Host OS | Ubuntu 24.04 (x86_64), 4 CPU |
| emscripten-forge compiler (`emscripten_emscripten-wasm32`) | **3.1.73** (channel latest; `variant.yaml` HEAD is moving to 4.0.9 but that build is not yet published) |
| `cross-python_emscripten-wasm32` | **3.13.1** |
| Target wasm CPython | 3.13.15 |
| Build orchestrator | `rattler-build` 0.67.x (from `ci_env.yml`) |
| Side-module ABI (from the compiler `activate.sh`) | `-fPIC -O2`, `-s WASM=1 -sWASM_BIGINT`, `-s SIDE_MODULE=1`, JS-based exceptions (single-threaded; no `-pthread`) |
| PyTorch | 2.8.0 (GitHub release tarball, bundles `third_party/*`; sha256 `c70a2c94…`) |

## The recipe / how to reproduce

- **Deliverable recipe:** [`recipe/recipe.yaml`](recipe/recipe.yaml) +
  [`recipe/build.sh`](recipe/build.sh) +
  [`recipe/patches/0001-emscripten-wasm32-portability.patch`](recipe/patches/0001-emscripten-wasm32-portability.patch)
  + [`recipe/emscripten_fixups.cmake`](recipe/emscripten_fixups.cmake) +
  [`recipe/variant-3173.yaml`](recipe/variant-3173.yaml) (pins the toolchain to
  the published 3.1.73 / 3.13.1).
- **Iteration driver** actually used for the empirical run:
  [`build_iter.sh`](build_iter.sh) (drives `emcmake cmake` / `emmake ninja`
  against a persistent pre-extracted, pre-patched tree so ninja resumes
  incrementally across `rattler-build` re-invocations) and
  [`apply_patches.py`](apply_patches.py) (idempotent source edits).

Build command (per the emscripten-forge docs' rattler-build flow):

```bash
micromamba create -n ef -f recipes/ci_env.yml            # rattler-build etc.
rattler-build build \
  --recipe recipes/recipes_emscripten/pytorch/recipe.yaml \
  --target-platform=emscripten-wasm32 \
  -c https://prefix.dev/emscripten-forge-dev -c conda-forge -c microsoft \
  -m variant-3173.yaml --keep-build
```

## Reduced configuration (what is disabled)

`USE_CUDA/ROCM/XPU=0`, `USE_MKLDNN/FBGEMM/NNPACK/QNNPACK/PYTORCH_QNNPACK/
XNNPACK=0`, `USE_KINETO=0`, `USE_DISTRIBUTED/TENSORPIPE/GLOO/MPI=0`,
`USE_OPENMP=0` (single-threaded, `ATEN_THREADING=NATIVE`), `USE_NUMA=0`,
`USE_NUMPY=0`, `USE_MAGMA=0`, `USE_ITT=0`, `USE_MIMALLOC=0`, `USE_OBSERVERS=0`,
`USE_ONNX=0`, `BUILD_CAFFE2/CAFFE2_OPS=0`, `BUILD_TEST=0`, `BUILD_BINARY=0`,
`USE_LITE_PROTO=1`. **Kept:** eager ATen CPU ops, autograd, TorchScript/JIT,
`torch::nn`, `torch::optim`.

## Blockers found & fixed (in build order)

Each fix is in [`apply_patches.py`](apply_patches.py) /
[`recipe/emscripten_fixups.cmake`](recipe/emscripten_fixups.cmake) /
[`build_iter.sh`](build_iter.sh).

1. **`emscripten_emscripten-wasm32=4.0.9` does not exist in the channel.** The
   recipes-repo `variant.yaml` HEAD pins 4.0.9, but the published compiler is
   **3.1.73** (`micromamba search`). Matching the *published* toolchain also
   matches the published `xeus-python` side-module ABI. Fix:
   [`recipe/variant-3173.yaml`](recipe/variant-3173.yaml).
2. **`Python::Module` target not found** (`cmake/Dependencies.cmake`). wasm
   CPython has no shared `libpython`, so `find_package(Python …
   Development.Module)` never creates the target that torch links into
   pybind11. Fix: inject a header-only `Python::Module` / `Python::Python`
   INTERFACE stub via `-DCMAKE_PROJECT_INCLUDE`
   ([`recipe/emscripten_fixups.cmake`](recipe/emscripten_fixups.cmake)). (Same
   class as the Pyodide sibling's blocker #2.)
3. **`install(EXPORT Caffe2Targets)` fails at generate time** — with the reduced
   feature set, private static deps of `torch_cpu` (`fp16`, `flatbuffers`,
   `onnx_library`, and the `ATEN_CPU_FILES_GEN_LIB` *custom* target) are "not in
   any export set". We only need to compile, so the CMake package-config export
   is disabled (`CMakeLists.txt`, `if(NOT BUILD_LIBTORCHLESS)` → `if(FALSE)`).
4. **`SymInt * size_t` ambiguous overload (ILP32).** On wasm32 `size_t` is a
   distinct 32-bit type; upstream only declares those operators for `__APPLE__`.
   Fix: extend the guard to `__EMSCRIPTEN__` in `c10/core/SymInt.h`. (Sibling #5.)
5. **`__assert_fail` exception-spec mismatch.** `c10/macros/Macros.h`
   forward-declares glibc's `__assert_fail` with `noexcept` under `NDEBUG`;
   Emscripten musl differs. Fix: skip that declaration on `__EMSCRIPTEN__`.
   (Sibling #3.)
6. **`ssize_t` undeclared** in `c10/util/Enumerate.h` (only pulled in for
   `_WIN32`). Fix: `#include <sys/types.h>` on `__EMSCRIPTEN__`. (Sibling #6.)
7. **Cross-compiled `protoc` can't run as a host tool (exit 126).** The vendored
   protobuf builds a *wasm* `protoc`; ninja then tries to execute it to generate
   `onnx_onnx_torch-ml.pb.{cc,h}` and fails
   ([`logs/11`](logs/11-build-torch_cpu.log), first run). Fix: build a
   **version-matched host `protoc` (libprotoc 3.13.0)** from the vendored sources
   with system g++ and point `CAFFE2_CUSTOM_PROTOC_EXECUTABLE` at it
   ([`logs/12`](logs/12-hostprotoc.log)). (Sibling #4.) **With #1–#7 the entire
   `torch_cpu` compile + link succeeds** ([`logs/11`](logs/11-build-torch_cpu.log),
   `BUILD_TORCH_CPU_RC=0`).

## Remaining work (honest boundary)

Toward an *importable* `import torch` that trains an MLP in xeus-python:

- **Blocker #8 (next):** force `BUILD_PYTHON ON` despite the missing
  `Development.Module` (patch the branch in `cmake/Dependencies.cmake` that flips
  it off), so `libtorch_python` and the `torch._C` extension side module build.
  The configure log confirms it is currently auto-disabled
  (`BUILD_PYTHON : OFF`). This is the Pyodide sibling's blocker #8.
- **Link + package:** link the (large) `torch._C` side module against
  `libtorch_cpu.a`, wire the `.so` `NEEDED`/dylink metadata, and emit a
  `wasm32-emscripten` conda package hosted in a local channel referenced from
  `environment.yml` so `jupyterlite-xeus` packs it next to `xeus-python`.
- **Runtime load:** the Pyodide sibling reached this stage and hit **blocker
  #11** — an unresolved `GOT.func` function-pointer relocation aborting a
  `libtorch_cpu` static initializer at load. The emscripten-forge loader
  (xeus dynamic linker) is a different implementation, so this needs
  independent investigation, but it is the expected next runtime wall.

## Reviewer demo status

Because `torch._C` is not yet built/loadable (blocker #8 onward), there is **no
working in-browser `import torch` training demo yet** — presenting one would be
dishonest. The JupyterLite scaffolding under
[`../jupyterlite-torch-wasm/`](../jupyterlite-torch-wasm/) contains a
throwaway pure-Python `microtorch` placeholder from an earlier iteration; it is
**not** the deliverable and is retained only as a labelled placeholder. The
deliverable here is the **real-`torch` emscripten-forge build** documented above.

## Artifact sizes (wasm static archives)

| Artifact | Size |
| --- | --- |
| `libtorch_cpu.a` | 368 MB |
| `libonnx.a` | 12 MB |
| `libc10.a` | 2.2 MB |
| `libprotobuf-lite.a` | 0.86 MB |
| `libcpuinfo.a` | 13 KB |
