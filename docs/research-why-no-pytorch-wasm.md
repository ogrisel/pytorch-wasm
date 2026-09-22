# Why PyTorch is not (yet) available on Pyodide or emscripten-forge

_Research notes for `pytorch-wasm` — an agent-driven exploration of WebAssembly support for PyTorch._

Last updated: 2026-09-22.

## TL;DR

There has never been a fundamental "it is impossible" reason. PyTorch's absence from
the two main scientific-Python-in-WASM distributions (Pyodide and emscripten-forge)
is the compound result of:

1. **PyTorch is huge and has a deep native dependency graph.** A full `libtorch`
   build pulls in XNNPACK, sleef, protobuf, cpuinfo, pthreadpool, FBGEMM/oneDNN
   (x86 only), fmt, etc. Every one of those has to be cross-compiled for the
   `wasm32-emscripten` target and made link-compatible before `torch` itself can
   link. This is a large recipe-graph problem, not a single patch.
2. **Historically, threading (pthreads) in WASM was effectively unavailable.**
   PyTorch's CPU path assumes a real threadpool. Emscripten pthreads rely on
   `SharedArrayBuffer`, which was disabled across browsers after Spectre (2018) and
   only came back gated behind COOP/COEP headers. Pyodide itself did not support
   threading until very recently, so a threaded `libtorch` had nowhere to run.
3. **Historically, `ctypes`/`cffi` were not supported in Pyodide.** PyTorch requires
   `cffi` at runtime; for years this was a hard blocker for a Pyodide package.
   **This blocker is now gone** — `cffi` is a shipping Pyodide recipe today
   (`packages/cffi/meta.yaml`, cffi 2.0.0) — but it shaped years of "not feasible"
   guidance.
4. **Dynamic vs. static linking mismatch.** Emscripten's `wasm-ld` cannot consume
   the prebuilt `libtorch_cpu.so` / `libtorch.so` shared objects (`unknown file
   type`), and it rejects GNU-ld flags such as `--as-needed`/`--no-as-needed` that
   PyTorch's CMake emits. A WASM build must be done from source with the Emscripten
   toolchain, not by relinking existing artifacts.
5. **Maintainer economics.** Both distributions have said repeatedly that a full
   `torch` recipe is "a very large project" for "fairly limited resources," and that
   ONNX Runtime Web / candle / tfjs already cover browser inference. So nobody with
   commit rights has driven it to completion.

The landscape has shifted a lot in 2025–2026: Pyodide now has an experimental
pthreads story, and PyTorch/ExecuTorch upstream is actively adding an Emscripten/WASM
build path ("Web Calling" work). The realistic near-term target is **ExecuTorch on
WASM** (inference runtime), not full eager-mode `libtorch`.

---

## Background: the two distributions

### Pyodide

Pyodide is CPython compiled to `wasm32-emscripten` plus a curated set of packages
built from source with the Emscripten toolchain (`pyodide-build`). Recipes live in
[`pyodide/pyodide-recipes`](https://github.com/pyodide/pyodide-recipes) as
`packages/<name>/meta.yaml`. Pure-Python wheels can be installed at runtime with
`micropip`, but anything with a compiled extension (like `torch`) must have a recipe
and be built into the distribution. As of this writing the recipes repo has ~335
packages; **none of them is `torch`/`pytorch`** (nor `onnx`).

### emscripten-forge

[`emscripten-forge`](https://emscripten-forge.org/) is a conda-forge-style channel
for the `emscripten-wasm32` platform (conda-forge does not build for it). Recipes
live in [`emscripten-forge/recipes`](https://github.com/emscripten-forge/recipes) as
`rattler-build` `recipe.yaml` files and are installed with `micromamba
--platform=emscripten-wasm32`. It already ships a deep native stack relevant to us:
`xnnpack`, `onnxruntime`, `numpy`, `scipy`, `sentencepiece`, `tokenizers-cpp`,
`protobuf`, `flatbuffers`, and — importantly — **`executorch-cpp`** (the ExecuTorch
C++ runtime as static libs for `emscripten-wasm32`). It does **not** ship a
Python-importable `torch`/`executorch` package.

---

## The concrete blockers, in detail

### 1. Threading / pthreads

- Emscripten implements pthreads on top of Web Workers + `SharedArrayBuffer`.
  `SharedArrayBuffer` was disabled by browsers after Spectre and re-enabled only
  behind cross-origin isolation (COOP: `same-origin`, COEP: `require-corp`). This
  makes any threaded WASM deployment opt-in and awkward.
- Pyodide tracked this for years in
  [pyodide#237](https://github.com/pyodide/pyodide/issues/237) and only reached an
  experimental threading prototype recently
  ([pyodide#6284](https://github.com/pyodide/pyodide/issues/6284)). Making it work
  required patching CPython's wasm-gc call trampoline (each pthread has its own
  function table; the trampoline funcptr created via `addFunction` on the main
  thread dies on spawned threads with `table index is out of bounds`), stripping
  `-sPROXY_TO_PTHREAD` from `LINKFORSHARED`, and a patched `pyodide-build` that
  stops discarding `-pthread`.
- PyTorch's ATen parallel backend expects a real threadpool. You can force
  single-threaded (`AT_PARALLEL_NATIVE` with 1 thread, `-sUSE_PTHREADS=0`), but then
  performance is poor and some third-party deps still assume atomics/bulk-memory.
- **`-pthread` is an ABI break in WASM**: object files compiled with atomics/
  bulk-memory cannot be linked with ones that were not
  (`error: --shared-memory is disallowed by X.o because it was not compiled with
  'atomics' or 'bulk-memory' features`). So the *entire* dependency graph must be
  built consistently. Upstream PyTorch addressed exactly this in
  [pytorch/pytorch#176542](https://github.com/pytorch/pytorch/pull/176542) by adding
  `-pthread -matomics -mbulk-memory` to cpuinfo, clog, pthreadpool, and XNNPACK.

### 2. `cffi` / `ctypes` (historical, now resolved)

- The classic 2020 answer ([SO](https://stackoverflow.com/questions/64358372/run-pytorch-in-pyodide),
  [pyodide-recipes#193](https://github.com/pyodide/pyodide-recipes/issues/193))
  cited `cffi` not being supported in Pyodide as a hard blocker for `torch`.
- **This is no longer true**: Pyodide ships `cffi` today
  (`packages/cffi/meta.yaml`). So this specific blocker should be struck from any
  "why not" list going forward.

### 3. Dependency graph size and native linking

- Building `torch` from source for WASM means first getting these to link as
  `wasm32` static libs / side modules: `sleef` (vectorized math), `XNNPACK`
  (needs its **WASM microkernels** explicitly compiled — otherwise runtime
  `Aborted(-1)` with missing `xnn_*__wasm_*` symbols, see
  [pytorch/pytorch#177983](https://github.com/pytorch/pytorch/commit/dc443f0eb1c93c8b32d210fa5330a7838a70c37b)),
  `pthreadpool`, `cpuinfo`/`clog`, `protobuf` (must be cross-compiled; generated
  `*.pb.cc` linked against a wasm-compatible libprotobuf), `fmt`, `FP16`/`FXdiv`,
  and `onnx`. x86-only pieces (FBGEMM, oneDNN/MKLDNN, NNPACK) must be disabled.
- Prebuilt artifacts do not help: `wasm-ld` reports `unknown file type` for
  `libtorch_cpu.so`/`libtorch.so` and rejects GNU-ld-isms like `--as-needed`
  ([SO](https://stackoverflow.com/questions/64062268/compiling-libtorch-to-webassembly-with-emscripten-using-cmake-build),
  [pytorch/pytorch#25691/#25699]). Everything must be built from source with
  `emcmake`/`emmake`.

### 4. Binary size and practicality

- Even CPU-only, a full `libtorch` WASM binary is very large (tens of MB), which is
  hostile to a browser download. This is why the community consistently steers users
  to ONNX Runtime Web, TF.js, `candle`, or a tree-shaken subset rather than full
  eager PyTorch.

### 5. Maintainer bandwidth / prioritization

- Both projects have explicitly framed a full `torch` recipe as high-effort, CPU-only,
  and lower-value than existing browser-inference options
  ([pyodide-recipes#193](https://github.com/pyodide/pyodide-recipes/issues/193)).

---

## What changed recently (2025–2026) — why a prototype is now realistic

- **Pyodide threading prototype** exists ([#6284](https://github.com/pyodide/pyodide/issues/6284));
  a downstream fork demonstrated a threaded boot in late 2025.
- **`cffi` now ships in Pyodide**, removing a long-standing hard blocker.
- **ExecuTorch is the pragmatic target.** ExecuTorch 1.0 (GA) added *experimental*
  JavaScript/browser support via WASM. PyTorch's "Web Calling" workstream added:
  - WASM/Emscripten compiler flags across third-party deps
    ([#176542](https://github.com/pytorch/pytorch/pull/176542)),
  - an XNNPACK WASM microkernel target
    ([#177983](https://github.com/pytorch/pytorch/commit/dc443f0eb1c93c8b32d210fa5330a7838a70c37b)),
  - and there is an in-tree example at
    [`pytorch/executorch/examples/wasm`](https://github.com/pytorch/executorch/tree/main/examples/wasm).
- **emscripten-forge already ships `executorch-cpp`** — the ExecuTorch C++ runtime
  built for `emscripten-wasm32` (static libs + CMake config), built with
  `-DEXECUTORCH_BUILD_WASM=ON`, XNNPACK off, portable kernels on. See
  `recipes/recipes_emscripten/executorch-cpp/` in the recipes repo. The missing
  piece is a **Python-importable** package (`torch`/`executorch`) on top of it.

---

## Recommended prototyping strategy (what this repo attempts)

Full eager-mode `libtorch` for WASM is a multi-subsystem port. Instead, prototype in
order of increasing difficulty and demonstrate each rung:

1. **Runtime toolchain hello-world (this sandbox):** install `emsdk`, compile a
   small C++ tensor kernel (matmul/relu) to WASM, run it under Node — proves the
   `wasm32-emscripten` numerical path end-to-end.
2. **emscripten-forge / ExecuTorch WASM (cloud subagent):** build the in-tree
   `pytorch/executorch/examples/wasm` (ExecuTorch runtime + XNNPACK WASM kernels +
   a lowered model) and run inference in Node/WASM. This is "PyTorch on WASM" in the
   inference sense and reuses the existing `executorch-cpp` recipe learnings.
3. **Pyodide package (cloud subagent):** attempt a Pyodide recipe for `executorch`
   (Python pybindings over the WASM runtime) — the realistic Python-importable
   target — and, separately, probe how far a minimal from-source `torch` recipe gets
   before hitting the size/threading wall, capturing the exact failure points.

See `../prototypes/emscripten-forge/README.md` and `../prototypes/pyodide/README.md`
for the concrete recipes and build scripts the cloud subagents run.

## Key references

- Pyodide PyTorch support: https://github.com/pyodide/pyodide-recipes/issues/193
- Upstream PyTorch WASM issue: https://github.com/pytorch/pytorch/issues/25691
- Run pytorch in pyodide (cffi/ctypes blocker): https://stackoverflow.com/questions/64358372/run-pytorch-in-pyodide
- Compiling libtorch with Emscripten (linking blockers): https://stackoverflow.com/questions/64062268/compiling-libtorch-to-webassembly-with-emscripten-using-cmake-build
- Pyodide threading: https://github.com/pyodide/pyodide/issues/237 and https://github.com/pyodide/pyodide/issues/6284
- PyTorch "Web Calling" WASM flags: https://github.com/pytorch/pytorch/pull/176542
- XNNPACK WASM microkernels: https://github.com/pytorch/pytorch/commit/dc443f0eb1c93c8b32d210fa5330a7838a70c37b
- ExecuTorch WASM example: https://github.com/pytorch/executorch/tree/main/examples/wasm
- ExecuTorch web runtime discussion: https://github.com/pytorch/executorch/discussions/8216
- emscripten-forge: https://emscripten-forge.org/ and https://github.com/emscripten-forge/recipes
