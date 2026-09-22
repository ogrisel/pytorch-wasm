# emscripten-forge / Emscripten PyTorch prototypes

This directory holds two things:

1. **`tensor-kernel-wasm/`** — a self-contained, local proof-of-concept: a tiny
   C++ tensor library (matmul + bias + ReLU, i.e. one dense layer) compiled to
   `wasm32-emscripten` and run under Node.js. It has no PyTorch dependency and is
   meant to prove the numerical WASM toolchain works end-to-end in a few seconds,
   before attempting the multi-GB ExecuTorch build. Build it with:

   ```bash
   ../../scripts/install-emsdk.sh
   source ../../.emsdk/emsdk_env.sh
   cd tensor-kernel-wasm && ./build.sh && node run_node.mjs
   ```

2. **`executorch-wasm/`** — the plan + driver script for the heavier prototype that
   a cloud subagent runs: build the in-tree
   [`pytorch/executorch/examples/wasm`](https://github.com/pytorch/executorch/tree/main/examples/wasm)
   target (ExecuTorch runtime + XNNPACK WASM microkernels + a lowered model) and run
   inference in WASM. This is the realistic "PyTorch on WASM" inference path and
   builds on the existing emscripten-forge `executorch-cpp` recipe.

## Why ExecuTorch instead of full `libtorch`?

See `../../docs/research-why-no-pytorch-wasm.md`. Full eager `libtorch` for WASM is a
multi-subsystem port (sleef, protobuf, FBGEMM/oneDNN off, XNNPACK WASM kernels,
consistent `-pthread` ABI across the whole graph, tens of MB output). ExecuTorch is
purpose-built for constrained targets, already has an in-tree WASM example, and
emscripten-forge already ships `executorch-cpp`, so it is the pragmatic first rung.

## Relationship to emscripten-forge recipes

The upstream recipe `recipes/recipes_emscripten/executorch-cpp/recipe.yaml` builds the
C++ runtime static libraries for `emscripten-wasm32` with:

- `-DEXECUTORCH_BUILD_WASM=ON`
- portable kernels on, XNNPACK/pthreadpool/cpuinfo off
- `flatbuffers`, `flatcc`, `gflags`, `nlohmann_json`, `fxdiv` as deps

Our prototype reuses those flags but additionally wires up XNNPACK WASM microkernels
and a runnable example, and (in the Pyodide sibling dir) targets Python bindings.
