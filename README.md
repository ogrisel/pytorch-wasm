# pytorch-wasm

Agent-driven exploration of WebAssembly support for PyTorch.

This repository investigates **why PyTorch is not available on
[Pyodide](https://pyodide.org/) or [emscripten-forge](https://emscripten-forge.org/)**
and prototypes packages for both, using cloud sandbox subagents for the heavy builds.

## Contents

| Path | What it is |
| --- | --- |
| [`docs/research-why-no-pytorch-wasm.md`](docs/research-why-no-pytorch-wasm.md) | The blocker analysis: threading/pthreads, the (now-resolved) `cffi` blocker, native dependency graph, dynamic-vs-static linking, binary size, and maintainer economics — plus what changed in 2025–2026. |
| [`prototypes/emscripten-forge/tensor-kernel-wasm/`](prototypes/emscripten-forge/tensor-kernel-wasm/) | Local proof-of-concept: a PyTorch-style dense layer (matmul + bias + ReLU) compiled to `wasm32-emscripten` and self-verified under Node. |
| [`prototypes/emscripten-forge/executorch-wasm/`](prototypes/emscripten-forge/executorch-wasm/) | Driver for the ExecuTorch WASM inference prototype (cloud subagent). |
| [`prototypes/pyodide/`](prototypes/pyodide/) | Pyodide recipe scaffolds: an `executorch` bindings target and a minimal `torch` probe recipe. |
| [`scripts/install-emsdk.sh`](scripts/install-emsdk.sh) | Pinned Emscripten SDK installer. |

## Quick start (local WASM proof-of-concept)

```bash
./scripts/install-emsdk.sh
source .emsdk/emsdk_env.sh
cd prototypes/emscripten-forge/tensor-kernel-wasm
./build.sh
node run_node.mjs
```

## Executive summary

There is no fundamental blocker; PyTorch's absence is the compound result of a large
native dependency graph, historically-missing WASM threading, a historically-missing
`cffi` (now resolved), dynamic-vs-static linking mismatches, large binary size, and
limited maintainer bandwidth. The pragmatic path in 2026 is **ExecuTorch on WASM**
(inference), which PyTorch upstream is actively enabling and which emscripten-forge
already partially ships as `executorch-cpp`. See the research doc for details and
citations.
