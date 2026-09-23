#!/usr/bin/env bash
# Compile the tensor kernel to wasm32-emscripten. Requires an activated emsdk
# (run ../../../scripts/install-emsdk.sh and `source .emsdk/emsdk_env.sh` first).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v emcc >/dev/null 2>&1; then
  echo "emcc not found. Run scripts/install-emsdk.sh and source .emsdk/emsdk_env.sh" >&2
  exit 1
fi

emcc --version | head -1

emcc tensor.cpp \
  -O3 \
  -lembind \
  -sMODULARIZE=1 \
  -sEXPORT_ES6=1 \
  -sENVIRONMENT=node,web \
  -sALLOW_MEMORY_GROWTH=1 \
  -o tensor_kernel.mjs

echo "Built: $(pwd)/tensor_kernel.mjs + tensor_kernel.wasm"
