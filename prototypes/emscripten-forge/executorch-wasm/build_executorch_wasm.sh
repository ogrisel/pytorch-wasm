#!/usr/bin/env bash
# Prototype driver: build ExecuTorch's in-tree WASM example and run it under Node.
#
# This is the heavier "PyTorch on WASM" inference prototype run by a cloud subagent.
# It follows pytorch/executorch/examples/wasm. Expect this to take a long time and to
# require iteration; the goal is to get as far as possible and capture exact failures
# in RESULTS.md.
#
# Prereqs the subagent must ensure: python3, cmake, ninja, git, node; emsdk activated.
set -euo pipefail

WORK="${WORK:-$(pwd)/.et-work}"
ET_REF="${ET_REF:-main}"
mkdir -p "${WORK}"
cd "${WORK}"

if ! command -v emcc >/dev/null 2>&1; then
  echo "emcc not found; activate emsdk first (scripts/install-emsdk.sh)" >&2
  exit 1
fi

if [ ! -d executorch ]; then
  git clone --depth 1 --branch "${ET_REF}" https://github.com/pytorch/executorch.git
fi
cd executorch
git submodule sync --recursive
git submodule update --init --recursive

# Host-side: install the python package so we can export/lower an example model.
python3 -m pip install --upgrade pip
./install_executorch.sh || python3 -m pip install -e . || true

# Follow the example's own README for the exact emcmake invocation; it wires up
# XNNPACK WASM microkernels and the wasm runtime wrapper.
echo "=== examples/wasm contents ==="
ls -la examples/wasm || { echo "examples/wasm missing at ref ${ET_REF}" >&2; exit 2; }
cat examples/wasm/README.md || true

# The example provides a build script / CMake target. Invoke via emcmake.
emcmake cmake -S examples/wasm -B examples/wasm/cmake-out -GNinja \
  -DCMAKE_BUILD_TYPE=Release || true
emmake cmake --build examples/wasm/cmake-out -j"$(nproc)" || true

echo "=== build artifacts ==="
find examples/wasm/cmake-out -name '*.wasm' -o -name '*.js' 2>/dev/null || true
