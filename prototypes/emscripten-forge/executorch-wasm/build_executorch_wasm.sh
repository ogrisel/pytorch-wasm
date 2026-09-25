#!/usr/bin/env bash
# Build ExecuTorch's in-tree WASM example (examples/wasm) and run model inference
# under Node.js. This is the "PyTorch on WASM" inference path.
#
# This script is a *verified working recipe*: on 2026-09-22 it built and ran both
# examples/models add_mul and mv2 (MobileNetV2) as wasm32-emscripten under Node.
# See RESULTS.md in this directory for the captured output, versions and blockers.
#
# It follows pytorch/executorch/examples/wasm/README.md (and its test_build_wasm.sh)
# rather than inventing its own CMake invocation. The only additions over upstream
# are host-environment fixes that a fresh Ubuntu-24.04 sandbox needs (see PREREQS).
set -euo pipefail

# ---------------------------------------------------------------------------
# PREREQS (host, one-time). On the sandbox these were required to get a working
# host toolchain + Python build environment before ExecuTorch would install:
#
#   # 1. A GCC C++ toolchain that can actually link. On this image /usr/bin/c++
#   #    resolved to clang-18, which auto-selected an incomplete gcc-14 and failed
#   #    with "cannot find -lstdc++". Point the alternatives at gcc-13:
#   sudo update-alternatives --set c++ /usr/bin/g++
#   sudo update-alternatives --set cc  /usr/bin/gcc
#
#   # 2. Python dev headers for the pybind11 extensions (pytorch_tokenizers etc.):
#   sudo apt-get update && sudo apt-get install -y python3-dev
#
#   # 3. ninja (no apt candidate on this image):
#   pip install ninja   # ends up on ~/.local/bin
#
#   # 4. PEP-668 externally-managed system Python:
#   export PIP_BREAK_SYSTEM_PACKAGES=1
#
#   # 5. Emscripten SDK. Upstream pins 4.0.10 (repo scripts/install-emsdk.sh
#   #    defaults to 3.1.58 which is fine for the tensor-kernel PoC but ExecuTorch
#   #    wants 4.0.10):
#   EMSDK_VERSION=4.0.10 ../../../scripts/install-emsdk.sh
#   source ../../../.emsdk/emsdk_env.sh
# ---------------------------------------------------------------------------

WORK="${WORK:-$(cd "$(dirname "$0")" && pwd)/.et-work}"
ET_REF="${ET_REF:-main}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
MODEL_NAME="${MODEL_NAME:-add_mul}"     # any name from examples/models (e.g. mv2)
export PATH="$HOME/.local/bin:$PATH"    # pip-installed ninja
export PIP_BREAK_SYSTEM_PACKAGES=1

if ! command -v emcc >/dev/null 2>&1; then
  echo "emcc not found; activate emsdk first (see PREREQS above)" >&2
  exit 1
fi
emcc --version | head -1

mkdir -p "${WORK}"
cd "${WORK}"

if [ ! -d executorch ]; then
  git clone --depth 1 --branch "${ET_REF}" https://github.com/pytorch/executorch.git
fi
cd executorch
echo "ExecuTorch commit: $(git rev-parse HEAD)"

# Host-side install: pulls torch, updates the required submodules (XNNPACK,
# flatbuffers, pthreadpool, ...) and builds+installs the executorch python
# package. Needed to export/lower example models to .pte.
if ! ${PYTHON_EXECUTABLE} -c "import executorch" 2>/dev/null; then
  ./install_executorch.sh
fi

# Build the host executorch libs + tools (libexecutorch.a, portable kernels, ...).
# This is examples/wasm/test_build_wasm.sh's cmake_install_executorch_lib step.
if [ ! -f cmake-out/lib/libexecutorch.a ]; then
  cmake -DCMAKE_INSTALL_PREFIX=cmake-out -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" -Bcmake-out .
  cmake --build cmake-out -j"$(nproc)" --target install --config Release
fi

# Export/lower the model to a .pte flatbuffer (host).
MODEL_DIR="$(pwd)/models_out"
rm -rf "${MODEL_DIR}"; mkdir -p "${MODEL_DIR}"
${PYTHON_EXECUTABLE} -m examples.portable.scripts.export \
  --model_name="${MODEL_NAME}" --output_dir="${MODEL_DIR}"

# Cross-compile the executor_runner to wasm32-emscripten, embedding ${MODEL_DIR}
# into the MEMFS virtual filesystem. emcmake injects the Emscripten toolchain file
# and uses node as the crosscompiling emulator.
BUILD_DIR="cmake-out/examples/wasm"
rm -rf "${BUILD_DIR}"
emcmake cmake -DWASM_MODEL_DIR="${MODEL_DIR}" -B"${BUILD_DIR}" .
cmake --build "${BUILD_DIR}" -j"$(nproc)" --target executor_runner

echo "=== build artifacts ==="
ls -la "${BUILD_DIR}"/executor_runner.* 2>/dev/null || true

# Run the wasm model under Node. Emscripten ships a compatible node as $EMSDK_NODE;
# system node >=18 also works.
echo "=== run under node ==="
"${EMSDK_NODE:-node}" "${BUILD_DIR}/executor_runner.js" --model_path="${MODEL_NAME}.pte"
