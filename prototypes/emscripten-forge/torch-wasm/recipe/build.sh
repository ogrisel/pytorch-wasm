#!/bin/bash
# Reduced, CPU-only, single-threaded PyTorch build for emscripten-wasm32.
#
# The emscripten-forge compiler activation (activate.sh) already exports:
#   CC=emcc CXX=em++ AR=emar RANLIB=emranlib
#   cmake()  -> emcmake cmake
#   CMAKE_ARGS with the Emscripten toolchain file + CMAKE_PREFIX_PATH=$PREFIX
#   EMCC_CFLAGS with -fPIC -O2 -msimd128 -fwasm-exceptions -sSUPPORT_LONGJMP=wasm
#   EM_FORGE_SIDE_MODULE_{C,LD}FLAGS with -s SIDE_MODULE=1
# so every object is compiled with the wasm-native EH ABI shared by all
# emscripten-forge side modules (xeus-python included).
set -euo pipefail

export USE_CUDA=0 USE_ROCM=0 USE_XPU=0
export USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 USE_XNNPACK=0
export USE_KINETO=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 USE_GLOO=0 USE_MPI=0
export USE_OPENMP=0 USE_NUMA=0 USE_NUMPY=0 USE_MAGMA=0 USE_ITT=0 USE_MIMALLOC=0
export USE_OBSERVERS=0 USE_LITE_PROTO=1 USE_ONNX=0
export BUILD_TEST=0 BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 BUILD_BINARY=0
export ATEN_THREADING=NATIVE
export MAX_JOBS="${CPU_COUNT:-4}"
export PYTORCH_BUILD_VERSION="${PKG_VERSION:-2.8.0}"
export PYTORCH_BUILD_NUMBER=0
export CMAKE_BUILD_TYPE=Release

export CFLAGS="${CFLAGS:-} ${EM_FORGE_SIDE_MODULE_CFLAGS:-}"
export CXXFLAGS="${CXXFLAGS:-} ${EM_FORGE_SIDE_MODULE_CFLAGS:-}"
export LDFLAGS="${LDFLAGS:-} ${EM_FORGE_SIDE_MODULE_LDFLAGS:-}"

# Reduced CMake configuration for the CPU-only, single-threaded core libs.
CMAKE_REDUCED_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_PYTHON=ON -DBUILD_TEST=OFF -DBUILD_BINARY=OFF
  -DBUILD_CAFFE2=OFF -DBUILD_CAFFE2_OPS=OFF
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF
  -DUSE_MKLDNN=OFF -DUSE_FBGEMM=OFF -DUSE_NNPACK=OFF
  -DUSE_QNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_XNNPACK=OFF
  -DUSE_KINETO=OFF -DUSE_DISTRIBUTED=OFF -DUSE_TENSORPIPE=OFF
  -DUSE_GLOO=OFF -DUSE_MPI=OFF -DUSE_OPENMP=OFF -DUSE_NUMA=OFF
  -DUSE_NUMPY=OFF -DUSE_MAGMA=OFF -DUSE_ITT=OFF -DUSE_MIMALLOC=OFF
  -DUSE_OBSERVERS=OFF -DUSE_LITE_PROTO=ON -DUSE_ONNX=OFF
  -DATEN_THREADING=NATIVE
  -DPython_EXECUTABLE="${PYTHON:-python}"
)

# Under emscripten-forge activation `cmake` is already `emcmake cmake`.
cmake -S "${SRC_DIR}" -B "${SRC_DIR}/build-wasm" -GNinja "${CMAKE_REDUCED_ARGS[@]}"
cmake --build "${SRC_DIR}/build-wasm" --target torch_cpu -j "${MAX_JOBS}"
cmake --build "${SRC_DIR}/build-wasm" -j "${MAX_JOBS}"
cmake --install "${SRC_DIR}/build-wasm" --prefix "${PREFIX}"
