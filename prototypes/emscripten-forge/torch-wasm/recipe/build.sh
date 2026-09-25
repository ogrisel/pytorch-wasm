#!/bin/bash
# Reduced, CPU-only, single-threaded PyTorch build for emscripten-wasm32.
#
# Runs inside rattler-build's activated build env. The emscripten-forge
# compiler activation exports CC=emcc CXX=em++, EMSDK_PYTHON, the Emscripten
# toolchain file via CMAKE_ARGS, EM_FORGE_SIDE_MODULE_{C,LD}FLAGS (-s
# SIDE_MODULE=1), and the single-threaded / JS-EH side-module ABI shared by
# every emscripten-forge side module (xeus-python included).
#
# See ../RESULTS.md for the empirical run and the exact blockers each step
# works around. This recipe reaches a linked wasm `libtorch_cpu.a`
# (ATen CPU + autograd + torch::nn + torch::optim). Producing the importable
# torch._C side module additionally requires forcing BUILD_PYTHON on (the
# Development.Module / blocker #8 work) and is documented as remaining work.
set -euxo pipefail

SRC="${SRC_DIR}"
BUILD="${SRC_DIR}/build-wasm"
mkdir -p "${BUILD}"

# --- Host protoc (blocker #4) -------------------------------------------------
# The vendored protoc cross-compiles to a wasm binary that cannot execute as a
# host tool. Build a version-matched host protoc (libprotoc 3.13.0) with a host
# compiler and point CAFFE2_CUSTOM_PROTOC_EXECUTABLE at it.
HOST_PROTOC_DIR="${SRC_DIR}/.hostprotoc"
if [ ! -x "${HOST_PROTOC_DIR}/protoc" ]; then
  HOST_CXX="$(command -v g++ || command -v clang++)"
  HOST_CC="$(command -v gcc || command -v clang)"
  "${CMAKE:-cmake}" -S "${SRC}/third_party/protobuf/cmake" -B "${HOST_PROTOC_DIR}" \
    -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${HOST_CC}" -DCMAKE_CXX_COMPILER="${HOST_CXX}" \
    -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF
  "${CMAKE:-cmake}" --build "${HOST_PROTOC_DIR}" --target protoc -j "${CPU_COUNT:-4}"
fi

# --- Reduced feature set ------------------------------------------------------
export USE_CUDA=0 USE_ROCM=0 USE_XPU=0
export USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 USE_XNNPACK=0
export USE_KINETO=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 USE_GLOO=0 USE_MPI=0
export USE_OPENMP=0 USE_NUMA=0 USE_NUMPY=0 USE_MAGMA=0 USE_ITT=0 USE_MIMALLOC=0
export USE_OBSERVERS=0 USE_LITE_PROTO=1 USE_ONNX=0
export BUILD_TEST=0 BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 BUILD_BINARY=0
export ATEN_THREADING=NATIVE
export MAX_JOBS="${CPU_COUNT:-4}"

# torch throws C++ exceptions; match the emscripten-forge side-module EH ABI.
EH="-fexceptions"
export CFLAGS="${CFLAGS:-} ${EM_FORGE_SIDE_MODULE_CFLAGS:-} ${EH}"
export CXXFLAGS="${CXXFLAGS:-} ${EM_FORGE_SIDE_MODULE_CFLAGS:-} ${EH}"
export LDFLAGS="${LDFLAGS:-} ${EM_FORGE_SIDE_MODULE_LDFLAGS:-} ${EH}"

PYEXE="${EMSDK_PYTHON:-${BUILD_PREFIX}/bin/python3}"

emcmake cmake -S "${SRC}" -B "${BUILD}" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON=ON -DBUILD_TEST=OFF -DBUILD_BINARY=OFF \
  -DBUILD_CAFFE2=OFF -DBUILD_CAFFE2_OPS=OFF \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF \
  -DUSE_MKLDNN=OFF -DUSE_FBGEMM=OFF -DUSE_NNPACK=OFF \
  -DUSE_QNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_XNNPACK=OFF \
  -DUSE_KINETO=OFF -DUSE_DISTRIBUTED=OFF -DUSE_TENSORPIPE=OFF \
  -DUSE_GLOO=OFF -DUSE_MPI=OFF -DUSE_OPENMP=OFF -DUSE_NUMA=OFF \
  -DUSE_NUMPY=OFF -DUSE_MAGMA=OFF -DUSE_ITT=OFF -DUSE_MIMALLOC=OFF \
  -DUSE_OBSERVERS=OFF -DUSE_LITE_PROTO=ON -DUSE_ONNX=OFF \
  -DATEN_THREADING=NATIVE \
  -DPython_EXECUTABLE="${PYEXE}" -DPYTHON_EXECUTABLE="${PYEXE}" \
  -DCMAKE_PROJECT_INCLUDE="${RECIPE_DIR}/emscripten_fixups.cmake" \
  -DWASM_PYTHON_INCLUDE_DIR="${PREFIX}/include/python3.13" \
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE="${HOST_PROTOC_DIR}/protoc" \
  ${CMAKE_ARGS:-}

emmake cmake --build "${BUILD}" --target torch_cpu -j "${MAX_JOBS}"
cmake --install "${BUILD}" --prefix "${PREFIX}" || true
