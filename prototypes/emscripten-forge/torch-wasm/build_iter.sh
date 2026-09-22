#!/bin/bash
# Iterative driver for the reduced emscripten-wasm32 PyTorch build.
# Runs inside rattler-build's activated build env (emcc/cross-python wrappers,
# CMAKE_ARGS toolchain file, EMSDK_PYTHON, host $PREFIX all set correctly).
set -uxo pipefail

SRC=/workspace/.src/pytorch
BUILD=/workspace/.torch-build
LOG=/workspace/prototypes/emscripten-forge/torch-wasm/logs

mkdir -p "$BUILD" "$LOG"

echo "=== emcc / python probe ==="
emcc --version | head -1 || true
echo "EMSDK_PYTHON=${EMSDK_PYTHON:-unset} PYTHON=${PYTHON:-unset}"
echo "BUILD_PREFIX=${BUILD_PREFIX:-unset}"
echo "PREFIX=${PREFIX:-unset}"
echo "CMAKE_ARGS=${CMAKE_ARGS:-unset}"

# Reduced, CPU-only, single-threaded feature set.
export USE_CUDA=0 USE_ROCM=0 USE_XPU=0
export USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 USE_XNNPACK=0
export USE_KINETO=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 USE_GLOO=0 USE_MPI=0
export USE_OPENMP=0 USE_NUMA=0 USE_NUMPY=0 USE_MAGMA=0 USE_ITT=0 USE_MIMALLOC=0
export USE_OBSERVERS=0 USE_LITE_PROTO=1 USE_ONNX=0
export BUILD_TEST=0 BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 BUILD_BINARY=0
export ATEN_THREADING=NATIVE
export MAX_JOBS="${CPU_COUNT:-4}"

# Torch throws C++ exceptions; emscripten 3.1.73 defaults to disabled catching.
# Match the emscripten-forge side-module ABI (JS-based EH on 3.1.73).
EH="-fexceptions"
export CFLAGS="${CFLAGS:-} ${EM_FORGE_SIDE_MODULE_CFLAGS:-} ${EH}"
export CXXFLAGS="${CXXFLAGS:-} ${EM_FORGE_SIDE_MODULE_CFLAGS:-} ${EH}"
export LDFLAGS="${LDFLAGS:-} ${EM_FORGE_SIDE_MODULE_LDFLAGS:-} ${EH}"

PYEXE="${EMSDK_PYTHON:-${BUILD_PREFIX}/bin/python3}"

CFG_ARGS=(
  -GNinja
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
  -DPython_EXECUTABLE="${PYEXE}"
  -DPYTHON_EXECUTABLE="${PYEXE}"
  -DCMAKE_PROJECT_INCLUDE=/workspace/prototypes/emscripten-forge/torch-wasm/recipe/emscripten_fixups.cmake
  -DWASM_PYTHON_INCLUDE_DIR="${PREFIX}/include/python3.13"
  # The vendored protoc cross-compiles to a wasm binary that cannot run as a
  # host tool (exit 126). Use a version-matched host protoc (protobuf 3.13.0).
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=/workspace/.hostprotoc/protoc
)

# Configure (idempotent; reuses cache on re-runs).
emcmake cmake -S "$SRC" -B "$BUILD" "${CFG_ARGS[@]}" ${CMAKE_ARGS:-} 2>&1 | tee "$LOG/10-configure.log"
cfg_rc=${PIPESTATUS[0]}
echo "CONFIGURE_RC=$cfg_rc" | tee -a "$LOG/10-configure.log"
if [ "$cfg_rc" != "0" ]; then
  echo "configure failed; stopping."
  exit "$cfg_rc"
fi

# Build the core CPU library first (surfaces the bulk of compile blockers).
emmake cmake --build "$BUILD" --target torch_cpu -j "$MAX_JOBS" 2>&1 | tee "$LOG/11-build-torch_cpu.log"
tc_rc=${PIPESTATUS[0]}
echo "BUILD_TORCH_CPU_RC=$tc_rc" | tee -a "$LOG/11-build-torch_cpu.log"
[ "$tc_rc" = "0" ] || exit "$tc_rc"

# Continue to the remaining targets: libtorch, libtorch_python and the
# torch._C python extension side module (the importable surface).
emmake cmake --build "$BUILD" -j "$MAX_JOBS" 2>&1 | tee "$LOG/13-build-all.log"
echo "BUILD_ALL_RC=${PIPESTATUS[0]}" | tee -a "$LOG/13-build-all.log"
