#!/bin/bash
# Full driver: reduced torch for emscripten-wasm32 WITH BUILD_PYTHON forced on,
# then link a single torch/_C.so SIDE_MODULE that statically pulls in
# libtorch_python + libtorch_cpu + libc10 (so cross-.so relocation is avoided).
set -uxo pipefail

SRC=/workspace/.src/pytorch
BUILD=/workspace/.torch-build
STAGE=/workspace/.torch-stage
LOG=/workspace/prototypes/emscripten-forge/torch-wasm/logs
mkdir -p "$BUILD" "$LOG" "$STAGE"

echo "=== probe ==="; emcc --version | head -1 || true
echo "EMSDK_PYTHON=${EMSDK_PYTHON:-unset} PREFIX=${PREFIX:-unset} BUILD_PREFIX=${BUILD_PREFIX:-unset}"

export USE_CUDA=0 USE_ROCM=0 USE_XPU=0
export USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 USE_XNNPACK=0
export USE_KINETO=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 USE_GLOO=0 USE_MPI=0
export USE_OPENMP=0 USE_NUMA=0 USE_NUMPY=0 USE_MAGMA=0 USE_ITT=0 USE_MIMALLOC=0
export USE_OBSERVERS=0 USE_LITE_PROTO=1 USE_ONNX=0
export BUILD_TEST=0 BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 BUILD_BINARY=0
export ATEN_THREADING=NATIVE
export MAX_JOBS="${CPU_COUNT:-4}"
export PYTORCH_BUILD_VERSION=2.8.0 PYTORCH_BUILD_NUMBER=0

EH="-fexceptions"
export CFLAGS="${CFLAGS:-} ${EM_FORGE_SIDE_MODULE_CFLAGS:-} ${EH}"
export CXXFLAGS="${CXXFLAGS:-} ${EM_FORGE_SIDE_MODULE_CFLAGS:-} ${EH}"
export LDFLAGS="${LDFLAGS:-} ${EM_FORGE_SIDE_MODULE_LDFLAGS:-} ${EH}"

PYEXE="${EMSDK_PYTHON:-${BUILD_PREFIX}/bin/python3}"
PYINC="${PREFIX}/include/python3.13"

emcmake cmake -S "$SRC" -B "$BUILD" -GNinja \
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
  -DCMAKE_PROJECT_INCLUDE=/workspace/prototypes/emscripten-forge/torch-wasm/recipe/emscripten_fixups.cmake \
  -DWASM_PYTHON_INCLUDE_DIR="${PYINC}" \
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=/workspace/.hostprotoc/protoc \
  ${CMAKE_ARGS:-} 2>&1 | tee "$LOG/10-configure.log"
cfg=${PIPESTATUS[0]}; echo "CONFIGURE_RC=$cfg" | tee -a "$LOG/10-configure.log"
[ "$cfg" = 0 ] || exit "$cfg"

# Confirm BUILD_PYTHON stuck ON.
grep -E "BUILD_PYTHON" "$LOG/10-configure.log" | tail -3 || true

emmake cmake --build "$BUILD" --target torch_python -j "$MAX_JOBS" 2>&1 | tee "$LOG/14-build-torch_python.log"
tp=${PIPESTATUS[0]}; echo "BUILD_TORCH_PYTHON_RC=$tp" | tee -a "$LOG/14-build-torch_python.log"
[ "$tp" = 0 ] || exit "$tp"

echo "=== static libs produced ===" | tee "$LOG/15-link-_C.log"
ls -la "$BUILD"/lib/*.a 2>&1 | tee -a "$LOG/15-link-_C.log"

# --- Link a single torch/_C.so SIDE_MODULE -----------------------------------
WHOLE=()
for l in libtorch_python.a libtorch.a libtorch_cpu.a; do
  [ -f "$BUILD/lib/$l" ] && WHOLE+=("$BUILD/lib/$l")
done
# Every other static archive under the build tree (deps), de-duplicated.
OTHER=()
while IFS= read -r a; do
  case "$a" in
    */libtorch_python.a|*/libtorch.a|*/libtorch_cpu.a) ;;
    *) OTHER+=("$a") ;;
  esac
done < <(find "$BUILD" -name '*.a' | sort -u)

set +e
# stub.c defines PyInit__C -> initModule(). It must be compiled as C so the
# reference to initModule keeps C linkage (`initModule`), matching the
# extern "C" definition in Module.cpp. Compiling it with em++ (C++) mangles the
# reference to _Z10initModulev, which then has no definition and becomes an
# unresolvable SIDE_MODULE import at load time.
emcc -sSIDE_MODULE=1 -fexceptions -O2 -I"$PYINC" \
  -c "$SRC/torch/csrc/stub.c" -o "$BUILD/stub.o" 2>&1 | tee -a "$LOG/15-link-_C.log"
# cpuinfo's dispatcher references cpuinfo_emscripten_init(), whose definition
# lives in third_party/cpuinfo/src/emscripten/init.c but is not compiled by the
# vendored cpuinfo CMake for this target. Compile it explicitly so the symbol
# resolves at load time instead of becoming an unresolvable SIDE_MODULE import.
CPUINFO="$SRC/third_party/cpuinfo"
emcc -sSIDE_MODULE=1 -fexceptions -O2 -DCPUINFO_LOG_LEVEL=2 -DCPUINFO_LOG_TO_STDIO=1 \
  -I"$CPUINFO/include" -I"$CPUINFO/src" \
  -c "$CPUINFO/src/emscripten/init.c" -o "$BUILD/cpuinfo_emscripten_init.o" 2>&1 | tee -a "$LOG/15-link-_C.log"
em++ -sSIDE_MODULE=1 -sWASM_BIGINT $EH -O2 -I"$PYINC" \
  "$BUILD/stub.o" "$BUILD/cpuinfo_emscripten_init.o" \
  -Wl,--whole-archive "${WHOLE[@]}" -Wl,--no-whole-archive \
  -Wl,--start-group "${OTHER[@]}" -Wl,--end-group \
  -o "$STAGE/_C.so" 2>&1 | tee -a "$LOG/15-link-_C.log"
lrc=${PIPESTATUS[0]}
set -e
echo "LINK_C_RC=$lrc" | tee -a "$LOG/15-link-_C.log"
ls -la "$STAGE/_C.so" 2>&1 | tee -a "$LOG/15-link-_C.log" || true
[ -f "$STAGE/_C.so" ] && file "$STAGE/_C.so" | tee -a "$LOG/15-link-_C.log" || true
echo "DONE"
