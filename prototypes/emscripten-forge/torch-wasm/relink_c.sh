#!/bin/bash
set -uxo pipefail
# Relink torch/_C.so with stub.c compiled as C (fixes _Z10initModulev import).
export PREFIX=/workspace/.mmroot/envs/torch-build
export BUILD_PREFIX=/workspace/.mmroot/envs/torch-build
export PATH="$PREFIX/bin:$PREFIX/opt/emsdk:$PREFIX/opt/emsdk/upstream/emscripten:$PATH"
source "$PREFIX/etc/conda/activate.d/activate_emscripten_emscripten-wasm32.sh"
export EM_CONFIG="/workspace/.jlite-test/em_config"

SRC=/workspace/.src/pytorch
BUILD=/workspace/.torch-build
STAGE=/workspace/.torch-stage
PYINC="/workspace/.mmroot/pkgs/https/prefix.dev/emscripten-forge-dev/emscripten-wasm32/python-3.13.1-h_c8de616_6_cp313/include/python3.13"
EH="-fexceptions"
LOG=/workspace/prototypes/emscripten-forge/torch-wasm/logs

echo "=== emcc version ==="; emcc --version | head -1

WHOLE=()
for l in libtorch_python.a libtorch.a libtorch_cpu.a; do
  [ -f "$BUILD/lib/$l" ] && WHOLE+=("$BUILD/lib/$l")
done
OTHER=()
while IFS= read -r a; do
  case "$a" in
    */libtorch_python.a|*/libtorch.a|*/libtorch_cpu.a) ;;
    *) OTHER+=("$a") ;;
  esac
done < <(find "$BUILD" -name '*.a' | sort -u)

echo "=== compile stub.c as C ==="
emcc -sSIDE_MODULE=1 -fexceptions -O2 -I"$PYINC" -c "$SRC/torch/csrc/stub.c" -o "$BUILD/stub.o"
NM="$PREFIX/opt/emsdk/upstream/bin/llvm-nm"
echo "stub.o initModule ref:"; $NM "$BUILD/stub.o" | grep -i initmodule

echo "=== compile cpuinfo emscripten init.c (defines cpuinfo_emscripten_init) ==="
CPUINFO=$SRC/third_party/cpuinfo
emcc -sSIDE_MODULE=1 -fexceptions -O2 \
  -DCPUINFO_LOG_LEVEL=2 -DCPUINFO_LOG_TO_STDIO=1 -I"$CPUINFO/include" -I"$CPUINFO/src" \
  -c "$CPUINFO/src/emscripten/init.c" -o "$BUILD/cpuinfo_emscripten_init.o"
echo "cpuinfo init symbol:"; $NM "$BUILD/cpuinfo_emscripten_init.o" | grep -i cpuinfo_emscripten_init

echo "=== link _C.so ==="
em++ -sSIDE_MODULE=1 -sWASM_BIGINT $EH -O2 -I"$PYINC" \
  "$BUILD/stub.o" "$BUILD/cpuinfo_emscripten_init.o" \
  -Wl,--whole-archive "${WHOLE[@]}" -Wl,--no-whole-archive \
  -Wl,--start-group "${OTHER[@]}" -Wl,--end-group \
  -o "$STAGE/_C.so" 2>&1 | tee "$LOG/15b-relink-_C.log"
echo "LINK_RC=${PIPESTATUS[0]}"
ls -la "$STAGE/_C.so"
echo "=== verify no mangled import remains ==="
strings -a "$STAGE/_C.so" | grep -c "_Z10initModulev" || true
$NM "$STAGE/_C.so" | grep -iw initmodule
