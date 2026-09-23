#!/usr/bin/env bash
# SINGLE-MODULE link experiment for Pyodide torch (blocker #11 avenue).
#
# Replicates the emscripten-forge sibling's winning strategy: instead of shipping
# libc10.so + libtorch_cpu.so + libtorch.so + libtorch_python.so + torch/_C.so as
# SEPARATE side modules (which forced cross-.so GOT.func / MEMORY_ADDR relocations,
# blocker #11), link ONE combined `torch/_C.*.so` SIDE_MODULE that contains every
# torch object (c10 + torch_cpu + torch + torch_python) plus the wasm deps
# (protobuf/onnx/onnx_proto/cpuinfo), so ALL cross-module C++ symbols resolve
# INTERNALLY within one module and there are no cross-.so relocations.
#
# Inputs come from a COMPLETED (shared) pyodide `build-recipes` run whose CMake
# build tree persists under $BUILD_DIR/build. Loose per-target .o files (under
# CMakeFiles/<target>.dir/) are always fully linked in, which is equivalent to
# --whole-archive for the torch libraries (preserves ATen op-registration static
# initializers). libonnx.a is whole-archived (schema registration) as in the
# original torch_cpu link; the other deps resolve normally.
#
# Output: $STAGE/_C.cpython-312-wasm32-emscripten.so
set -euxo pipefail

BUILD_DIR="${BUILD_DIR:-/workspace/build_torch/packages/torch/build/torch-2.8.0}"
CMBUILD="$BUILD_DIR/build"
XBENV="/home/ubuntu/.cache/pyodide-build/.pyodide-xbuildenv-b2b15c7d3f61/0.27.8"
# EMSDK toolchain: default to the Pyodide-0.27.8 xbuildenv emsdk (3.1.58), but
# allow overriding to a candidate revision (e.g. /workspace/.emsdk activated at
# 3.1.73) to LINK objects that were RECOMPILED with that same candidate.
EMSDK="${EMSDK:-$XBENV/emsdk}"
export PATH="$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$PATH"
PYINC="$XBENV/xbuildenv/pyodide-root/cpython/installs/python-3.12.7/include/python3.12"
STAGE="${STAGE:-/workspace/build_torch/single_module_stage}"
LOGDIR="${LOGDIR:-/workspace/prototypes/pyodide/logs}"
mkdir -p "$STAGE"

echo "=== emcc version ==="; emcc --version | head -1
NM="$(command -v llvm-nm || echo "$EMSDK/upstream/bin/llvm-nm")"

# 1) stub.c compiled as C -> unmangled PyInit__C -> extern "C" initModule().
echo "=== compile stub.c (C) ==="
emcc -c -O2 -g0 -fPIC -I"$PYINC" "$BUILD_DIR/torch/csrc/stub.c" -o "$STAGE/stub.o"
"$NM" "$STAGE/stub.o" | grep -i "initmodule\|PyInit__C" || true

# 2) cpuinfo_emscripten_init (referenced by torch_cpu, an unresolved GOT.func).
CPUINFO="$BUILD_DIR/third_party/cpuinfo"
echo "=== compile cpuinfo emscripten init.c ==="
emcc -c -O2 -g0 -fPIC \
  -DCPUINFO_LOG_LEVEL=2 -DCPUINFO_LOG_TO_STDIO=1 \
  -I"$CPUINFO/include" -I"$CPUINFO/src" \
  "$CPUINFO/src/emscripten/init.c" -o "$STAGE/cpuinfo_emscripten_init.o"
"$NM" "$STAGE/cpuinfo_emscripten_init.o" | grep -i cpuinfo_emscripten_init || true

# 3) Gather every torch object (loose .o == whole-archive equivalent).
OBJRSP="$STAGE/objects.rsp"
: > "$OBJRSP"
for tgt in c10.dir torch_cpu.dir torch.dir torch_python.dir; do
  find "$CMBUILD" -path "*/${tgt}/*" -name '*.o' >> "$OBJRSP"
done
NOBJ=$(wc -l < "$OBJRSP")
echo "=== gathered $NOBJ torch objects ==="
test "$NOBJ" -gt 1000

# 4) Third-party static-lib deps (mirror the original torch_cpu link line).
ONNX_A="$CMBUILD/lib/libonnx.a"
DEP_AS=()
for a in libcpuinfo.a libonnx_proto.a libprotobuf.a; do
  [ -f "$CMBUILD/lib/$a" ] && DEP_AS+=("$CMBUILD/lib/$a")
done

# 5) Export just the CPython init symbol (single self-contained extension).
echo '["_PyInit__C"]' > "$STAGE/exports.json"

echo "=== link single-module torch/_C.so ==="
OUT="$STAGE/_C.cpython-312-wasm32-emscripten.so"
em++ -O2 -g0 -sWASM_BIGINT -sSIDE_MODULE=2 \
  -sEXPORTED_FUNCTIONS=@"$STAGE/exports.json" \
  "$STAGE/stub.o" "$STAGE/cpuinfo_emscripten_init.o" \
  @"$OBJRSP" \
  -Wl,--whole-archive "$ONNX_A" -Wl,--no-whole-archive \
  -Wl,--start-group "${DEP_AS[@]}" -Wl,--end-group \
  -o "$OUT" 2>&1 | tee "$LOGDIR/34-single-module-link.log"
echo "LINK_RC=${PIPESTATUS[0]}"
ls -la "$OUT"
echo "=== exported PyInit__C? ==="
"$NM" "$OUT" 2>/dev/null | grep -i "PyInit__C" | head || true
