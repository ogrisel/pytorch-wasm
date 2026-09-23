#!/usr/bin/env bash
# OPTION (a) build driver: recompile the WHOLE torch tree with emsdk 3.1.73
# (newer clang) instead of the Pyodide-0.27.8-pinned 3.1.58, to test whether a
# newer clang emits CORRECT MEMORY_ADDR data relocations for the large module
# (blocker #11/#12). The compiled objects are still linked as a single
# torch/_C.*.so SIDE_MODULE and loaded under the Pyodide 0.27.8 runtime.
#
# Mechanism for using a non-pinned emsdk with pyodide-build 0.39.0:
#   SKIP_EMSCRIPTEN_VERSION_CHECK=1 makes pyodide_build.build_env.ensure_emscripten
#   return immediately (see build_env.py), so pyodide-build uses whatever `emcc`
#   is on PATH (here: emsdk 3.1.73) without its 3.1.58 version gate.
#
# ccache: EM_COMPILER_WRAPPER=ccache makes emcc run the underlying clang through
#   ccache, so every clang compile is cached under $CCACHE_DIR (persistent disk).
set -euxo pipefail

ROOT=/workspace
LOGDIR="${LOGDIR:-$ROOT/prototypes/pyodide/logs}"
RECIPES="$ROOT/build_torch/recipes"

# 1) Toolchain: emsdk 3.1.73 on PATH.
source "$ROOT/.emsdk/emsdk_env.sh"
emcc --version | head -1

# 2) ccache wraps clang under emcc.
export CCACHE_DIR="${CCACHE_DIR:-$ROOT/.ccache}"
export EM_COMPILER_WRAPPER=ccache
ccache -s | head -3 || true

# 3) pyodide-build: use ambient (3.1.73) emcc, skip the 3.1.58 version gate.
export SKIP_EMSCRIPTEN_VERSION_CHECK=1
export MAX_JOBS="${MAX_JOBS:-4}"

# 4) Build the recipe in place under $RECIPES/packages/torch/build.
source "$ROOT/.venv/bin/activate"
cd "$RECIPES"
pyodide build-recipes-no-deps torch --recipe-dir=packages
