#!/usr/bin/env bash
# OPTION (a) build driver: full RECOMPILE of the whole reduced torch tree with a
# CANDIDATE emsdk/clang revision (default 3.1.73), to test whether a newer clang
# emits CORRECT MEMORY_ADDR data relocations for the large module (blocker
# #11/#12) while the resulting single torch/_C.*.so still LOADS under the Pyodide
# 0.27.8 runtime ABI.
#
# Usage:
#   EMSDK_VERSION=3.1.73 bash build_torch_candidate.sh
#
# Mechanism for using a non-pinned emsdk with pyodide-build 0.39.0:
#   SKIP_EMSCRIPTEN_VERSION_CHECK=1 makes pyodide_build.build_env.ensure_emscripten
#   return immediately, so pyodide-build uses whatever `emcc` is on PATH (here the
#   requested candidate) without its 3.1.58 version gate.
#
# ccache: EM_COMPILER_WRAPPER=<abs path to ccache> makes emcc run the underlying
#   clang through ccache (a full path is REQUIRED — emcc execv()s it directly and
#   does NOT do a PATH lookup), so every clang compile is cached under $CCACHE_DIR
#   (persistent disk). We also pass CMAKE_{C,CXX}_COMPILER_LAUNCHER=ccache as a
#   belt-and-braces measure for any compile that bypasses emcc's wrapper hook.
set -euxo pipefail

ROOT=/workspace
EMSDK_VERSION="${EMSDK_VERSION:-3.1.73}"
LOGDIR="${LOGDIR:-$ROOT/prototypes/pyodide/logs}"
RECIPES="$ROOT/build_torch/recipes"
mkdir -p "$LOGDIR" "$RECIPES/packages/torch"

# 1) Toolchain: requested emsdk revision on PATH.
cd "$ROOT/.emsdk"
./emsdk install "$EMSDK_VERSION"
./emsdk activate "$EMSDK_VERSION"
# shellcheck disable=SC1091
source "$ROOT/.emsdk/emsdk_env.sh"
emcc --version | head -1

# 2) ccache wraps clang under emcc. EM_COMPILER_WRAPPER MUST be an absolute path.
export CCACHE_DIR="${CCACHE_DIR:-$ROOT/.ccache}"
CCACHE_BIN="$(command -v ccache)"
export EM_COMPILER_WRAPPER="$CCACHE_BIN"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
ccache -o max_size=30G || true
ccache -s | head -3 || true

# 3) pyodide-build: use ambient (candidate) emcc, skip the 3.1.58 version gate.
export SKIP_EMSCRIPTEN_VERSION_CHECK=1
export MAX_JOBS="${MAX_JOBS:-4}"

# 4) Recipe in place under $RECIPES/packages/torch.
cp "$ROOT/prototypes/pyodide/torch-probe/meta.yaml" "$RECIPES/packages/torch/meta.yaml"

# 5) Build the recipe.
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
cd "$RECIPES"
pyodide build-recipes-no-deps torch --recipe-dir=packages --force-rebuild
