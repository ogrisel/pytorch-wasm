#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for pytorch-wasm.
#
# Provisions the two WebAssembly toolchains the prototypes use:
#   1. Emscripten SDK 3.1.58 in ./.emsdk (for the self-contained tensor-kernel
#      C++ -> wasm32 PoC run under Node.js).
#   2. A Python venv with pyodide-build 0.39.0 + its cross-build environment
#      (emsdk 3.1.58 + target CPython) for building wasm32-emscripten wheels.
#
# Safe to run repeatedly: existing venv, emsdk, and xbuildenv installs are reused.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Pinned to match prototypes/pyodide/logs/00-versions-summary.txt
PYODIDE_BUILD_VERSION="0.39.0"

echo "==> Ensuring required system packages"
if ! dpkg -s python3-venv >/dev/null 2>&1 \
    || ! dpkg -s python3-dev >/dev/null 2>&1 \
    || ! dpkg -s build-essential >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-venv python3-dev build-essential
fi

echo "==> Setting up Python virtual environment (.venv)"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "pyodide-build==${PYODIDE_BUILD_VERSION}"

echo "==> Installing Pyodide cross-build environment (emsdk + target CPython)"
# xbuildenv version is pinned by pyodide-build; only install if not present yet.
if ! pyodide config get emscripten_version >/dev/null 2>&1; then
    pyodide xbuildenv install
fi

echo "==> Installing standalone Emscripten SDK for the tensor-kernel PoC"
./scripts/install-emsdk.sh

echo "==> Toolchain versions"
python -c "import sys; print('host python', sys.version.split()[0])"
pyodide --version
echo "pyodide emscripten_version: $(pyodide config get emscripten_version)"
# shellcheck disable=SC1091
source .emsdk/emsdk_env.sh >/dev/null 2>&1
emcc --version | head -1

echo "==> Install complete"
