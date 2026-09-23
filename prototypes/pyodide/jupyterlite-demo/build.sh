#!/usr/bin/env bash
# Build a static JupyterLite site (Pyodide kernel) that ships the from-source
# wasm32-emscripten `torch` wheel and a notebook that trains a small MLP.
#
# Usage:
#   ./build.sh [path/to/torch-*.whl]
#
# If no wheel path is given, the newest wheel under ./dist is used.
# Output: ./_site  (serve with: python -m http.server -d _site 8000)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pyodide runtime pinned to match the wheel's ABI (pyemscripten_2024_0_wasm32).
PYODIDE_VERSION="${PYODIDE_VERSION:-0.27.8}"
PYODIDE_URL="https://github.com/pyodide/pyodide/releases/download/${PYODIDE_VERSION}/pyodide-${PYODIDE_VERSION}.tar.bz2"

WHEEL="${1:-}"
if [ -z "${WHEEL}" ]; then
    WHEEL="$(ls -t dist/torch-*.whl 2>/dev/null | head -1 || true)"
fi
if [ -z "${WHEEL}" ] || [ ! -f "${WHEEL}" ]; then
    echo "ERROR: no torch wheel found. Pass one explicitly or place it under ./dist/." >&2
    exit 1
fi
echo "==> Using torch wheel: ${WHEEL}"

if [ ! -d .jlite-venv ]; then
    python3 -m venv .jlite-venv
fi
# shellcheck disable=SC1091
source .jlite-venv/bin/activate
python -m pip install -q --upgrade pip
# Pinned to a jupyterlite-pyodide-kernel that targets the Pyodide 0.27 series.
python -m pip install -q \
    "jupyterlite-core==0.6.*" \
    "jupyterlite-pyodide-kernel==0.6.*" \
    jupyterlab_server jupyterlab

mkdir -p wheels
cp -f "${WHEEL}" wheels/

echo "==> Building JupyterLite site (Pyodide ${PYODIDE_VERSION})"
jupyter lite build \
    --contents content \
    --output-dir _site \
    --pyodide "${PYODIDE_URL}" \
    --piplite-wheels "wheels/$(basename "${WHEEL}")"

echo "==> Done. Serve locally with:"
echo "    python -m http.server -d '${HERE}/_site' 8000"
echo "    then open http://localhost:8000/lab/index.html"
