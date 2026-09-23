#!/usr/bin/env bash
# Install a pinned Emscripten SDK into ./.emsdk and print the activation command.
# Idempotent: re-running only re-activates the requested version.
set -euo pipefail

EMSDK_VERSION="${EMSDK_VERSION:-3.1.58}"
EMSDK_DIR="${EMSDK_DIR:-$(pwd)/.emsdk}"

if [ ! -d "${EMSDK_DIR}" ]; then
  git clone https://github.com/emscripten-core/emsdk.git "${EMSDK_DIR}"
fi

cd "${EMSDK_DIR}"
git pull --ff-only || true
./emsdk install "${EMSDK_VERSION}"
./emsdk activate "${EMSDK_VERSION}"

echo
echo "Emscripten ${EMSDK_VERSION} installed at ${EMSDK_DIR}"
echo "Activate with: source ${EMSDK_DIR}/emsdk_env.sh"
