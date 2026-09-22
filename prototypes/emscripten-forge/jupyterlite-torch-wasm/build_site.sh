#!/usr/bin/env bash
# One-command build of the JupyterLite site that runs a PyTorch-style MLP training
# demo in a WebAssembly xeus-python kernel (emscripten-forge) using microtorch.
#
# Requirements (host, one-time):
#   python3 -m venv .venv-jlite && source .venv-jlite/bin/activate
#   pip install "jupyterlite-core==0.6.*" "jupyterlite-xeus==4.*" jupyter_server
#   # micromamba is required by jupyterlite-xeus to solve the emscripten env:
#   curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$VIRTUAL_ENV" bin/micromamba
#
# Usage:  ./build_site.sh [OUTPUT_DIR]     (default OUTPUT_DIR: ../../../docs)
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${1:-$(cd ../../../ && pwd)/docs}"
echo "==> Building JupyterLite site into: ${OUT_DIR}"

command -v micromamba >/dev/null 2>&1 || {
  echo "micromamba not found on PATH (needed by jupyterlite-xeus). See header." >&2
  exit 1
}
command -v jupyter >/dev/null 2>&1 || {
  echo "jupyter not found; activate the .venv-jlite venv first. See header." >&2
  exit 1
}

# Build into a scratch dir first, then sync into OUT_DIR. This keeps any non-site
# files already living in OUT_DIR (notably docs/research-why-no-pytorch-wasm.md)
# intact across rebuilds instead of nuking the whole directory.
SITE_TMP="$(mktemp -d)"
rm -rf .jupyterlite.doit.db _build
jupyter lite build \
  --XeusAddon.environment_file=environment.yml \
  --contents content \
  --output-dir "${SITE_TMP}"

mkdir -p "${OUT_DIR}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete --exclude 'research-why-no-pytorch-wasm.md' "${SITE_TMP}/" "${OUT_DIR}/"
else
  # Fallback: preserve the research doc manually.
  cp -f "${OUT_DIR}/research-why-no-pytorch-wasm.md" /tmp/_research_backup.md 2>/dev/null || true
  find "${OUT_DIR}" -mindepth 1 -maxdepth 1 ! -name 'research-why-no-pytorch-wasm.md' -exec rm -rf {} +
  cp -a "${SITE_TMP}/." "${OUT_DIR}/"
  cp -f /tmp/_research_backup.md "${OUT_DIR}/research-why-no-pytorch-wasm.md" 2>/dev/null || true
fi
rm -rf "${SITE_TMP}"

echo "==> Done. Site at ${OUT_DIR}"
echo "    Local preview:  python3 -m http.server -d '${OUT_DIR}' 8000"
echo "    Then open:      http://localhost:8000/lab/index.html?path=mlp_training_demo.ipynb"
