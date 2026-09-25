#!/usr/bin/env bash
# Assemble the local conda channel (torch + tabicl-wasm + pyodide-http stub),
# bundle the TabICL checkpoint, regenerate the demo notebooks, and build the
# JupyterLite site. Run AFTER the wasm torch rebuild has produced
# /workspace/.torch-stage/_C.so.
set -euxo pipefail

PROTO=/workspace/prototypes/emscripten-forge/torch-wasm
JLITE="$PROTO/jupyterlite"
STAGE=/workspace/.torch-stage
export MAMBA_ROOT_PREFIX=/workspace/.mmroot
MM=/workspace/bin/micromamba
PY="$MM run -r $MAMBA_ROOT_PREFIX -n jlite python"
HOSTPY=/workspace/.mmroot/envs/hosttabicl/bin/python
HOSTPIP=/workspace/.mmroot/envs/hosttabicl/bin/pip

[ -f "$STAGE/_C.so" ] || { echo "MISSING $STAGE/_C.so — run the rebuild first"; exit 2; }

echo "=== [1] stage importable torch payload from source + built _C.so ==="
$PY "$PROTO/stage_payload.py"
$PY "$PROTO/assemble_payload.py"

echo "=== [2] local channel: torch (writes fresh repodata) ==="
$PY "$PROTO/make_conda_pkg.py"
echo "=== [3] local channel: pyodide-http noarch stub ==="
$PY "$PROTO/make_pyodide_http_stub.py"
echo "=== [4] local channel: tabicl-wasm pure-python bundle (merges repodata) ==="
$HOSTPY "$PROTO/make_tabicl_pkg.py" --pip "$HOSTPIP"

echo "=== [4b] local channel: PURE-PYTHON scikit-learn override ==="
# The compiled emscripten-forge scikit-learn wasm .so extensions crash the
# xeus-python kernel at boot ("XKernel is already registered"). Fetch that
# package for emscripten-wasm32, strip every .so, and republish it pure-python
# with a higher build number (TabICL only needs sklearn's pure-python surface).
"$MM" create -y -r "$MAMBA_ROOT_PREFIX" -n sklfetch --platform emscripten-wasm32 \
  -c https://prefix.dev/emscripten-forge-dev -c conda-forge \
  "scikit-learn=1.8.0" 2>&1 | tail -3
# Keep exactly the compiled extensions in TabICLClassifier's import closure
# (utils / __check_build / _cyutility / _loss / decomposition / linear_model /
# metrics / neighbors / preprocessing / svm). This 43-.so subset boots the
# kernel cleanly; shipping the FULL compiled scikit-learn (~69 .so, incl.
# ensemble/tree/cluster/manifold/mixture/feature_*) instead triggers the
# "XKernel is already registered" boot crash.
$PY "$PROTO/make_sklearn_pure_pkg.py" \
  --src-site "$MAMBA_ROOT_PREFIX/envs/sklfetch/lib/python3.13/site-packages" \
  --keep-so "utils,__check_build,_cyutility,_isotonic,_loss,decomposition,linear_model,metrics,neighbors,preprocessing,svm"

echo "=== [5] bundle TabICL checkpoint into site content (git-ignored) ==="
$HOSTPY "$PROTO/bundle_checkpoint.py" --python "$HOSTPY"

echo "=== [6] regenerate demo notebooks ==="
$PY "$JLITE/make_notebook.py" || true
$PY "$JLITE/make_tabicl_notebook.py"

echo "=== [7] jupyter lite build ==="
cd "$JLITE"
rm -rf _output .jupyterlite.doit.db
PATH="/workspace/bin:$PATH" $MM run -r "$MAMBA_ROOT_PREFIX" -n jlite \
  jupyter lite build --XeusAddon.environment_file=environment.yml \
  --contents content --output-dir _output 2>&1 | tee "$PROTO/logs/40-lite-build.log"
echo "SITE_BUILD_DONE"
ls -la _output/ | head
