# AGENTS.md

Guidance for AI agents working on **pytorch-wasm** — an agent-driven effort to run the
real PyTorch codebase in the browser via WebAssembly.

## Project overview

Goal: build **real upstream PyTorch** for `wasm32-emscripten` and run it client-side in a
JupyterLite notebook (fit/predict real models), and honestly document the blockers. There
are two independent packaging tracks, each on its own branch/PR:

- **emscripten-forge track** — branch `cursor/emscripten-forge-torch-wasm-6b2e` (PR #3).
  WORKING: real `torch` 2.8.0 imports, trains an MLP, and runs a real third-party model
  (TabICL: `fit`+`predict`, ~96.7% acc on a small set) in-browser via the `xeus-python`
  kernel (`jupyterlite-xeus`). Uses a fully-consistent **emscripten 3.1.73 / CPython
  3.13.1** toolchain (compile + link + runtime), which is why it succeeds.
- **pyodide track** — branch `cursor/pyodide-torch-wasm-447a` (PR #4). Builds+links real
  torch from source for the **Pyodide 0.27.8** runtime (emscripten 3.1.58 / CPython 3.12),
  but is blocked: a data-relocation defect in Pyodide 0.27.8's load-time
  `__wasm_apply_data_relocs` corrupts rodata at `libtorch_cpu` scale, so `import torch`
  aborts. Root-caused to the **runtime** (not the compiler/linker — a full 3.1.73 recompile
  reproduces it); fix requires splitting `libtorch_cpu` or bumping the Pyodide runtime.

`main` is empty. The Cloud Agent environment config originated on
`cursor/setup-pytorch-wasm-env-fde6` and is merged into both track branches. Background
analysis: `docs/research-why-no-pytorch-wasm.md`. Each track's `.../RESULTS.md` is the
detailed, honest write-up and blocker ledger — read it before iterating.

## Environment

The Cloud Agent environment is repo-managed via `.cursor/environment.json` →
`.cursor/install.sh` (idempotent). It installs emsdk, `pyodide-build` + its xbuildenv, Node,
`ninja-build`, and fixes the GNU compiler alternatives (the default image points `c++` at
clang, which fails native links with `cannot find -lstdc++`).

## Building & running — emscripten-forge track (branch `...-6b2e`)

Toolchain: **emscripten 3.1.73 / cross-python 3.13.1** — must match the published
`xeus-python` side-module ABI. Scripts live under `prototypes/emscripten-forge/torch-wasm/`.

Build real torch → wasm and package for JupyterLite:
- `build_iter.sh` — compile+link the reduced torch into a single `torch/_C.so` SIDE_MODULE
  (`--whole-archive` of `libtorch_python`+`libtorch`+`libtorch_cpu`+deps; ~143 MB,
  git-ignored). The single-module link avoids the cross-`.so` relocation wall.
- `assemble_payload.py` / `stage_payload.py` — Python payload + Emscripten runtime guards.
- `make_conda_pkg.py` + `make_pyodide_http_stub.py` — build the emscripten-wasm32 conda
  package into a local channel referenced by `jupyterlite/environment.yml`.
- `build_site.sh` — `jupyter lite build` producing the static site.
- Prefer the `rattler-build` recipe flow (`recipe/`): it supplies a **host `protoc`** from
  conda-forge, avoiding the bare-VM failure where the cross-built wasm `protoc.js` is run as
  a host tool (`Exec format error`, exit 126).

Verify in a real headless browser:
- MLP demo: `jupyterlite/content/torch_mlp_demo.ipynb` via `jupyterlite/test/run_torch.js`.
- TabICL demo: `jupyterlite/content/tabicl_demo.ipynb` via `jupyterlite/test/run_retry.js`.

Test suite: `tests/` has upstream PyTorch core test files, a capability-gated
wasm-inapplicable skip manifest (`tests/conftest.py`, `tests/skip_manifest.json`), and host
+ in-wasm runners. The true in-wasm run needs a freshly-built module; see
`tests/RUN_IN_WASM.md`. Keep the skip manifest (environment-inapplicable) strictly separate
from genuine reduced-build failures.

## Building & running — pyodide track (branch `...-447a`)

Toolchain pinned to Pyodide 0.27.8 (emscripten 3.1.58). Under `prototypes/pyodide/`:
recipe `torch-probe/meta.yaml`; single-module link `torch-probe/link_single_module.sh`;
build drivers `torch-probe/build_torch_candidate.sh` / `build_torch_3173.sh`; Node harnesses
`jupyterlite-demo/verify_wheel_node.mjs`, `verify_single_module_node.mjs`,
`diag_single_module.mjs`. Blocker analysis in `prototypes/pyodide/RESULTS.md` (#10 solved;
#11/#12/#13 = the runtime relocation wall). Do not re-try "newer clang, keep 0.27.8" — it is
proven infeasible (the defect is applied by the 0.27.8 runtime, not codegen).

## Cursor Cloud specific instructions

- **Long builds on ~4 vCPU — use ccache.** A full torch build is ~1500 objects (~25 min);
  set up ccache and reuse it. Gotcha: `EM_COMPILER_WRAPPER` must be an **absolute path**
  (`emcc` `execv`s it with no PATH lookup — a bare `ccache` gives `FileNotFoundError`).
  Also set `-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`, a
  persistent `CCACHE_DIR` (e.g. `/workspace/.ccache`), and `compiler_check=content`. Warm
  rebuilds hit ~97% (~3 min).
- **Large artifacts are git-ignored and do NOT persist across fresh VMs:** the ~143 MB
  `torch/_C.so`, the ~110 MB TabICL checkpoint, `.emsdk/`, `.venv/`, `.ccache/`, and
  Pyodide's `pyodide.asm.wasm`. On a fresh VM you must rebuild the module first; restore the
  Pyodide Node runtime with `npm install pyodide@0.27.8` before running the harnesses.
  Nothing over GitHub's 100 MB limit can be committed — if a one-click hosted demo is needed,
  publish the built site/package as a GitHub Release asset or via Git LFS.
- **JupyterLite kernel-boot race:** `xeus-python` intermittently aborts boot with
  `XKernel is already registered` (seen even in numpy-only envs — not package-specific). Use
  a retry harness that opens fresh browser contexts until a clean boot:
  `jupyterlite/test/run_retry.js`.
- **scikit-learn in xeus:** the full compiled emscripten-forge sklearn (~69 `.so`) crashes
  kernel boot. Ship only the import closure the target package actually needs (for TabICL:
  43 `.so`). See `make_sklearn_pure_pkg.py`.
- **numpy bridge is disabled** in the reduced build (`USE_NUMPY=0`): `torch.from_numpy` /
  `Tensor.numpy` are absent. Shim them via `.tolist()` (see `tabicl_wasm_shim.py`) or pass
  torch tensors directly.
- **HuggingFace checkpoints:** don't rely on runtime download in-browser (CORS/fetch).
  Bundle at build time (`bundle_checkpoint.py`) and pass a local `model_path` with
  `allow_auto_download=False`.
- `microtorch` is a labelled throwaway placeholder, not a deliverable; do not re-add the
  removed stale `docs/` JupyterLite site.

## Working conventions

- Keep changes on your own track branch; do not touch the sibling track or the shared
  prototype branch. Commit + push; PRs are managed by the orchestrator.
- Report honestly: never fabricate a working demo. A precise negative result (exact failing
  step/symbol + logs) is a valid, valuable outcome — record it in the track's `RESULTS.md`.
