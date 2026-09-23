# Pyodide PyTorch prototypes

Pyodide builds packages from source for `wasm32-emscripten` via `pyodide-build`.
Recipes are `packages/<name>/meta.yaml`. A compiled package like `torch` cannot be
installed at runtime with `micropip` (that only accepts pure-Python `none-any`
wheels); it must have a recipe and be built into the distribution.

## What we attempt (in order of realism)

1. **`executorch/` recipe** — the realistic Python-importable target. ExecuTorch's
   Python runtime bindings (pybindings) over the WASM C++ runtime. This mirrors the
   emscripten-forge `executorch-cpp` work but produces an importable Python module in
   Pyodide. A cloud subagent iterates on `packages/executorch/meta.yaml`.

2. **`torch-probe/` recipe** — a deliberately minimal from-source `torch` recipe used
   as a *probe*: build it far enough to enumerate the exact blockers (dependency
   graph, `-pthread` ABI, sleef/protobuf/XNNPACK, binary size) with real logs, rather
   than relying on years-old "it can't be done" claims. `cffi` is now available in
   Pyodide (see the research doc), so at least that historical blocker is gone.

## How a recipe is built

```bash
python3 -m pip install pyodide-build
# emsdk version must match pyodide-build's expectation:
pyodide xbuildenv install            # or: pyodide config get emscripten_version
# build a single recipe from a checkout of pyodide/pyodide-recipes:
pyodide build-recipes torch --recipe-dir=packages
```

See `torch-probe/meta.yaml` for a starting scaffold and
`../../docs/research-why-no-pytorch-wasm.md` for the blocker analysis the probe is
meant to confirm/refute empirically.
