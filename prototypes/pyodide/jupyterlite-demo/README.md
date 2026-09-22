# JupyterLite demo: train a PyTorch MLP in the browser (wasm32-emscripten)

This is a static [JupyterLite](https://jupyterlite.readthedocs.io/) site with a
Pyodide kernel. It ships the from-source **CPU-only `torch` wheel** built for
`wasm32-emscripten` (see `../torch-probe/meta.yaml`) and a notebook
(`content/torch_mlp_demo.ipynb`) that trains a small MLP (`nn.Linear` + `ReLU`,
`CrossEntropyLoss`, `SGD`) on a synthetic 2D dataset, showing the loss decreasing.

## Build it

```bash
# From this directory, with a built wheel in ./dist/ (or pass a path):
./build.sh /path/to/torch-2.8.0-cp312-cp312-pyodide_2024_0_wasm32.whl
```

`build.sh` pins the Pyodide runtime to the version whose ABI matches the wheel
(`PYODIDE_VERSION`, default `0.27.8`) and wires the wheel into the site's piplite
index so `await piplite.install("torch")` resolves it locally.

## Run it locally

```bash
python -m http.server -d _site 8000
# open http://localhost:8000/lab/index.html and run content/torch_mlp_demo.ipynb
```

## Headless Node harness (CI-style load/train check)

`verify_wheel_node.mjs` loads the same Pyodide 0.27.8 runtime in Node, unpacks the
wheel, loads the `.so` side modules in dependency order, and (once loading succeeds)
trains the MLP end-to-end. The Pyodide runtime binary (`pyodide.asm.wasm`, ~10 MB) is
**not committed**, so install it first:

```bash
npm install            # fetches pyodide@0.27.8 (provides pyodide.asm.wasm)
node verify_wheel_node.mjs dist_pyodide/torch-2.8.0a0+gitunknown-cp312-cp312-pyemscripten_2024_0_wasm32.whl
```

On the **current** wheel this reproduces the load up to blocker #11 (see below).

### Reproducing / inspecting blocker #11

`blocker11_instrument.mjs` patches `node_modules/pyodide/pyodide.asm.js` to record
`GOT.mem`/`GOT.func` symbol resolution, capture the C++ throw stack, and print the raw
`const char*` arguments (with a memory window) handed to `c10::detail::torchInternalAssertFail`.
`blocker11_diagnose.mjs` drives it:

```bash
npm install
node blocker11_instrument.mjs           # applies instrumentation (writes a .bak)
node blocker11_diagnose.mjs dist_pyodide/torch-2.8.0a0+gitunknown-cp312-cp312-pyemscripten_2024_0_wasm32.whl
cp node_modules/pyodide/pyodide.asm.js.bak node_modules/pyodide/pyodide.asm.js   # restore
```

The captured output is committed at `../logs/30-blocker11-assert-args.log`.

## Notes / limitations

The wheel is a reduced build: single-threaded, no XNNPACK / MKLDNN / FBGEMM /
distributed / CUDA / quantization. Eager mode, autograd, `torch.nn`, and the
`torch.optim` optimizers used here are intended to work **once blocker #11 is
resolved**; on the current wheel `libtorch_cpu.so` aborts during its static
initializers (see `../RESULTS.md` §blocker #11 for the exact throw site, the exact
unresolved symbols, and direct evidence of mis-offset rodata pointers).
