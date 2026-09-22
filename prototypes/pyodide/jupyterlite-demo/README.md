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

## Notes / limitations

The wheel is a reduced build: single-threaded, no XNNPACK / MKLDNN / FBGEMM /
distributed / CUDA / quantization. Eager mode, autograd, `torch.nn`, and the
`torch.optim` optimizers used here work. See `../RESULTS.md` for the full list of
disabled features, the wheel size, and the build blockers that were fixed.
