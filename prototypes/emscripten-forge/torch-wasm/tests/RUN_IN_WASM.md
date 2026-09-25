# Running the upstream PyTorch test suite against the wasm32 torch build

Two runners share one `conftest.py` + `skip_manifest.json`, so the
wasm-inapplicable skip policy is identical in both:

1. **Host reference runner** (`run_tests.py`) — validates the harness + the
   skip manifest against a full host CPU torch. Used to produce the committed
   `logs/results.json` (host baseline) and `logs/results-wasm-sim.json`
   (simulated-wasm applicable subset). This does **not** run torch in wasm.
2. **In-wasm runner** (`run_pytest_wasm.js` + `wasm_pytest_driver.py`) — runs
   the same files inside the xeus-python wasm32 kernel, where
   `sys.platform == 'emscripten'` so the skips activate automatically.

## Host reference runner (reproducible now)

```bash
# a python env with torch==2.8.0 (cpu), pytest, expecttest, hypothesis, numpy
PY=/path/to/python

# real host baseline (proves the upstream tests execute; only host-absent
# capabilities such as GPU are skipped):
$PY run_tests.py --python "$PY" \
    --files test_type_promotion.py test_optim.py test_torch.py

# simulated-wasm classification+execution (skip everything the reduced wasm
# runtime cannot run, execute the applicable subset on the reference torch):
TORCH_WASM_SIMULATE=1 $PY run_tests.py --python "$PY" \
    --files test_type_promotion.py test_optim.py test_torch.py test_nn.py \
    --out logs/results-wasm-sim.json

# collect-only classification for huge files (safe, no OOM), then consolidate:
TORCH_WASM_SIMULATE=1 TORCH_WASM_SKIPREPORT=logs/skipreport-test_ops.py.json \
    $PY -m pytest upstream/test_ops.py --collect-only -q
$PY summarize.py            # -> logs/skip_classification.json
```

## In-wasm runner (requires the built wasm torch module)

Prerequisites:

1. The wasm `torch` conda package (the `torch/_C.so` side module) built and in
   the local channel — see `../RESULTS.md` → "How to reproduce".
2. Add test deps to the JupyterLite env next to `torch`:

   ```yaml
   # jupyterlite/environment.yml
   dependencies: [xeus-python, numpy, sympy, torch, pytest, hypothesis]
   ```
   (`expecttest` is pure-python; ship it as a wheel or vendor it if not on the
   channel.)
3. Copy this `tests/` tree into the JupyterLite content so it lands at
   `/drive/tests/` in the kernel FS, and add a `pytest_wasm.ipynb` whose one
   cell is:

   ```python
   import runpy, sys
   sys.argv = ['', 'test_type_promotion.py']   # one file at a time (memory)
   runpy.run_path('tests/wasm_pytest_driver.py', run_name='__main__')
   ```
4. Build + serve the site (COOP/COEP headers, see `../jupyterlite/test/serve.py`)
   and drive it:

   ```bash
   node run_pytest_wasm.js       # greps for WASM_PYTEST_SUMMARY, writes logs/
   ```

`wasm_pytest_driver.py` prints one `WASM_PYTEST_FILE <name> {json}` line per
file and a final `WASM_PYTEST_SUMMARY {json}` the harness captures.

> Status: the in-wasm runner is committed but was **not executed** in the cloud
> run that produced the current results, because the wasm `torch/_C.so` could
> not be rebuilt within the time budget on that VM (see `../RESULTS.md` →
> "Running the test suite"). It is the intended path to real wasm numbers once
> the module is packaged.
