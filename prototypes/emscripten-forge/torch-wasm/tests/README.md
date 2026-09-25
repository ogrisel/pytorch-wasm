# Upstream PyTorch test suite vs. the wasm32-emscripten torch build

Exercises upstream PyTorch 2.8.0's own test files against the reduced,
CPU-only, single-threaded `wasm32-emscripten` torch build packaged by this
prototype, and iteratively skips only the tests that are **fundamentally
inapplicable** to the wasm runtime.

## Layout

| Path | What |
| --- | --- |
| `upstream/` | Verbatim upstream test files at tag `v2.8.0` (`test_torch.py`, `test_autograd.py`, `test_nn.py`, `test_optim.py`, `test_type_promotion.py`, `test_ops.py`) plus the `optim/` and `autograd/` sub-package modules they import. |
| `skip_manifest.json` | Machine-readable manifest of wasm-inapplicable test categories (GPU, distributed, subprocess, threads, disabled backends, cpp-extension/compile, numpy-bridge, OS-FS, slow/large-memory, profiler), each with rationale + a `nodeid` regex + a capability gate. |
| `conftest.py` | Applies the manifest at collection time via runtime capability probes. Correct under both host torch and the wasm kernel; `TORCH_WASM_SIMULATE=1` forces the wasm capability values on a host interpreter for classification. |
| `run_tests.py` | Host reference-torch runner; per-file passed/skipped/failed/error → `logs/results*.json`. |
| `run_pytest_wasm.js` + `wasm_pytest_driver.py` | In-wasm runner (xeus-python / JupyterLite). See `RUN_IN_WASM.md`. |
| `summarize.py` | Consolidates `logs/skipreport-*.json` → `logs/skip_classification.json`. |
| `logs/` | Results, per-file skip reports, the consolidated ledger, `known_failures.md`, and the JUnit XML. |

## Ground rules (honesty)

* **Skip only inapplicable-by-environment** tests (the manifest). These are not
  failures.
* **Genuine reduced-build failures** go in `logs/known_failures.md`, never into
  the skip manifest.
* **Harness/isolation artifacts** are also recorded in `known_failures.md` so
  they are not confused with either.

See `RUN_IN_WASM.md` to reproduce, and `../RESULTS.md` → "Running the test
suite" for the summary and the honest status of the true in-wasm run.
