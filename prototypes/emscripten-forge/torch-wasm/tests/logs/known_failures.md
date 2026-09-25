# Failure / limitation ledger — upstream PyTorch tests vs. the wasm32 torch build

This ledger is deliberately kept **separate** from the wasm-inapplicable skip
manifest (`../skip_manifest.json` / `skip_classification.json`). The rule is:

* **(A) Skipped-inapplicable** — inapplicable *by environment* (no GPU, no
  distributed, single-threaded, disabled backends, no host compiler, no numpy
  bridge, no OS FS/mmap/signals, slow/large-memory). These are handled by the
  skip manifest and are **not** failures. 766 tests across the 6 core files
  (see `skip_classification.json`).
* **(B) Genuine reduced-build failures / xfails** — tests that are *applicable*
  to a CPU-only eager runtime but fail specifically because of a bug or
  limitation in this reduced build. **These must be recorded here, never
  skipped as "inapplicable".**
* **(C) Harness / isolation artifacts** — failures caused by the *runner*, not
  by torch. Recorded here only so they are not mistaken for (A) or (B).

---

## (B) Genuine reduced-build failures / xfails

**Status: not yet observable on this machine.** Populating this section
requires executing the tests against the *actual* wasm `torch/_C.so` module. On
the fresh cloud VM used for this run that module is not present and could not be
rebuilt within the time budget — the emscripten-forge build reproduction gets
through toolchain setup + source + host `protoc` + full CMake configure, then
stops early in compilation at the ONNX protobuf codegen step (the
cross-compiled wasm `protoc.js` is invoked as a host tool → `Exec format error`,
exit 126), and even past that, compiling `libtorch_cpu` (~368 MB of objects) on
4 CPUs far exceeds the budget. See `RESULTS.md` → "Running the test suite" for
the exact evidence (`.build_attempt_logs/`).

Known runtime-level limitations already documented for the reduced build that
would turn into entries here when executed under wasm (predicted, not yet
measured):

| Area | Expected behavior under wasm | Manifest category it is instead *classified* under, if name-detectable |
| --- | --- | --- |
| numpy bridge (`from_numpy`/`.numpy()`) | raises (USE_NUMPY=0) | `numpy_bridge` (A) when the nodeid names numpy; otherwise would surface here |
| multi-threaded intra-op parallelism | serialized (single-threaded) | `threading_threadpool` (A) |
| `torch.compile` / inductor | no host compiler at runtime | `cpp_extension_jit_compile` (A) |

These are listed for transparency; they are predictions from the build config,
**not** executed results.

---

## (C) Harness / isolation artifacts (host reference-torch proxy)

Observed while validating the runner against host CPU torch 2.8.0 (a full,
non-reduced build). These are **not** torch failures and **not**
wasm-inapplicable:

### `test_autograd.py`: 63 failures, all logging-instrumented

* Tests: `TestAutogradLogging::test_logging`, `TestAutograd::test_function`,
  `TestAutograd::test_multi_grad_all_hooks`, and ~60
  `TestAutogradFunctional::*_logging_tensor` cases.
* Root cause: running the whole file in a **single pytest process** accumulates
  global `pt2` logging handlers; the upstream logging harness then asserts
  `assertLessEqual(len(handlers), 2)` and trips with
  `AssertionError: 5 not less than or equal to 2 : All pt2 loggers should only
  have at most two handlers`.
* Proof it is an isolation artifact, not a real failure: the same test passes
  when run alone —
  `pytest test_autograd.py::TestAutogradFunctional::test_vjp_scalar_logging_tensor`
  → `1 passed`.
* Applies equally to any single-process runner (host or wasm). Upstream avoids
  it by running each file via its own `run_tests()`/unittest process with fresh
  logging state. Mitigation for a real wasm run: execute logging-instrumented
  classes in isolation, or reset logging handlers between tests. Not fixed here
  because it does not affect the correctness signal for the reduced build.

All other executed files reported **0 failures** on the host baseline
(`results.json`) and **0 failures** on the simulated-wasm applicable subset
(`results-wasm-sim.json`).
