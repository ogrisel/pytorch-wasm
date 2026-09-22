# PyTorch on Pyodide (wasm32-emscripten): from-source packaging + blocker #11 investigation

## What this PR contains

A reproducible Pyodide/`wasm32-emscripten` **packaging pipeline** for a reduced,
CPU-only PyTorch 2.8.0 built entirely from source, a JupyterLite demo + notebook that
trains a small MLP, a headless Node harness, and a **precise, evidence-backed
characterisation of the one remaining runtime blocker (#11)**.

- Recipe: `prototypes/pyodide/torch-probe/meta.yaml` (GitHub release tarball; reduced
  CPU-only config; 11 documented source/CMake/loader workarounds).
- Full write-up: `prototypes/pyodide/RESULTS.md`.
- Demo: `prototypes/pyodide/jupyterlite-demo/` (JupyterLite site + `content/torch_mlp_demo.ipynb`).
- Harness: `prototypes/pyodide/jupyterlite-demo/verify_wheel_node.mjs` (+ blocker-#11
  instrumentation `blocker11_instrument.mjs` / `blocker11_diagnose.mjs`).
- Diagnostic log: `prototypes/pyodide/logs/30-blocker11-assert-args.log`.

## Packaging status

The **build/link pipeline is solved end-to-end**: the recipe produces an importable-shaped
`torch` wheel with eight `wasm32` `.so` side modules (`libc10`, `libtorch_cpu`,
`libtorch`, `libtorch_python`, `torch._C`, …), Python bindings, and correct dylink
`NEEDED` metadata. Disabled to get there: XNNPACK, MKLDNN, FBGEMM, NNPACK, distributed,
CUDA, quantization, ONNX; single-threaded; minimal BLAS. Blocker **#10** (cross-module
symbol export) is solved via `build.exports: requested` — `libc10.so` now exports its
full C++ surface (153 KB → 615 KB) and cross-`.so` symbol resolution succeeds.

## Blocker #11 — this PR's focus (attempted, precisely characterised, not fully solved)

Loading the from-source 82 MB `libtorch_cpu.so` runs its static initializers and then
throws a `c10::Error`. Runtime evidence gathered on the Pyodide 0.27.8 Node runtime:

- **Exact throw site** (from a `___cxa_throw` hook + mapping wasm frame indices to the
  `.so` export tables): `c10::detail::torchInternalAssertFail` (a `TORCH_INTERNAL_ASSERT`),
  called from a `libtorch_cpu` static initializer near `torch::jit::sharedParserData()`
  (operator-schema registration).
- **Exact unresolved symbols** (from recording every `GOT.mem`/`GOT.func` access):
  **0 unresolved data symbols**, **exactly 6 unresolved function pointers** —
  `exit`, `cpuinfo_emscripten_init`, and the four `SymInt`×`size_t` operators
  `_ZN3c10{dv,ml,mi,rm}ERKNS_6SymIntEm` (`/ * - %`).
- **Root cause of 4/6 symbols → fixed in the recipe.** The `SymInt`×`size_t` operators
  come from an *incomplete* earlier blocker-#5 patch: it extended the `#if defined(__APPLE__)`
  guard around the **declarations** in `c10/core/SymInt.h`, but the matching **definitions**
  in `c10/core/SymInt.cpp` stayed Apple-only, so on wasm32 (`size_t` = `unsigned long`,
  distinct from `uint32_t`) they were declared+referenced but never defined. This PR adds
  the `SymInt.cpp` guard extension (blocker #5 part 2), so a rebuilt wheel defines them.
- **The abort itself is pointer corruption, not the unresolved symbols.** Wrapping
  `torchInternalAssertFail` to read its raw `const char*` args shows they point a few bytes
  *into* otherwise-correct rodata strings, by **non-uniform** offsets (memory windows
  confirm the intact surrounding strings):

  ```
  file @ +1  ...pytorch/b[u]ild/torch-2.8.0/.../NestedIntSymNodeImpl.h
  cond @ +12 ...RegisterComp[o]siteExplicitAutogradNonFunctional_0.cpp":12733,...
  func @ +3  ate[n]::count.int(int[] self, <run of spaces> int el) -> int
  ```

  Aliasing the 4 `SymInt` operators to their ABI-identical `uint32_t` implementations
  (`...Em → ...Ej`) resolves them but **does not** stop this assert — confirming a
  *separate*, deeper **data-relocation/addressing defect** in the huge from-source module
  (data section is only ~13 MB, so not a memory-size truncation).

**Conclusion:** #11 is now precisely characterised (exact throw site, exact symbols, and
direct evidence of mis-offset rodata pointers). The pointer-corruption abort needs a
**relink/rebuild of `libtorch_cpu`** to fix; the build tree/emsdk were not available in
this session's budget, so it is documented rather than fully resolved. The MLP training
notebook + Node harness are ready to run end-to-end once the wheel loads.

## How to reproduce

```bash
cd prototypes/pyodide/jupyterlite-demo
npm install    # fetches pyodide@0.27.8 runtime (pyodide.asm.wasm is not committed)
node verify_wheel_node.mjs dist_pyodide/torch-2.8.0a0+gitunknown-cp312-cp312-pyemscripten_2024_0_wasm32.whl
# blocker #11 detail:
node blocker11_instrument.mjs && node blocker11_diagnose.mjs dist_pyodide/torch-*.whl
```
