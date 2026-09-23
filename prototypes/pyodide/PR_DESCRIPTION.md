# PyTorch on Pyodide (wasm32-emscripten): from-source packaging + single-module (blocker #11/#12) investigation

## What this PR contains

A reproducible Pyodide/`wasm32-emscripten` **packaging pipeline** for a reduced,
CPU-only PyTorch 2.8.0 built entirely from source, a JupyterLite demo + notebook that
trains a small MLP, a headless Node harness, and a **precise, evidence-backed
characterisation of the remaining runtime blocker** — including the emscripten-forge
sibling's **single-module** strategy, which is replicated here and shown to remove the
cross-`.so` relocation wall and the `lexer.h:143` assert, but to hit the same underlying
data-relocation defect as `std::bad_alloc`.

- Recipe: `prototypes/pyodide/torch-probe/meta.yaml` (GitHub release tarball; reduced
  CPU-only config; documented source/CMake/loader workarounds).
- Single-module link: `prototypes/pyodide/torch-probe/link_single_module.sh`.
- Full write-up: `prototypes/pyodide/RESULTS.md`.
- Demo: `prototypes/pyodide/jupyterlite-demo/` (JupyterLite site + `content/torch_mlp_demo.ipynb`).
- Harnesses: `verify_wheel_node.mjs` (multi-module), `verify_single_module_node.mjs` +
  `diag_single_module.mjs` (single-module), and blocker-#11 instrumentation
  `blocker11_instrument.mjs` / `blocker11_diagnose.mjs`.
- Diagnostic logs: `logs/30` (blocker #11 assert args), `logs/34` (single-module link),
  `logs/35` (single-module `bad_alloc` stack), `logs/36` (3.1.73-linked, same failure).

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
  the `SymInt.cpp` guard extension (blocker #5 part 2); the clean rebuild **verifies** it —
  unresolved `GOT.func` drops **6 → 2** (only `exit`, `cpuinfo_emscripten_init` remain).
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

**Clean from-scratch rebuild — hypothesis disproven, defect is inherent
(`logs/32-blocker11-clean-build.log`).** A full clean build with the current recipe
(`exports: requested` applied *from the start*, `SymInt.cpp` fix included) compiled and
linked all 1536 targets in one consistent pass and produced a wheel with
`libtorch_cpu.so` = 82.68 MB (essentially identical to the previously-relinked 82.72 MB).
Re-running the harness on the fresh wheel shows:

- ✅ **The `SymInt.cpp` fix works:** unresolved `GOT.func` dropped **6 → 2** — only
  `exit` and `cpuinfo_emscripten_init` remain; the four `SymInt`×`size_t` operators now
  resolve at build time.
- ❌ **The `torchInternalAssertFail` abort is unchanged** — same throw stack, same
  mis-offset `const char*` pointers. So the post-hoc-relink hypothesis is **wrong**: the
  defect is inherent to the from-source wasm32 build, not the relink.

**Precise nature of the abort.** The assert is `lexer.h:143` `AT_ASSERT(kind == 0)` inside
`torch::jit::TokenTrie::insert` (building the JIT keyword trie at static-init). The captured
`line = 143` **matches `lexer.h:143` exactly** — the `__LINE__` *immediate* argument is
correct — but the `__func__`/`__FILE__`/`condMsg` **pointer** arguments point into
*unrelated* rodata (a different assert's precompiled message). So integer immediates
relocate correctly while `MEMORY_ADDR` (pointer-to-rodata) relocations are mis-applied in
this 82 MB module, corrupting the token strings the trie reads and tripping `kind == 0`.
This is a wasm-ld/Emscripten **data-relocation defect that scales with module size**
(`libc10` is fine; `libtorch_cpu` is not), not a source bug.

**Fallback tried — `wasm-opt`/Binaryen ruled out (`logs/33-blocker11-noopt-relink.log`).**
Relinked `libtorch_cpu` from the same `.o` with the optimizer fully disabled (stripped every
`-O2/-Oz` from both the CMake link line and the pywasmcross-injected `ldflags`, leaving only
`-O0`). This produced the raw **273 MB unoptimized** side module (vs 82.68 MB with `-Oz`,
proving `wasm-opt` was actually skipped). It reproduces the **identical** abort with the
**same mis-offset pointers** — so Binaryen is not the cause; the defect is in **wasm-ld's
`MEMORY_ADDR` relocation emission** (or Pyodide's load-time `__wasm_apply_data_relocs`). As a
bonus, the unstripped binary keeps the wasm name section, so the stack now reads literally
`torch::jit::TokenTrie::insert(char const*, int)` ←
`torch::jit::SharedParserData::SharedParserData()`, independently confirming the `lexer.h:143`
static-init throw site.

## Blocker #12 — single-module strategy (from the emscripten-forge sibling)

The sibling got real upstream torch 2.8.0 to import + train an MLP in-browser by linking
**one** `torch/_C.*.so` SIDE_MODULE (~143 MB) that `--whole-archive`s
`libtorch_python + libtorch + libtorch_cpu + c10 + deps` and exports `PyInit__C`, so every
cross-module C++ symbol resolves *inside one module*. This PR replicates that for Pyodide
0.27.8 (`link_single_module.sh`): from the completed shared build's CMake tree, it links all
**1241** loose torch `.o` (whole-include) + `stub.o` (compiled as **C**, so `PyInit__C →
initModule` is unmangled) + `cpuinfo_emscripten_init.o` + the whole-archived `libonnx.a` and
grouped `libcpuinfo/onnx_proto/protobuf` into one `-sSIDE_MODULE=2` module exporting only
`PyInit__C`.

- ✅ **Links + exports** a 101 MB `_C.cpython-312-wasm32-emscripten.so` (dylink `memsize`
  13.0 MB, `tablesize` 165760 — all sane) — `logs/34`.
- ✅ **The `lexer.h:143` `torchInternalAssertFail` abort is eliminated** — loading runs its
  static initializers past that exact site. The single-module approach genuinely side-steps
  the cross-`.so` `GOT.func`/relocation wall.
- ⚠️ **New wall: `std::bad_alloc`** in the *same* JIT schema-parse subsystem. Full demangled
  stack (`logs/35`): `__wasm_call_ctors` → `_GLOBAL__sub_I_TraceType_2.cpp` →
  `TORCH_LIBRARY_IMPL_init_aten_Tracer_2` → `torch::Library::_impl` → `parseSchemaOrName` →
  `torch::jit::Source::calc_line_start_offsets()` → `vector<size_t>::push_back` →
  `std::bad_alloc`. The op-schema `string_view` handed to `Source` has a **corrupted (huge)
  size**, so `calc_line_start_offsets` scans unboundedly and overflows the offsets vector. A
  `memcpy`-import hook never fires (it's a non-owning *view*, not a huge copy), so the *size
  field* is mis-relocated — the **same rodata mis-relocation defect as #11**, now surfacing
  as an absurd allocation instead of a bad assert pointer.
- ❌ **Not a wasm-ld version issue.** Relinking the identical objects with **emsdk 3.1.73**
  (the sibling's revision) reproduces the same `std::bad_alloc` at the same site (`logs/36`),
  so the defect is baked into the **clang-3.1.58-compiled objects** (or Pyodide 0.27.8's
  load-time `__wasm_apply_data_relocs`), not the linker. The sibling succeeded with a
  *fully* 3.1.73 toolchain (compile + link + runtime); this track is pinned to Pyodide
  0.27.8 (emscripten 3.1.58 / CPython 3.12), and a full 3.1.73 recompile is not guaranteed
  ABI-loadable there. Loading the module in Pyodide 0.28.3 (CPython 3.13) is inconclusive —
  it fails earlier on an unrelated omitted `libshm` symbol.

## Blocker #13 — option (a) tested: full 3.1.73 RECOMPILE (the defect follows the RUNTIME)

The one lever the prior runs had **not** pulled was the *compiler*: #12 proved a 3.1.73
**relink** of 3.1.58 objects reproduces the defect, but a full **recompile** with the newer
clang was still open. This run does it — and it is decisive.

- **Full recompile with emsdk/clang 3.1.73** (`build_torch_candidate.sh`,
  `SKIP_EMSCRIPTEN_VERSION_CHECK=1` so pyodide-build 0.39.0 uses the ambient non-pinned
  `emcc`). Every one of the ~1500 objects rebuilds with the newer clang + 3.1.73 sysroot;
  the 3.1.58-tuned source patches port with **zero** changes.
- **3.1.73 is a stricter linker (multi-`.so`):** `lib/libtorch_cpu.so` now fails with
  `em++: error: undefined exported symbol: "_cpuinfo_cache" [-Wundefined] [-Werror]`
  (3.1.73 promotes an undefined *exported* symbol under `exports: requested` to a hard
  error). Objects are still all produced; the single-module link is unaffected.
- **Single-module link with 3.1.73** (`link_single_module.sh`, now `EMSDK`-overridable):
  clean → **101,148,292 B** `_C.*.so` exporting `PyInit__C` (`logs/38`).
- **Load under Pyodide 0.27.8** (`logs/37`):
  - ✅ **Loads with no undefined-symbol / `LinkError`** — the fully-3.1.73-recompiled
    module is **load-ABI-compatible** with the 0.27.8 main module (all libc++/libc imports
    resolve; static init runs deep). The feared "3.1.73 build may not be 0.27.8-loadable"
    incompatibility **does not exist for loading**.
  - ❌ **Same mis-relocated-rodata defect** — `loadDynlib` throws a `c10::Error` whose
    `what()` `string_view` has a corrupted (huge) size that sweeps disjoint rodata
    (`SparseTensor.cpp":633` + `please report a bug to PyTorch.` + `True or False) with` +
    `got input with sizes` + `custom_class_detail.h` + the GRU weight docstring). The
    failure *site* shifted vs the 3.1.58 single-module (early `aten` Tracer `bad_alloc` →
    later corrupted error-message view) because the newer codegen reshuffled `.data`, but
    the **scale-dependent corruption is unchanged**.

**Decisive narrowing:** compiler = 3.1.73, linker = 3.1.73, optimizer ruled out, EH mode
ruled out — the **only** remaining 3.1.58-era component is the **Pyodide 0.27.8 runtime**.
So the mis-applied `MEMORY_ADDR` (pointer/`string_view`-size) relocations are applied by
**Pyodide 0.27.8's load-time `__wasm_apply_data_relocs`** at this module scale, **not** by
clang codegen. The emscripten-forge sibling worked because it used the **3.1.73 runtime**
end-to-end.

## ccache (accelerator for the many full rebuilds)

Wired ccache into the emcc compile path so each candidate-toolchain rebuild after the first
is cheap. `EM_COMPILER_WRAPPER` must be an **absolute** path (emcc `execv()`s it, no `PATH`
lookup — the original groundwork's bare `ccache` was a bug, now fixed); `CCACHE_DIR=
/workspace/.ccache` (persistent, 30 GB, `compiler_check=content`). Confirmed on a second
build: **96.8 % hits (1347/1391)** vs the cold build's 2 %, cutting the compile phase from
~25 min to ~3 min (`ccache -s`).

## Conclusion (honest)

Blocker #11 is precisely characterised (throw site `lexer.h:143` confirmed by symbol names;
exact symbols; direct mis-offset rodata pointers); post-hoc relink, optimizer, EH mode,
**and now a full newer-clang (3.1.73) recompile** are all ruled out as fixes. The
**single-module strategy (blocker #12) is a real advance** — it removes the cross-`.so`
relocation wall and the `lexer.h:143` assert — but the underlying **scale-dependent
data-relocation defect** persists, and blocker #13 pins it to the **Pyodide 0.27.8 load-time
relocation applier** (`__wasm_apply_data_relocs`), *independent of the compiler and linker*.
So **option (a) — recompile with a newer clang while keeping the 0.27.8 runtime — is not
feasible under the 0.27.8 pin**, and bisecting intermediate revisions is unnecessary (no
*compiler* revision can fix a *runtime*-applied defect). Real torch therefore does **not yet
import/train** on Pyodide 0.27.8. The two remaining viable paths: **(b) split `libtorch_cpu`**
below the relocation-scale threshold, or **bump the Pyodide runtime** past 0.27.8 (a
coordinated emscripten/CPython upgrade, as the sibling did). No demo is fabricated: the
notebook + harnesses report the exact current failure.

## How to reproduce

```bash
# 1) Build the reduced from-source torch (produces the CMake tree + .o objects):
cp prototypes/pyodide/torch-probe/meta.yaml <pyodide-recipes>/packages/torch/meta.yaml
pyodide build-recipes-no-deps torch --recipe-dir=packages --force-rebuild

# 2) Link the single combined module (blocker #12) from that build tree:
BUILD_DIR=<...>/packages/torch/build/torch-2.8.0 \
  bash prototypes/pyodide/torch-probe/link_single_module.sh
#    -> build_torch/single_module_stage/_C.cpython-312-wasm32-emscripten.so (101 MB)

cd prototypes/pyodide/jupyterlite-demo
npm install    # fetches pyodide@0.27.8 runtime (pyodide.asm.wasm is not committed)

# multi-module (blocker #11, lexer.h:143 assert):
node verify_wheel_node.mjs dist_pyodide/torch-2.8.0a0+gitunknown-cp312-cp312-pyemscripten_2024_0_wasm32.whl
# single-module (blocker #12, std::bad_alloc past the assert):
node verify_single_module_node.mjs /abs/.../_C.cpython-312-wasm32-emscripten.so dist_pyodide/torch-*.whl
node diag_single_module.mjs /abs/.../_C.cpython-312-wasm32-emscripten.so   # demangled throw stack
```
