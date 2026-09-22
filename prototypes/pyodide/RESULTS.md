# Pyodide `torch` (wasm32-emscripten) packaging — empirical results

_Cloud-sandbox run, 2026-09-22. Everything below was actually executed; raw output is
in [`logs/`](logs/). This is the empirical companion to
[`../../docs/research-why-no-pytorch-wasm.md`](../../docs/research-why-no-pytorch-wasm.md)._

## TL;DR

Starting from a probe that reached ~89% of `torch_cpu`, this run drives a **complete
from-source build of PyTorch 2.8.0 for `wasm32-emscripten`** and **produces an
installable `torch` wheel**:

- ✅ Cross-compiles `protobuf` + `cpuinfo` + `onnx` to wasm static libs.
- ✅ Links **`libc10.so`**, **`libtorch_cpu.so`**, **`libtorch.so`**,
  **`libtorch_python.so`**, **`functorch/_C…so`**.
- ✅ Builds the Python bindings (`torch/_C…so`) — required `BUILD_PYTHON` to be
  force-enabled (see blocker #8) — and **packages a wheel** (~36 MB compressed).
- ✅ **Blocker #10 SOLVED — cross-side-module C++ symbols now export/resolve.** The real
  root cause was **not** compiler visibility but **pyodide-build's `pywasmcross` default
  `exports: pyinit`**, which links every side module with
  `-sSIDE_MODULE=2 -sEXPORTED_FUNCTIONS=@<PyInit_* only>`. A non-extension library like
  `libc10.so` has no `PyInit`, so its **entire C++ surface was GC-stripped**. Setting
  **`build.exports: requested`** makes each side module export all its public symbols.
  `libc10.so` grew **153 KB → 615 KB** and now exports
  `c10::getRuntimeDispatchKeySet`; cross-`.so` symbol resolution succeeds and
  `import torch` gets strictly further — `libc10.so` loads and runs its C++ static
  initializers cleanly ([`logs/28`](logs/28-node-verify-exportfix.log)).
- ⚠️ **Blocker #11 (remaining): a `c10::Error` / `TORCH_INTERNAL_ASSERT` aborts a
  `libtorch_cpu` static initializer, driven by mis-offset `const char*` pointers in the
  from-source 82 MB module.** Loading `libtorch_cpu.so` runs its ctors and then throws a
  `c10::Error` from `c10::detail::torchInternalAssertFail` (throw stack: a `libtorch_cpu`
  static initializer near `torch::jit::sharedParserData()` → `torchInternalAssertFail` in
  `libc10`). Reading the raw arguments passed to `torchInternalAssertFail` shows the
  `file`/`cond`/`func` `const char*` **point a few bytes *into* otherwise-correct rodata
  strings** (`file` at `+1` inside `.../pytorch/build/torch-2.8.0/.../NestedIntSymNodeImpl.h`,
  `cond` at `+12` inside `RegisterCompositeExplicitAutogradNonFunctional_0.cpp:12733`,
  `func` at `+3` inside `aten::count.int(...)`). The string bytes in memory are intact and
  contiguous, but the pointer *values* are shifted by small, **non-uniform** amounts — a
  data-relocation/addressing defect in this huge side module (not a memory-size issue: its
  data section is only ~13 MB). See [`logs/30`](logs/30-blocker11-assert-args.log).
- ✅ **Sub-finding #11a (root-caused + recipe-fixed): 4 of the 6 unresolved `GOT.func`
  symbols are the `SymInt`×`size_t` operators from an *incomplete* blocker #5 patch.**
  Enumerating `GOT.func`/`GOT.mem` at load time shows **all data (`GOT.mem`) symbols
  resolve** and exactly **6 function pointers are unresolved**:
  `_ZN3c10{ml,dv,mi,rm}ERKNS_6SymIntEm` (`operator *,/,-,%` on `(SymInt const&, size_t)`),
  plus `exit` and `cpuinfo_emscripten_init`. The blocker #5 patch had only extended the
  `#if defined(__APPLE__)` guard around the **declarations** in `c10/core/SymInt.h`; the
  matching **definitions** in `c10/core/SymInt.cpp` stayed Apple-only, so on wasm32 these
  `size_t` operators were declared+referenced but never defined → `GOT.func == 0`.
  `meta.yaml` now also extends the `SymInt.cpp` guard (blocker #5 part 2). A runtime
  experiment aliasing the missing `...Em` (size_t) symbols to the ABI-identical defined
  `...Ej` (uint32_t) implementations confirmed the aliases resolve, but the
  `TORCH_INTERNAL_ASSERT` above **still fires** — so #11a is a real but *separate*
  defect from the pointer-corruption abort ([`logs/30`](logs/30-blocker11-assert-args.log)).

Net: the *build/packaging* pipeline for a reduced CPU-only `torch` on Pyodide is solved
end-to-end, **and the cross-module symbol-export wall (#10) is solved**. Blocker #11 is now
**precisely characterised** (exact throw site, exact unresolved symbols, and direct
evidence of mis-offset rodata pointers), but the pointer-corruption abort needs a
**relink/rebuild of `libtorch_cpu`** to fix and is out of the hard time budget.

## Environment / versions

| Component | Version |
| --- | --- |
| Host OS | Ubuntu 24.04 (x86_64), 4 CPU, ~15 GiB RAM |
| Host Python | 3.12.3 |
| `pyodide-build` | 0.39.0 |
| Pyodide xbuildenv / runtime | 0.27.8 |
| Target CPython | 3.12.7 (`wasm32-emscripten`) |
| Emscripten (`emcc`) | 3.1.58 |
| Pyodide (Node test runtime) | 0.27.8 (`pyodide` npm) |

Reproducible toolchain: [`.cursor/install.sh`](../../.cursor/install.sh) provisions
`pyodide-build==0.39.0` + `resolvelib>=1.1.0` + the xbuildenv/emsdk, so
`pyodide build-recipes` works without sourcing emsdk manually.

## The recipe

[`torch-probe/meta.yaml`](torch-probe/meta.yaml) — points at the GitHub **release**
tarball `pytorch-v2.8.0.tar.gz` (which bundles `third_party/*`; PyTorch publishes no
PyPI sdist), forces a reduced CPU-only config, sets `build.exports: requested` (the
blocker #10 fix), and carries the documented source/CMake workarounds. Build command
per iteration:

```bash
cp torch-probe/meta.yaml <recipes>/packages/torch/meta.yaml
pyodide build-recipes-no-deps torch --recipe-dir=packages --force-rebuild
# (--continue reuses the extracted tree + objects for fast re-links, but pins the
#  build to a now-deleted temp build-env dir, so a version/flag change needs a full run)
```

Reduced config (what is disabled): `USE_CUDA/ROCM/XPU/MKLDNN/FBGEMM/NNPACK/QNNPACK/
XNNPACK/KINETO/DISTRIBUTED/TENSORPIPE/GLOO/MPI/OPENMP/NUMA/PYTORCH_QNNPACK=0`,
`USE_NUMPY=0`, `BUILD_TEST=0`, `BUILD_CAFFE2=0`, `ATEN_THREADING=NATIVE`
(single-threaded). Caffe2, quantization, distributed, ONNX runtime export, and the
vectorized XNNPACK microkernels are therefore out; eager ATen, autograd, `torch.nn`,
`torch.optim`, and the JIT/`nativert` C++ do build.

## Blockers found & fixed (in build order)

Each has a workaround in `torch-probe/meta.yaml`'s `build.script` and a log.

1. **No PyPI sdist.** All 50 `torch` releases ship only wheels; the scaffold
   `pythonhosted` source URL 404s ([`logs/08`](logs/08-torch-sdist-info.log)). Use the
   GitHub release asset (bundles submodules; [`logs/09`](logs/09-pytorch-github-release-assets.log),
   [`logs/11`](logs/11-pytorch-thirdparty-check.log)).
2. **No target `libpython` → pybind11 `Python::Module` missing.**
   `find_package(Python COMPONENTS Development.Module)` fails for wasm (CPython is
   statically linked into `pyodide.asm.wasm`); CMake generate aborts at
   `cmake/Dependencies.cmake`. Fix: stub a header-only `Python::Module` INTERFACE
   target ([`logs/12`](logs/12-torch-build-attempt1.log)).
3. **`__assert_fail` exception-spec mismatch.** `c10/macros/Macros.h` forward-declares
   glibc's `__assert_fail`; Emscripten musl differs. Fix: exclude `__EMSCRIPTEN__` from
   that `#if defined(NDEBUG)` block ([`logs/13`](logs/13-torch-build-attempt2.log)).
4. **`protoc` cross-compile.** PyTorch builds its own `protoc`, but under emcc it is a
   wasm/Node artifact that can't run as a host tool (`Permission denied`, exit 126).
   Fix: build a **host** `protoc` (version-matched libprotoc 3.13.0) from the vendored
   sources with host gcc, and point `CAFFE2_CUSTOM_PROTOC_EXECUTABLE` at it
   ([`logs/14`](logs/14-torch-build-attempt3.log), [`logs/15`](logs/15-host-protoc-build.log)).
5. **`SymInt * size_t` ambiguity (ILP32).** On wasm32 `size_t` is a distinct 32-bit
   type; upstream already special-cases Apple. Fix: extend the guard to
   `__EMSCRIPTEN__` in `c10/core/SymInt.h` ([`logs/16`](logs/16-torch-build-attempt4.log)).
6. **`ssize_t` undeclared** in `c10/util/Enumerate.h` (used by `torch/nativert`). Fix:
   use `std::ptrdiff_t` ([`logs/17`](logs/17-torch-build-attempt5.log)).
   *With #1–#6 the entire `torch_cpu` compilation (ATen + JIT + nativert) succeeds.*
7. **`libtorch_cpu.so` link: `llvm-nm: unknown argument '-,'`.**
   `caffe2_interface_library` (`cmake/public/utils.cmake`) wraps static libs as a
   single token `-Wl,--whole-archive,"PATH" -Wl,--no-whole-archive`; pyodide's
   `pywasmcross.filter_objects()` collects any arg ending in `.a` and feeds the whole
   `-Wl,…,<lib>.a` token to `emnm`, which errors. Fix: fold `--no-whole-archive` into
   the same `-Wl,` group so the token ends in `--no-whole-archive` (a form torch itself
   uses for `torch_xpu_ops`); verified to still force-include the archive
   ([`logs/20`](logs/20-torch-build-run1.log)). **`libtorch_cpu.so` (~59 MB) then links.**
8. **No importable module: `BUILD_PYTHON` force-disabled.** Because
   `Development.Module` is "missing", `cmake/Dependencies.cmake` sets `BUILD_PYTHON
   OFF`, so `libtorch_python` / `torch._C` are never built (and `functorch` then fails
   to find `-ltorch_python`). Pyodide extensions are `SIDE_MODULE`s that resolve Python
   symbols at load, so keep `BUILD_PYTHON ON`. Fix: patch that branch to keep it on
   ([`logs/21`](logs/21-torch-build-run2-continue.log),
   [`logs/22`](logs/22-torch-build-run3-buildpython.log)). **`libtorch_python.so`,
   `functorch/_C`, and `torch/_C` then build and link.**
9. **Wheel packaging aborts on version.** `setup.py` derives `2.8.0a0+gitUnknown` from a
   tarball with no git metadata, so pyodide-build's `check_versions_match` aborts before
   the wheel is retagged to the pyodide platform. Fix: `PYTORCH_BUILD_VERSION=2.8.0`
   ([`logs/24`](logs/24-torch-build-run4-repackage.log)). **A wheel is produced.**
10. **Cross-side-module symbols GC-stripped — SOLVED.** Installing the wheel and loading
    its libraries first failed with:

    ```
    Dynamic linking error: cannot resolve symbol
    _ZN3c1024getRuntimeDispatchKeySetENS_11DispatchKeyE   (c10::getRuntimeDispatchKeySet)
    ```

    `libtorch_cpu.so` imports this from `libc10.so`, but `emnm libc10.so` showed the
    symbol entirely absent and the whole module only ~150 KB (a full c10 is far larger).
    Forcing `-fvisibility=default` (an early attempt) was **necessary but not
    sufficient**. **The real root cause:** pyodide-build's `pywasmcross` defaults to
    `exports: pyinit`, which links every `SIDE_MODULE` with
    `-sSIDE_MODULE=2 -sEXPORTED_FUNCTIONS=@<PyInit_* only>`; `libc10.so` has no `PyInit`
    symbol, so wasm-ld had **no export roots** and GC-stripped its entire C++ surface.

    **Fix:** set `build.exports: requested` in the recipe. `pywasmcross` then exports
    **all public symbols** of each object (`calculate_object_exports_readobj`), keeping
    `-sSIDE_MODULE=2` (so static-archive dedup still works — plain `SIDE_MODULE=1` /
    `whole_archive` force-includes the doubled `libcpuinfo.a` and dies on duplicate
    `cpuinfo_get_uarch`). Relinked (objects reused):

    | lib | before | after |
    | --- | --- | --- |
    | `libc10.so` | 153 KB | **615 KB** (exports `getRuntimeDispatchKeySet`) |
    | `libtorch_cpu.so` | 59 MB | **83 MB** |
    | `libtorch_python.so` | 1.8 MB | **10.9 MB** |

    With this, `libc10.so` loads and runs its static initializers; cross-`.so`
    resolution succeeds ([`logs/28`](logs/28-node-verify-exportfix.log)).
11. **`libtorch_cpu` static-initializer abort — precisely characterised (remaining).**
    After #10, loading `libtorch_cpu.so` runs its ctors and then aborts. Attacking it with
    the Node harness on Pyodide 0.27.8 produced the following runtime evidence
    ([`logs/30`](logs/30-blocker11-assert-args.log)):

    - **Exact throw site.** Hooking `___cxa_throw` prints the throw stack. It is a
      `c10::Error` thrown by `c10::detail::torchInternalAssertFail` (a
      `TORCH_INTERNAL_ASSERT`), called from a `libtorch_cpu` static initializer whose
      nearest exported symbol is `torch::jit::sharedParserData()` (JIT frontend / operator
      schema registration). Mapping the wasm stack frame indices to the `.so` **export
      tables** (no name section is present, but exports suffice) gave the names:
      `libc10` `func[548] = c10::detail::torchInternalAssertFail`,
      `func[1828] = c10::detail::torchCheckFail`; `libtorch_cpu`
      `func[23534] = torch::jit::sharedParserData()`.

    - **Exact unresolved symbols.** Recording every `GOT.mem`/`GOT.func` access shows
      **0 unresolved data symbols** and **exactly 6 unresolved function pointers**:
      `_ZN3c10dvERKNS_6SymIntEm`, `_ZN3c10mlERKNS_6SymIntEm`, `_ZN3c10miERKNS_6SymIntEm`,
      `_ZN3c10rmERKNS_6SymIntEm` (the `SymInt`×`size_t` `/ * - %` operators — see
      sub-finding #11a and the blocker #5 part-2 recipe fix), plus `exit` and
      `cpuinfo_emscripten_init`.

    - **The abort is pointer corruption, not the unresolved symbols.** Wrapping
      `torchInternalAssertFail` at symbol-resolution time to read its raw arguments shows
      the `file`/`cond`/`func` `const char*` **point a few bytes *into* correct rodata
      strings**, by *non-uniform* offsets. A memory window around each pointer confirms the
      surrounding bytes are the intact strings:

      ```
      file @ +1  ...pytorch/b[u]ild/torch-2.8.0/.../NestedIntSymNodeImpl.h
      cond @ +12 ...RegisterComp[o]siteExplicitAutogradNonFunctional_0.cpp":12733,...
      func @ +3  ate[n]::count.int(int[] self, <run of spaces> int el) -> int
      ```

      So `libtorch_cpu`'s static initializers hand `c10` **mis-offset pointers**, which
      trips the internal assert. Aliasing the 4 missing `SymInt` operators to their
      ABI-identical `uint32_t` implementations (`...Em → ...Ej`) resolves them but **does
      not** stop this abort — confirming it is a *separate*, deeper data-relocation defect
      in the from-source 82 MB module (its data section is only ~13 MB, so it is not a
      memory-size truncation). Fixing it requires a relink/rebuild of `libtorch_cpu`
      (out of the hard time budget). The Node harness
      [`verify_wheel_node.mjs`](jupyterlite-demo/verify_wheel_node.mjs) reproduces the load
      up to this abort.

    - **Clean from-scratch rebuild — hypothesis DISPROVEN, defect is inherent
      ([`logs/32`](logs/32-blocker11-clean-build.log)).** A full clean build with the
      current recipe (`exports: requested` applied *from the start*, blocker #5 part-2
      `SymInt.cpp` fix included) compiled and linked all 1536 targets in one consistent
      pass and produced a wheel (`libtorch_cpu.so` = 82.68 MB, essentially identical to the
      previously-relinked 82.72 MB). Re-running the harness shows:
        - ✅ **The `SymInt.cpp` fix works:** unresolved `GOT.func` dropped **6 → 2** — only
          `exit` and `cpuinfo_emscripten_init` remain; the 4 `SymInt`×`size_t` operators now
          resolve.
        - ❌ **The `torchInternalAssertFail` abort is unchanged** — same throw stack
          (`torch::jit::sharedParserData()` internals → `torchInternalAssertFail`), same
          mis-offset `const char*` pointers. So the post-hoc-relink hypothesis is **wrong**:
          the defect is inherent to the from-source wasm32 build, not the relink.
    - **Precise nature of the abort.** The assert is `lexer.h:143` `AT_ASSERT(kind == 0)`
      inside `torch::jit::TokenTrie::insert` (building the JIT keyword trie at static-init).
      The captured `line = 143` **matches `lexer.h:143` exactly**, i.e. the `__LINE__`
      *immediate* argument is correct — but the `__func__`/`__FILE__`/`condMsg`
      **pointer** arguments point into *unrelated* rodata (a different assert's
      precompiled message, e.g. `.../c10/core/TensorImpl.h":1733,...`). So integer
      immediates relocate correctly while `MEMORY_ADDR` (pointer-to-rodata) relocations are
      mis-applied in this 82 MB module — corrupting the token strings the trie reads, which
      trips `kind == 0`. This is a wasm-ld/Emscripten **data-relocation defect that scales
      with module size** (`libc10` is fine; `libtorch_cpu` is not), not a source bug.
    - **Fallback tried — `wasm-opt`/Binaryen RULED OUT
      ([`logs/33`](logs/33-blocker11-noopt-relink.log)).** Relinked `libtorch_cpu` from the
      same `.o` objects with Emscripten's optimizer fully disabled (stripped every
      `-O2/-Oz` from both the CMake link line *and* the pywasmcross-injected `ldflags`, so
      only `-O0` remained). This produced the raw **273 MB unoptimized** side module
      (vs 82.68 MB with `-Oz` — proof `wasm-opt` was actually skipped; the three prior
      relinks that kept `-Oz` came out byte-identical, sha256 `a45912cb…`). Loading it in
      the harness reproduces the **identical** abort with the **same mis-offset pointers**
      (`file`→`SymBool.h`, `func`→`aten::count.int` schema, `cond`→`TensorImpl.h:1733`,
      `line = 143`). Two conclusions:
        - ❌ **Binaryen/`wasm-opt` is not the cause** — the mis-relocation is present in the
          pre-optimizer wasm, so the defect lives in **wasm-ld's `MEMORY_ADDR` relocation
          emission** (or Pyodide's load-time `__wasm_apply_data_relocs`), not the optimizer.
        - ✅ **Throw site independently confirmed by symbol names.** The `-O0`/`-g0` binary
          keeps the wasm name section, so the stack now reads literally
          `torch::jit::TokenTrie::insert(char const*, int)` ←
          `torch::jit::SharedParserData::SharedParserData()` — exactly the `lexer.h:143`
          static-init site inferred above.
    - **Remaining fallback (out of budget):** **split `libtorch_cpu` into smaller side
      modules** so no single module hits the relocation-scale defect. This is now the only
      untried avenue (optimizer, EH mode, and post-hoc-relink are all ruled out), but it is
      invasive CMake surgery (partitioning ~1.5k sources across multiple shared libs with
      correct inter-lib symbol exports) and was not attempted within budget.

## How the wheel is loaded / tested

The wheel bundles **eight** `.so` side modules with correct dylink `NEEDED` metadata
(`libc10 ← libtorch_cpu ← {libtorch, libshm} ← libtorch_python ← {torch/_C,
functorch/_C}`). Two loader facts were established (both needed for any consumer,
including JupyterLite):

- micropip's auto-loader (`_load_libraries` → `loadDynlibsFromPackage`) loads the libs
  **non-globally** and resolves `NEEDED` from a `torch.libs/` dir; our libs live in
  `torch/lib/`, so it throws
  `Error: Didn't expect to load any more file_packager files!`
  ([`logs/23`](logs/23-node-verify-import-train.log),
  [`logs/25`](logs/25-node-verify-pyodide-wheel.log)).
- The reliable path is to unpack the wheel and load the libraries **in dependency
  order** with `pyodide._api.loadDynlib(path, /*global=*/true, [".../torch/lib"])`.
  With the export fix, `libc10.so` now loads **and runs its C++ static initializers**,
  and cross-`.so` symbol resolution succeeds; the loader then stops at blocker #11 while
  running `libtorch_cpu.so`'s static initializers
  ([`logs/28`](logs/28-node-verify-exportfix.log)).

The headless reproducer is
[`jupyterlite-demo/verify_wheel_node.mjs`](jupyterlite-demo/verify_wheel_node.mjs)
(Node + `pyodide@0.27.8`); the intended browser demo is the JupyterLite site below.

## Reviewer demo (JupyterLite)

[`jupyterlite-demo/`](jupyterlite-demo/) ships a one-command build of a static
JupyterLite site (Pyodide kernel) that includes the built wheel and a notebook
(`content/torch_mlp_demo.ipynb`) that trains a small MLP (`nn.Linear`+`ReLU`,
`CrossEntropyLoss`, `SGD`) on a synthetic 2D dataset:

```bash
cd prototypes/pyodide/jupyterlite-demo
./build.sh dist/torch-*_wasm32.whl        # pins Pyodide 0.27.8 to match the wheel ABI
python -m http.server -d _site 8000       # open http://localhost:8000/lab/index.html
```

The notebook uses the ordered `loadDynlib` bootstrap above and reports status honestly:
it shows **blocker #10 is solved** (`libc10.so` loads and exports its symbols) and then
surfaces **blocker #11** (the `libtorch_cpu` static-initializer abort). **On this build
the training cells (2–5) do not yet run**; they are the intended demonstration and
execute end-to-end once blocker #11 is resolved. The notebook + logs make the exact
current failure visible for a reviewer in-browser.

## What this refines vs. the research doc

- Concretely resolves most of the "dependency graph / native linking" concerns:
  protobuf/onnx **do** cross-compile, and **all** torch libraries (including the 59 MB
  `libtorch_cpu.so` and the Python bindings) **link** as wasm side modules.
- **Solves** the long-standing "dynamic linking" wall for symbol *export/resolution*:
  the fix was a pyodide-build packaging setting (`exports: requested`), not a
  compiler-visibility or `EXPORT_ALL`/`LINKABLE` problem as first assumed.
- Pins the *true* remaining runtime wall to a specific, reproducible problem: **one
  unresolved `GOT.func` function-pointer** in `libtorch_cpu`'s static initializers,
  plus the ancillary **multi-`.so` load-order/scope** handling in micropip.
- Adds structural blockers the doc under-weights: **no PyPI sdist** and **no target
  `libpython`** (breaking `find_package(Python Development.Module)` / pybind11 /
  `BUILD_PYTHON`).

## Committed artifacts

- Recipe: [`torch-probe/meta.yaml`](torch-probe/meta.yaml) — 9 build workarounds +
  `exports: requested` (blocker #10 fix), all documented inline.
- Logs: [`logs/`](logs/) — full build runs (`20`–`22`, `24`, `27`), wheel-load probes
  (`23`, `25`, `26`), the **export-fix verification** ([`28`](logs/28-node-verify-exportfix.log)),
  the **unresolved-`GOT.func` diagnostic** ([`29`](logs/29-got-unresolved-eh-funcptrs.log)),
  and earlier probe logs (`00`–`18`).
- JupyterLite demo + Node verifier: [`jupyterlite-demo/`](jupyterlite-demo/); the rebuilt
  **symbol-exporting wheel** (~36 MB) is committed under
  [`jupyterlite-demo/dist_pyodide/`](jupyterlite-demo/dist_pyodide/) and `dist/`.
