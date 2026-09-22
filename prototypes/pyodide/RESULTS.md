# Pyodide `torch` build probe — empirical results

_Cloud-sandbox run, 2026-09-22. Everything below was actually executed; raw output
is in [`logs/`](logs/). This is the empirical companion to
[`../../docs/research-why-no-pytorch-wasm.md`](../../docs/research-why-no-pytorch-wasm.md):
it confirms/refines that blocker analysis with real commands and real error messages._

## TL;DR

- The Pyodide cross-build toolchain works: a compiled recipe (`cffi`) builds
  end-to-end to a `wasm32-emscripten` wheel. ✅
- A from-source **`torch`** build for `wasm32-emscripten` is **not** blocked by a
  single wall. With four small, targeted workarounds it gets **surprisingly far**:
  it cross-compiles `protobuf`/`cpuinfo`/`onnx` to wasm static libs, **links
  `libc10.so` as a wasm side-module**, and compiles **~1219/1374 (~89%) of
  `torch_cpu`** (ATen + JIT) before stopping on the *next* source-level portability
  bug. The remaining work is a long tail of similar 32-bit/POSIX/Emscripten source
  fixes, then the link-time `-pthread`/`wasm-ld`/binary-size walls.
- Two **structural** blockers are worth calling out because they gate any recipe:
  1. PyTorch publishes **no sdist** on PyPI, so the canonical pyodide `source.url`
     404s; you must use the GitHub release tarball (which bundles submodules).
  2. The wasm cross-build has **no target `libpython`** (CPython is statically linked
     into `pyodide.asm.wasm`), so CMake `find_package(Python COMPONENTS
     Development.Module)` fails and pybind11's `Python::Module` target is missing.

## Environment / versions

Captured in [`logs/00-versions-summary.txt`](logs/00-versions-summary.txt),
[`logs/02`](logs/02-pyodide-version.log), [`logs/04`](logs/04-config.log),
[`logs/06`](logs/06-emcc-version.log).

| Component | Version |
| --- | --- |
| Host OS | Ubuntu 24.04.4 LTS (x86_64), 4 CPU, ~15 GiB RAM |
| Host Python | 3.12.3 |
| `pyodide-build` | 0.39.0 (`pyodide-cli` 0.5.0) |
| Pyodide xbuildenv | 0.27.8 |
| Target CPython | 3.12.7 (`wasm32-emscripten`, `pyemscripten_2024_0`) |
| Emscripten (`emcc`) | 3.1.58 (`a41843e0860e…`) |

Setup that was needed beyond the README steps:
- `pip install pyodide-build` pulls `resolvelib==1.0.1`, but `pyodide-build` 0.39.0
  imports `resolvelib.providers.AbstractProvider` as a generic (`[...]`), which needs
  **`resolvelib>=1.1.0`**. Symptom before the fix:
  `TypeError: type 'AbstractProvider' is not subscriptable`. Fixed with
  `pip install 'resolvelib>=1.1.0'`.
- `pyodide xbuildenv install` installs the cross-build env but **not** emsdk; a
  separate `pyodide xbuildenv install-emscripten` is required (installs emsdk 3.1.58).
- Host build tools `ninja` + `pyyaml` were installed for torch's build.

## Step 1 & 2 — toolchain sanity (PASS)

```bash
pip install pyodide-build 'resolvelib>=1.1.0' ninja pyyaml
pyodide xbuildenv install
pyodide xbuildenv install-emscripten
pyodide config get emscripten_version      # -> 3.1.58
git clone --depth 1 https://github.com/pyodide/pyodide-recipes
pyodide build-recipes cffi --recipe-dir=packages
```

Result ([`logs/07-sanity-cffi-build.log`](logs/07-sanity-cffi-build.log)): built
`cffi` (a C-extension package) and its deps in ~5s, producing
`cffi-2.0.0-cp312-cp312-pyemscripten_2024_0_wasm32.whl`. The toolchain is correct.
(Note: `cffi` shipping is itself the death of the historical "torch needs cffi and
Pyodide lacks it" blocker — see the research doc §2.)

## Step 3 — `torch` probe

Recipe: [`torch-probe/meta.yaml`](torch-probe/meta.yaml) (final version, synced from
the working copy in the recipes checkout). Command used for each attempt:

```bash
cp torch-probe/meta.yaml <recipes>/packages/torch/meta.yaml
pyodide build-recipes-no-deps torch --recipe-dir=packages --force-rebuild
```

`build-recipes-no-deps` is used so the probe isolates `torch` itself rather than
building its whole runtime dependency graph (numpy/sympy/…). The recipe forces
CPU-only, single-threaded, and disables x86-only/threaded backends
(`USE_CUDA/ROCM/XPU/MKLDNN/FBGEMM/NNPACK/QNNPACK/XNNPACK/KINETO/DISTRIBUTED/
TENSORPIPE/GLOO/MPI/OPENMP/NUMA=0`, `ATEN_THREADING=NATIVE`, `BUILD_CAFFE2=0`,
`USE_NUMPY=0`).

### Source: PyTorch has no PyPI sdist (structural blocker #1)

[`logs/08-torch-sdist-info.log`](logs/08-torch-sdist-info.log):

```
torch versions WITH an sdist on PyPI: NONE
total torch releases: 50
=== HEAD check scaffold URL (torch 2.8.0 source tarball) ===
HTTP/2 404
```

All 50 `torch` releases publish only binary wheels. The scaffold's
`files.pythonhosted.org/.../torch-2.8.0.tar.gz` **404s**. The only from-source option
is the GitHub *release asset* `pytorch-v2.8.0.tar.gz` (331 MB), which — unlike the
auto-generated GitHub archive — **bundles the `third_party/*` submodules**
(XNNPACK, sleef, protobuf, cpuinfo, pthreadpool, fmt, FP16, FXdiv, onnx, pybind11, …;
verified in [`logs/11`](logs/11-pytorch-thirdparty-check.log)). The recipe was pointed
at it with `sha256 c70a2c9488f6f6e8af5982a10d1cc2c37b7df5e6506d839daa5d5e250953d7b5`
([`logs/09`](logs/09-pytorch-github-release-assets.log),
[`logs/10`](logs/10-pytorch-tarball-sha256.log)).

### Attempt 1 — CMake generate fails: pybind11 `Python::Module` (structural blocker #2)

[`logs/12-torch-build-attempt1.log`](logs/12-torch-build-attempt1.log). CMake
configures for 54.7s (all disable flags honored), then the **generate** step fails:

```
-- Found Python: .../bin/python (found version "3.12.3") found components: Interpreter
   missing components: Development.Module NumPy
...
CMake Error at cmake/Dependencies.cmake:874 (target_link_libraries):
  The link interface of target "pybind::pybind11" contains:
    Python::Module
  but the target was not found.
```

Root cause (verified): the xbuildenv provides target Python **headers** but **no
`libpython`** — CPython is statically linked into `pyodide.asm.wasm`
(`sysconfigdata`: `STATIC_LIBPYTHON=1`, `Py_ENABLE_SHARED=0`). So
`find_package(Python COMPONENTS Development.Module)` can't be satisfied for the wasm
target, the `Python::Module` imported target is never created, and
`cmake/Dependencies.cmake:878 target_link_libraries(pybind::pybind11 INTERFACE
Python::Module)` aborts. Pyodide extension modules are `SIDE_MODULE`s that resolve
Python symbols at load time, so a link-time `libpython` is intentionally absent —
`torch`'s CMake does not account for that mode.

**Workaround in the recipe:** define a header-only stub `Python::Module` INTERFACE
target so generation proceeds (headers still come from `CFLAGS`).

### Attempt 2 — c10 compiles, then `__assert_fail` exception-spec mismatch (blocker #3)

[`logs/13-torch-build-attempt2.log`](logs/13-torch-build-attempt2.log). With the stub,
generation passes and compilation starts (reaches ninja step ~173/1374, builds
`protoc`, compiles 170+ objects). It then fails:

```
c10/macros/Macros.h:397:5: error: exception specification in declaration does not
  match previous declaration
      __assert_fail(
emscripten/cache/sysroot/include/assert.h:19:16: note: previous declaration is here
  _Noreturn void __assert_fail (const char *, const char *, int, const char *);
```

`c10/macros/Macros.h` (under `#if defined(NDEBUG)`) forward-declares glibc's
`void __assert_fail(const char*, const char*, unsigned int, const char*) noexcept`.
Emscripten's **musl** declares it without a C++ exception spec and with `int` (not
`unsigned int`) line, so clang reports a mismatch.

**Workaround:** exclude Emscripten from that forward declaration
(`#if defined(NDEBUG) && !defined(__EMSCRIPTEN__)`), using the system decl.

### Attempt 3 — protobuf/ONNX cross-compile wall: wasm protoc can't run (blocker #4)

[`logs/14-torch-build-attempt3.log`](logs/14-torch-build-attempt3.log). c10 now
compiles past the assert. The build cross-compiles its own `protoc`, then tries to run
it to generate the ONNX `*.pb.cc/.h`:

```
[197/1374] Running C++ protocol buffer compiler on ...onnx_onnx_torch-ml.proto
FAILED: [code=126]
/bin/sh: 1: .../build/bin/protoc.js-3.13.0.0: Permission denied
```

The cross-built `protoc` is an **Emscripten artifact** — `bin/protoc.js-3.13.0.0` (a
78 KB JS launcher, no shebang, not executable) + `protoc.js-3.13.0.wasm` (1.3 MB
WASM) — so it cannot run as a host build tool. This is the classic host-vs-target
codegen problem: `protoc` must be built **for the host** while `libprotobuf` is built
for the target. `cmake/ProtoBuf.cmake` documents exactly this and honors
`CAFFE2_CUSTOM_PROTOC_EXECUTABLE`.

**Workaround:** the recipe now builds a **host** `protoc` from the *same vendored*
`third_party/protobuf` sources (version-matched libprotoc **3.13.0**) using the host
gcc in a clean env (so `emcc`/`pywasmcross` is not used —
[`logs/15-host-protoc-build.log`](logs/15-host-protoc-build.log)), then patches
`cmake/ProtoBuf.cmake` to set `CAFFE2_CUSTOM_PROTOC_EXECUTABLE` to it (torch's
`cmake.py` only forwards `BUILD_/USE_/CMAKE_*` env vars, so the value is injected via a
source patch, not an env var).

### Attempt 4 — ONNX + `libc10.so` build; ATen `SymInt * size_t` ambiguity (blocker #5)

[`logs/16-torch-build-attempt4.log`](logs/16-torch-build-attempt4.log). Host protoc
runs (`libprotoc 3.13.0`), ONNX protos are generated and compiled, `libonnx.a` is
built, and **`lib/libc10.so` links as a wasm shared library**
(`em++: warning: ignoring unsupported linker flag: -soname` — benign). Building
`torch_cpu` then fails:

```
aten/src/ATen/core/TensorBase.h:311:31: error: use of overloaded operator '*' is
  ambiguous (with operand types 'c10::SymInt' and 'size_t' (aka 'unsigned long'))
    return impl_->sym_numel() * impl_->itemsize();
```

On the 32-bit wasm target (ILP32) `size_t` is `unsigned long`, a *distinct* type from
the `int32/uint32/int64/uint64` operands for which `c10::SymInt` declares `operator*`.
Tellingly, upstream **already special-cases this for Apple** in `c10/core/SymInt.h`:

```cpp
// On OSX size_t is different than uint64_t so we have to define it separately
#if defined(__APPLE__)
DECLARE_SYMINT_OP_INTONLY(size_t, SymInt)
DECLARE_SYMINT_OP(size_t, SymInt)
#endif
```

**Workaround:** extend that guard to `#if defined(__APPLE__) || defined(__EMSCRIPTEN__)`.

### Attempt 5 — ~89% of `torch_cpu`, then `ssize_t` undeclared (blocker #6, stopping point)

[`logs/17-torch-build-attempt5.log`](logs/17-torch-build-attempt5.log) (2197 lines,
build ran **1166 s**). With the SymInt fix the build sails through all of ATen and most
of the JIT: it linked `lib/libprotobuf.a`, `lib/libcpuinfo.a`, `lib/libonnx.a`,
`lib/libc10.so`, and compiled to **ninja step 1219/1374 (~89% of `torch_cpu`)** with
zero errors, before:

```
torch/nativert/executor/ExecutionFrame.cpp -> c10/util/Enumerate.h:72:29:
  error: unknown type name 'ssize_t'; did you mean 'size_t'?
    using difference_type = ssize_t;
```

`c10/util/Enumerate.h` uses the POSIX type `ssize_t` (from `<sys/types.h>`) without
including it; it happens to be transitively available on glibc builds but not for this
translation unit under Emscripten musl. This is another one-line portability fix
(include `<sys/types.h>` or use `std::ptrdiff_t`), representative of the remaining tail.

**This is where the probe stops.** Continuing is mechanical whack-a-mole through more
32-bit/POSIX/Emscripten source fixes, after which the *real* headline walls from the
research doc apply at link time and were deliberately deferred here:

- **`-pthread`/`-matomics`/`-mbulk-memory` ABI consistency.** Note the compile lines in
  the logs show pyodide-build **stripped** the `-pthread -matomics -mbulk-memory` we
  set in `cflags/cxxflags` (they don't appear in the emitted `em++` commands), matching
  the research doc's note that pyodide-build discards `-pthread`. Because the probe
  forces single-threaded and `USE_XNNPACK=0`, this hasn't bitten yet, but a threaded /
  XNNPACK build would hit `--shared-memory is disallowed … not compiled with 'atomics'
  or 'bulk-memory'` unless the *entire* graph is compiled consistently (cf.
  pytorch/pytorch#176542).
- **`wasm-ld` + sleef/XNNPACK microkernels** (deferred via `USE_XNNPACK=0`).
- **Binary size** of a full `libtorch_cpu.so` (tens of MB), hostile to a browser.

### What this refines vs. the research doc

- Confirms **§3 (dependency graph / native linking)** concretely: protobuf/onnx *do*
  cross-compile to wasm static libs, and `libc10.so` links — the linking issue is
  specifically about host-vs-target codegen tools (protoc) and the eventual big
  `torch_cpu`/`torch_python` link, not "wasm-ld can't make a .so at all."
- Adds a blocker the doc doesn't emphasize: **no PyPI sdist**, plus **no target
  `libpython`** breaking `find_package(Python Development.Module)`/pybind11.
- Adds the concrete **32-bit/ILP32 source portability** class (`SymInt*size_t`,
  `ssize_t`) that must be ported before the threading/size walls are even reachable.

## Step 4 — `executorch` (optional): not attempted to build, with evidence why

[`logs/18-executorch-sdist-info.log`](logs/18-executorch-sdist-info.log):

```
executorch latest version: 1.5.0
release files for latest:  cp310..cp314 × {macosx_arm64, manylinux_{aarch64,x86_64}, win_amd64}  (all bdist_wheel)
versions WITH sdist: ['0.1.0', '0.1.2']
requires_dist: ... torchao<0.19,>=0.18.0 ...  numpy>=2.0.0 ... pandas>=2.2.2 ... pytorch-tokenizers>=1.5.0 ...
```

The realistic Python-importable target still hits the same class of blockers, plus a
new one:
- ExecuTorch 1.5.0 ships **only x86/arm binary wheels** — **no `wasm32-emscripten`
  wheel and no sdist**; only ancient `0.1.0/0.1.2` have sdists at all.
- Its Python distribution **depends on `torch`** (via `torchao`), i.e. it is gated on
  exactly the from-source `torch` blockers documented above, and adds a large
  ExecuTorch C++ build on top.

So a Pyodide `executorch` *pybindings* recipe is not tractable in this sandbox without
first landing a `torch` (or a torch-free ExecuTorch runtime) wasm build. The
emscripten-forge `executorch-cpp` runtime (see the research doc) remains the more
promising C++-only rung; the missing piece is still a Python-importable layer.

## Committed artifacts

- Recipe (final): [`torch-probe/meta.yaml`](torch-probe/meta.yaml) — points at the
  GitHub source tarball with correct sha256 and carries the four documented
  workarounds; header documents how far it builds and the current stop point.
- All raw logs: [`logs/`](logs/) (`00`–`18`), including the three full failing torch
  build logs (`12`,`13`,`14`,`16`,`17`) and the host-protoc build (`15`).
- Not committed (large/gitignored build trees): the wasm outputs produced during the
  run — e.g. `libc10.so` (a 153 KB WebAssembly module), `libprotobuf.a`,
  `libcpuinfo.a`, `libonnx.a`, and the host `protoc-3.13.0.0` — live under the
  out-of-tree recipes checkout and are described here rather than checked in.
