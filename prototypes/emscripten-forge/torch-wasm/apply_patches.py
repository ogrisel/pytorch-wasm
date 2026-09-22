#!/usr/bin/env python3
"""Apply wasm32 (ILP32) + Emscripten musl portability edits to a PyTorch tree.

Idempotent: safe to run repeatedly. Prints what it changed.
"""
import sys, pathlib

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/.src/pytorch")

def patch(rel, old, new):
    p = root / rel
    txt = p.read_text()
    if new in txt:
        print(f"[skip] {rel}: already patched")
        return
    if old not in txt:
        print(f"[WARN] {rel}: anchor not found:\n{old!r}")
        return
    p.write_text(txt.replace(old, new, 1))
    print(f"[ok]   {rel}: patched")

# 1) SymInt * size_t is only declared for __APPLE__ (where size_t != uint64_t).
#    wasm32 is ILP32 so size_t is a distinct 32-bit type there too; extend the
#    guard to __EMSCRIPTEN__ to avoid ambiguous-overload errors in ATen.
patch(
    "c10/core/SymInt.h",
    "#if defined(__APPLE__)\nDECLARE_SYMINT_OP_INTONLY(size_t, SymInt)",
    "#if defined(__APPLE__) || defined(__EMSCRIPTEN__)\nDECLARE_SYMINT_OP_INTONLY(size_t, SymInt)",
)

# 2) c10 forward-declares glibc's __assert_fail with `noexcept` under NDEBUG.
#    Emscripten's musl declares it differently -> conflicting/ambiguous decl.
#    Skip that forward declaration on Emscripten.
patch(
    "c10/macros/Macros.h",
    "#else // __APPLE__, _MSC_VER\n#if defined(NDEBUG)\nextern \"C\" {",
    "#else // __APPLE__, _MSC_VER\n#if defined(NDEBUG) && !defined(__EMSCRIPTEN__)\nextern \"C\" {",
)

# 3) c10/util/Enumerate.h uses ssize_t but only pulls it in for _WIN32. On
#    Emscripten musl ssize_t needs <sys/types.h>.
patch(
    "c10/util/Enumerate.h",
    "#ifdef _WIN32\n#include <basetsd.h> // @manual\nusing ssize_t = SSIZE_T;\n#endif",
    "#ifdef _WIN32\n#include <basetsd.h> // @manual\nusing ssize_t = SSIZE_T;\n#endif\n#ifdef __EMSCRIPTEN__\n#include <sys/types.h> // ssize_t\n#endif",
)

# 4) Skip install(EXPORT Caffe2Targets): with the reduced feature set several
#    private static deps of torch_cpu (fp16, flatbuffers, onnx_library, the
#    ATEN_CPU_FILES_GEN_LIB custom target) are not in any export set, so the
#    CMake package-config export fails at generate time. We only need to
#    *compile* torch_cpu, so disable that export block.
patch(
    "CMakeLists.txt",
    "  if(NOT BUILD_LIBTORCHLESS)\n    install(\n      EXPORT Caffe2Targets",
    "  if(FALSE)  # emscripten probe: skip Caffe2Targets export\n    install(\n      EXPORT Caffe2Targets",
)

print("done")
