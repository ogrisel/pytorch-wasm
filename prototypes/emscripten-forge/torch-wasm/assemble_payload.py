#!/usr/bin/env python3
"""Assemble the importable wasm32 torch payload from build outputs.

Applies the Python-side runtime portability edits that the wasm32-emscripten
xeus-python kernel needs (the C++ source edits live in apply_patches.py, the
_C.so link fixes live in build_iter.sh). Idempotent: safe to re-run.

Inputs:
  --pkg-torch   the assembled torch/ site-packages dir (default: staged pkg)
  --src         the PyTorch source tree (for torch_version.py + torchgen)
  --c-so        the relinked torch/_C.so SIDE_MODULE (default: staged _C.so)

Steps:
  1. Drop the relinked _C.cpython-313-wasm32-emscripten.so into the payload.
  2. Guard `_load_global_deps()` on Emscripten (no shared libtorch to preload;
     symbols are already inside the single _C.so side module).
  3. Guard `_manager_path()` on Emscripten (no torch_shm_manager binary; the
     single-process wasm runtime has no shared-memory manager).
  4. Guard `multiprocessing.resource_tracker` import on Emscripten (wasm
     CPython has no _multiprocessing C module).
  5. Restore the real torch/torch_version.py from source (a build stub had
     replaced it, dropping the TorchVersion class that torch.utils._pytree
     imports).
  6. Ship the top-level `torchgen` pure-Python package (torch.utils
     ._python_dispatch imports torchgen / torchgen.model).
"""
import argparse, pathlib, shutil, sys

DEF_PKG = "/workspace/.torch-stage/pkg/lib/python3.13/site-packages/torch"
DEF_SRC = "/workspace/.src/pytorch"
DEF_CSO = "/workspace/.torch-stage/_C.so"
EXT = "_C.cpython-313-wasm32-emscripten.so"


def edit(path: pathlib.Path, old: str, new: str, tag: str) -> None:
    txt = path.read_text()
    if new in txt:
        print(f"[skip] {tag}: already applied")
        return
    if old not in txt:
        print(f"[WARN] {tag}: anchor not found")
        return
    path.write_text(txt.replace(old, new, 1))
    print(f"[ok]   {tag}: applied")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg-torch", default=DEF_PKG)
    ap.add_argument("--src", default=DEF_SRC)
    ap.add_argument("--c-so", default=DEF_CSO)
    a = ap.parse_args()

    torch_dir = pathlib.Path(a.pkg_torch)
    src = pathlib.Path(a.src)
    site = torch_dir.parent

    # 1) relinked _C.so
    cso = pathlib.Path(a.c_so)
    if cso.exists():
        dst = torch_dir / EXT
        if not dst.exists() or dst.stat().st_size != cso.stat().st_size:
            shutil.copy2(cso, dst)
            print(f"[ok]   _C.so: copied ({cso.stat().st_size} bytes)")
        else:
            print("[skip] _C.so: same size already in place")
    else:
        print(f"[WARN] _C.so not found at {cso}")

    init = torch_dir / "__init__.py"

    # 2) _load_global_deps guard
    edit(
        init,
        "    if USE_GLOBAL_DEPS:\n        _load_global_deps()",
        '    if USE_GLOBAL_DEPS and platform.system() != "Emscripten":\n        _load_global_deps()',
        "__init__: _load_global_deps Emscripten guard",
    )

    # 3) _manager_path guard
    edit(
        init,
        'def _manager_path():\n    if _running_with_deploy() or platform.system() == "Windows":\n        return b""',
        'def _manager_path():\n    if (\n        _running_with_deploy()\n        or platform.system() == "Windows"\n        or platform.system() == "Emscripten"\n    ):\n        return b""',
        "__init__: _manager_path Emscripten guard",
    )

    # 4) multiprocessing resource_tracker guard
    mp = torch_dir / "multiprocessing" / "__init__.py"
    edit(
        mp,
        "from multiprocessing.resource_tracker import ResourceTracker as _RT\n\n\nif (\n    sys.platform == \"darwin\"",
        'if sys.platform != "emscripten":\n    from multiprocessing.resource_tracker import ResourceTracker as _RT\nelse:\n    _RT = None\n\n\nif (\n    sys.platform == "darwin"',
        "multiprocessing: resource_tracker Emscripten guard",
    )

    # 5) restore real torch_version.py
    real_tv = src / "torch" / "torch_version.py"
    dst_tv = torch_dir / "torch_version.py"
    if real_tv.exists():
        if "class TorchVersion" not in dst_tv.read_text():
            shutil.copy2(real_tv, dst_tv)
            print("[ok]   torch_version.py: restored real one from source")
        else:
            print("[skip] torch_version.py: already the real one")

    # 6) torchgen package
    tg_src = src / "torchgen"
    tg_dst = site / "torchgen"
    if not (tg_dst / "__init__.py").exists():
        shutil.copytree(tg_src, tg_dst)
        for pc in tg_dst.rglob("__pycache__"):
            shutil.rmtree(pc, ignore_errors=True)
        print("[ok]   torchgen: copied into site-packages")
    else:
        print("[skip] torchgen: already present")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
