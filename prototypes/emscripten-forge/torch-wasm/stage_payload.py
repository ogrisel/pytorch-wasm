#!/usr/bin/env python3
"""Stage the importable wasm32 ``torch/`` site-packages payload from source.

The reduced cmake build (``build_iter.sh`` / ``.torch_rebuild.sh``) produces the
compiled ``torch/_C.so`` side module but does not run ``setup.py``, so the pure
Python ``torch/`` package tree is not laid down anywhere. This script assembles
it: it copies the source ``torch/`` package (minus the C++ sources and other
build-only trees that are irrelevant at import time), writes the generated
``torch/version.py`` that ``setup.py`` would otherwise emit, and leaves the rest
(``_C.so`` drop-in, Emscripten runtime guards, ``torch_version.py`` restore,
``torchgen``) to ``assemble_payload.py``.

Idempotent: re-running refreshes the tree in place.

Usage:
  python stage_payload.py [--src /workspace/.src/pytorch]
                          [--pkg /workspace/.torch-stage/pkg]
                          [--version 2.8.0]
"""
import argparse
import os
import shutil

# Trees under torch/ that are build-only (C++ sources / headers) or otherwise
# never imported by the reduced runtime; skipped to keep the payload small.
EXCLUDE_TOP = {"csrc", "include"}
EXCLUDE_NAMES = {"__pycache__"}
EXCLUDE_SUFFIX = (".o", ".a", ".obj", ".cpp", ".cc", ".h", ".cu", ".cuh")


def _ignore(dirpath, names):
    drop = set()
    for n in names:
        if n in EXCLUDE_NAMES:
            drop.add(n)
        elif n.endswith(EXCLUDE_SUFFIX):
            drop.add(n)
    return drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/.src/pytorch")
    ap.add_argument("--pkg", default="/workspace/.torch-stage/pkg")
    ap.add_argument("--version", default="2.8.0")
    a = ap.parse_args()

    src_torch = os.path.join(a.src, "torch")
    site = os.path.join(a.pkg, "lib", "python3.13", "site-packages")
    dst_torch = os.path.join(site, "torch")
    os.makedirs(site, exist_ok=True)

    if os.path.exists(dst_torch):
        shutil.rmtree(dst_torch)
    os.makedirs(dst_torch)

    copied = 0
    for entry in sorted(os.listdir(src_torch)):
        if entry in EXCLUDE_TOP or entry in EXCLUDE_NAMES:
            continue
        s = os.path.join(src_torch, entry)
        d = os.path.join(dst_torch, entry)
        if os.path.isdir(s):
            shutil.copytree(s, d, ignore=_ignore)
        else:
            if entry.endswith(EXCLUDE_SUFFIX):
                continue
            shutil.copy2(s, d)
        copied += 1
    print(f"[ok] copied {copied} top-level entries from {src_torch}")

    # setup.py normally generates torch/version.py; write the minimal runtime one.
    with open(os.path.join(dst_torch, "version.py"), "w") as f:
        f.write("__version__ = %r\n" % a.version)
        f.write("debug = False\n")
        f.write("cuda = None\n")
        f.write("git_version = 'unknown'\n")
        f.write("hip = None\n")
        f.write("xpu = None\n")
    print("[ok] wrote torch/version.py")
    print("staged payload at", dst_torch)


if __name__ == "__main__":
    main()
