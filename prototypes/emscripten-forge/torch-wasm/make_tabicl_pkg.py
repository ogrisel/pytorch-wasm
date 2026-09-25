#!/usr/bin/env python3
"""Package TabICL + its PURE-PYTHON runtime deps (+ a psutil stub) into a
minimal emscripten-wasm32 conda package in the local channel, so
jupyterlite-xeus can pack it into the site FS.

Compiled deps that the emscripten-forge channel already provides
(numpy/scipy/scikit-learn/joblib/threadpoolctl/torch) and the un-portable
compiled psutil are EXCLUDED here; psutil is replaced by a tiny pure stub.

Usage:
  python make_tabicl_pkg.py --pip /path/to/host/pip
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile, time

CHANNEL = "/workspace/.torch-channel"
NAME, VERSION, BUILD = "tabicl-wasm", "2.2.0", "0"
SUBDIR = "emscripten-wasm32"
DEPENDS = ["python 3.13.*"]

# Provided by the emscripten-forge channel (do NOT bundle x86 pip copies).
CHANNEL_PROVIDED = {
    "numpy", "scipy", "sklearn", "scikit_learn", "scikit-learn", "joblib",
    "threadpoolctl", "torch", "sympy", "psutil",
}

PSUTIL_STUB = '''"""Minimal pure-Python psutil stub for the wasm runtime (TabICL only reads
virtual_memory().available/.total)."""
import collections
_svmem = collections.namedtuple("svmem", ["total", "available", "percent", "used", "free"])
_MEM = 1_800_000_000
def virtual_memory():
    return _svmem(_MEM, _MEM, 0.0, 0, _MEM)
def cpu_count(logical=True):
    return 1
class Process:
    def __init__(self, *a, **k):
        pass
    def memory_info(self):
        import types
        return types.SimpleNamespace(rss=0, vms=0)
__version__ = "0.0.0-wasm-stub"
'''

HF_STUB_INIT = '''"""Minimal huggingface_hub stub for the wasm runtime.

TabICL imports ``hf_hub_download`` at module load but only calls it when no
local ``model_path`` is provided. The demo bundles the checkpoint locally and
passes ``model_path=...``, so download is never invoked.
"""
__version__ = "0.0.0-wasm-stub"


def hf_hub_download(*args, **kwargs):
    raise RuntimeError(
        "huggingface_hub is stubbed in the wasm build; pass a local model_path "
        "and allow_auto_download=False (the checkpoint is bundled)."
    )


def snapshot_download(*args, **kwargs):
    raise RuntimeError("huggingface_hub is stubbed in the wasm build.")
'''

HF_STUB_UTILS = '''"""Minimal huggingface_hub.utils stub."""


class LocalEntryNotFoundError(Exception):
    pass
'''


def is_pure(top_dir):
    # exclude packages carrying compiled extensions
    for _r, _d, fs in os.walk(top_dir):
        for f in fs:
            if f.endswith((".so", ".pyd", ".dll", ".dylib")):
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pip", default=sys.executable.replace("python", "pip"))
    ap.add_argument("--stage-only", action="store_true")
    a = ap.parse_args()

    stage = tempfile.mkdtemp(prefix="tabicl-stage-")
    print("pip target:", stage)
    # Curated MINIMAL import-time closure for `from tabicl import
    # TabICLClassifier`. numpy/scipy/scikit-learn/joblib/torch come from the
    # emscripten-forge channel; psutil and huggingface_hub are stubbed (below).
    #
    # IMPORTANT: the full huggingface_hub dependency cluster (huggingface_hub +
    # requests + urllib3 + fsspec + filelock + certifi + idna +
    # charset_normalizer + pyyaml) CRASHES the xeus-python kernel at boot
    # ("XKernel is already registered"), even though it is pure Python and is
    # not imported at boot. Empirically bisected: dropping that cluster and
    # replacing huggingface_hub with a tiny stub makes the kernel boot. TabICL
    # only imports `hf_hub_download` / `LocalEntryNotFoundError` from it and
    # never calls them when a local `model_path` is supplied, so the stub is
    # sufficient. --no-deps so we never pull torch/CUDA/triton wheels.
    PURE = ["tabicl==2.2.0", "einops", "tqdm"]
    subprocess.check_call([a.pip, "install", "--quiet", "--no-deps",
                           "--target", stage] + PURE)

    # site-packages payload for the conda package
    pkg_root = "/workspace/.tabicl-stage/pkg"
    site = os.path.join(pkg_root, "lib", "python3.13", "site-packages")
    if os.path.exists(pkg_root):
        shutil.rmtree(pkg_root)
    os.makedirs(site, exist_ok=True)

    kept, skipped = [], []
    for entry in sorted(os.listdir(stage)):
        full = os.path.join(stage, entry)
        base = entry.split("-")[0].lower().replace(".dist", "")
        top = entry.split(".")[0].lower()
        if top in CHANNEL_PROVIDED or base in CHANNEL_PROVIDED:
            skipped.append(entry + " [channel-provided]")
            continue
        if entry.endswith(".dist-info") or entry.endswith(".data"):
            shutil.copytree(full, os.path.join(site, entry)) if os.path.isdir(full) else None
            kept.append(entry)
            continue
        if os.path.isdir(full):
            if not is_pure(full):
                # keep pure .py, drop compiled .so (rely on pure fallback)
                dst = os.path.join(site, entry)
                shutil.copytree(full, dst)
                nremoved = 0
                for r, _d, fs in os.walk(dst):
                    for f in fs:
                        if f.endswith((".so", ".pyd", ".dll", ".dylib")):
                            os.remove(os.path.join(r, f)); nremoved += 1
                kept.append(f"{entry} [stripped {nremoved} compiled]")
            else:
                shutil.copytree(full, os.path.join(site, entry))
                kept.append(entry)
        else:
            shutil.copy2(full, os.path.join(site, entry))
            kept.append(entry)

    # Patch tabicl to import ONLY the classifier. The regressor pulls
    # sklearn.ensemble / sklearn.tree / sklearn.model_selection, whose compiled
    # wasm extensions crash the xeus-python kernel at boot; the classifier path
    # needs only sklearn base/utils/preprocessing/neighbors.
    sk_init = os.path.join(site, "tabicl", "_sklearn", "__init__.py")
    if os.path.exists(sk_init):
        with open(sk_init, "w") as f:
            f.write('"""Classifier-only import for the wasm build (regressor pulls '
                    'sklearn.ensemble/tree/model_selection whose wasm .so crash '
                    'the kernel boot)."""\n')
            f.write("from .classifier import TabICLClassifier\n")
        print("patched tabicl/_sklearn/__init__.py -> classifier only")
    top_init = os.path.join(site, "tabicl", "__init__.py")
    if os.path.exists(top_init):
        txt = open(top_init).read()
        txt = txt.replace("from ._sklearn import TabICLClassifier, TabICLRegressor",
                          "from ._sklearn import TabICLClassifier")
        open(top_init, "w").write(txt)
        print("patched tabicl/__init__.py -> drop TabICLRegressor")

    # psutil stub (channel/compiled psutil excluded)
    with open(os.path.join(site, "psutil.py"), "w") as f:
        f.write(PSUTIL_STUB)
    kept.append("psutil.py [wasm stub]")

    # huggingface_hub stub: TabICL imports `hf_hub_download` and
    # `LocalEntryNotFoundError` at module load but never calls them when a local
    # model_path is supplied. The real package's dependency cluster crashes the
    # xeus kernel at boot, so ship a tiny stub instead.
    hf = os.path.join(site, "huggingface_hub")
    os.makedirs(os.path.join(hf, "utils"), exist_ok=True)
    with open(os.path.join(hf, "__init__.py"), "w") as f:
        f.write(HF_STUB_INIT)
    with open(os.path.join(hf, "utils", "__init__.py"), "w") as f:
        f.write(HF_STUB_UTILS)
    kept.append("huggingface_hub/ [wasm stub]")

    print("=== kept ===");  [print(" +", k) for k in kept]
    print("=== skipped (channel-provided) ===");  [print(" -", s) for s in skipped]

    if a.stage_only:
        print("stage-only; payload at", site)
        return

    # --- pack as emscripten-wasm32 conda package ---
    subdir_dir = os.path.join(CHANNEL, SUBDIR)
    os.makedirs(subdir_dir, exist_ok=True)
    os.makedirs(os.path.join(CHANNEL, "noarch"), exist_ok=True)
    rel = []
    for b, _d, fs in os.walk(pkg_root):
        for f in fs:
            rel.append(os.path.relpath(os.path.join(b, f), pkg_root))
    rel.sort()
    info = {"name": NAME, "version": VERSION, "build": BUILD, "build_number": int(BUILD),
            "subdir": SUBDIR, "platform": "emscripten", "arch": "wasm32",
            "depends": DEPENDS, "license": "BSD-3-Clause", "timestamp": int(time.time()*1000)}

    def path_entry(fn):
        data = open(os.path.join(pkg_root, fn), "rb").read()
        return {"_path": fn, "path_type": "hardlink",
                "sha256": hashlib.sha256(data).hexdigest(), "size_in_bytes": len(data)}
    paths = {"paths_version": 1, "paths": [path_entry(f) for f in rel]}

    build_dir = "/workspace/.tabicl-stage/pkgbuild"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    shutil.copytree(pkg_root, build_dir, symlinks=True)
    info_dir = os.path.join(build_dir, "info"); os.makedirs(info_dir, exist_ok=True)
    json.dump(info, open(os.path.join(info_dir, "index.json"), "w"), indent=2)
    json.dump(paths, open(os.path.join(info_dir, "paths.json"), "w"), indent=2)
    json.dump({"type": "generic"}, open(os.path.join(info_dir, "about.json"), "w"))

    fn = f"{NAME}-{VERSION}-{BUILD}.tar.bz2"
    out = os.path.join(subdir_dir, fn)
    with tarfile.open(out, "w:bz2") as tf:
        for f in rel:
            tf.add(os.path.join(build_dir, f), arcname=f)
        tf.add(info_dir, arcname="info")

    # merge into existing repodata (keep the torch entry)
    repofile = os.path.join(subdir_dir, "repodata.json")
    repo = json.load(open(repofile)) if os.path.exists(repofile) else {
        "info": {"subdir": SUBDIR}, "packages": {}, "packages.conda": {}, "repodata_version": 1}
    data = open(out, "rb").read()
    repo.setdefault("packages", {})[fn] = {**info, "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "md5": hashlib.md5(data).hexdigest(),
        "fn": fn, "subdir": SUBDIR}
    json.dump(repo, open(repofile, "w"), indent=2)
    print("wrote", out, len(data), "bytes")


if __name__ == "__main__":
    main()
