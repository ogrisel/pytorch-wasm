#!/usr/bin/env python3
"""Repackage the emscripten-forge scikit-learn as a PURE-PYTHON build.

The compiled emscripten-forge ``scikit-learn`` wasm package crashes the
xeus-python kernel at boot (the many preloaded ``.so`` extensions trigger a
"XKernel is already registered" abort during module init). TabICL only needs
scikit-learn's pure-python surface (``sklearn.base``, ``sklearn.utils``,
``sklearn.preprocessing.LabelEncoder``, ``sklearn.utils.multiclass``), so we
strip every compiled extension, replace ``sklearn.__check_build`` with a no-op,
and publish the result in the local channel with a higher build number so it is
selected over the channel's compiled build. Estimators that need the C
extensions will not work, but TabICL's sklearn usage does.

Usage:
  python make_sklearn_pure_pkg.py --src-tar /path/to/scikit-learn-*.tar.gz
"""
import argparse, glob, hashlib, json, os, shutil, tarfile, tempfile, time

CHANNEL = "/workspace/.torch-channel"
NAME, VERSION = "scikit-learn", "1.8.0"
BUILD = "pure_999"
SUBDIR = "emscripten-wasm32"

CHECK_BUILD_STUB = '''"""No-op sklearn.__check_build for the pure-python wasm bundle."""
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-tar", help="emscripten-forge scikit-learn .tar.gz to strip")
    ap.add_argument("--src-site", help="site-packages dir containing sklearn/ to strip")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--keep-so", default="",
                    help="comma-separated sklearn subpaths whose .so to KEEP "
                         "(e.g. 'utils,__check_build'); default strips all")
    a = ap.parse_args()
    keep = [k.strip() for k in a.keep_so.split(",") if k.strip()]

    work = tempfile.mkdtemp(prefix="skl-strip-")
    site = os.path.join(work, "lib", "python3.13", "site-packages")
    os.makedirs(site, exist_ok=True)
    if a.src_site:
        shutil.copytree(os.path.join(a.src_site, "sklearn"), os.path.join(site, "sklearn"))
    elif a.src_tar:
        with tarfile.open(a.src_tar) as tf:
            tf.extractall(work)
    else:
        raise SystemExit("need --src-site or --src-tar")

    skl = os.path.join(site, "sklearn")
    assert os.path.isdir(skl), skl

    removed = kept_so = 0
    for r, _d, fs in os.walk(skl):
        for f in fs:
            if f.endswith((".so", ".pyd", ".dll", ".dylib")):
                full = os.path.join(r, f)
                relp = os.path.relpath(full, skl)
                if keep and any(relp.startswith(k.rstrip("/") + "/") or relp.startswith(k)
                                for k in keep):
                    kept_so += 1
                    continue
                os.remove(full); removed += 1
    print(f"stripped {removed} compiled extensions from sklearn (kept {kept_so})")

    cb = os.path.join(skl, "__check_build", "__init__.py")
    if os.path.exists(os.path.dirname(cb)):
        with open(cb, "w") as f:
            f.write(CHECK_BUILD_STUB)
        print("stubbed sklearn/__check_build/__init__.py")

    pkg_root = "/workspace/.tabicl-stage/skl-pure"
    if os.path.exists(pkg_root):
        shutil.rmtree(pkg_root)
    os.makedirs(pkg_root)
    # keep only lib/ and any info-independent payload
    shutil.copytree(os.path.join(work, "lib"), os.path.join(pkg_root, "lib"))

    if a.stage_only:
        print("stage-only; payload at", os.path.join(pkg_root, "lib"))
        return

    subdir_dir = os.path.join(CHANNEL, SUBDIR)
    os.makedirs(subdir_dir, exist_ok=True)
    rel = []
    for b, _d, fs in os.walk(pkg_root):
        for f in fs:
            rel.append(os.path.relpath(os.path.join(b, f), pkg_root))
    rel.sort()
    info = {"name": NAME, "version": VERSION, "build": BUILD, "build_number": 999,
            "subdir": SUBDIR, "platform": "emscripten", "arch": "wasm32",
            "depends": ["python 3.13.*", "numpy", "scipy", "joblib", "threadpoolctl"],
            "license": "BSD-3-Clause", "timestamp": int(time.time() * 1000)}

    def path_entry(fn):
        data = open(os.path.join(pkg_root, fn), "rb").read()
        return {"_path": fn, "path_type": "hardlink",
                "sha256": hashlib.sha256(data).hexdigest(), "size_in_bytes": len(data)}
    paths = {"paths_version": 1, "paths": [path_entry(f) for f in rel]}

    build_dir = "/workspace/.tabicl-stage/skl-pkgbuild"
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
