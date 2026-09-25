#!/usr/bin/env python3
"""Package the assembled torch payload into a minimal emscripten-wasm32 conda
package + a local channel with repodata.json (consumable by jupyterlite-xeus).
"""
import json, os, sys, tarfile, hashlib, time, shutil

PKG_ROOT = "/workspace/.torch-stage/pkg"          # contains lib/python3.13/site-packages/torch
CHANNEL = "/workspace/.torch-channel"
NAME, VERSION, BUILD = "torch", "2.8.0", "0"
SUBDIR = "emscripten-wasm32"
DEPENDS = ["python 3.13.*"]

def files_under(root):
    out = []
    for base, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(base, f)
            out.append(os.path.relpath(p, root))
    return sorted(out)

def main():
    subdir_dir = os.path.join(CHANNEL, SUBDIR)
    os.makedirs(subdir_dir, exist_ok=True)
    os.makedirs(os.path.join(CHANNEL, "noarch"), exist_ok=True)

    rel_files = files_under(PKG_ROOT)
    info = {
        "name": NAME, "version": VERSION, "build": BUILD, "build_number": int(BUILD),
        "subdir": SUBDIR, "platform": "emscripten", "arch": "wasm32",
        "depends": DEPENDS, "license": "BSD-3-Clause",
        "timestamp": int(time.time() * 1000),
    }
    def _path_entry(f):
        full = os.path.join(PKG_ROOT, f)
        with open(full, "rb") as fh:
            data = fh.read()
        return {
            "_path": f,
            "path_type": "hardlink",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_in_bytes": len(data),
        }

    paths = {"paths_version": 1, "paths": [_path_entry(f) for f in rel_files]}

    # Stage info/ inside a temp build dir alongside the payload.
    build_dir = "/workspace/.torch-stage/pkgbuild"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    shutil.copytree(PKG_ROOT, build_dir, symlinks=True)
    info_dir = os.path.join(build_dir, "info")
    os.makedirs(info_dir, exist_ok=True)
    json.dump(info, open(os.path.join(info_dir, "index.json"), "w"), indent=2)
    json.dump(paths, open(os.path.join(info_dir, "paths.json"), "w"), indent=2)
    json.dump({"type": "generic"}, open(os.path.join(info_dir, "about.json"), "w"))

    fn = f"{NAME}-{VERSION}-{BUILD}.tar.bz2"
    out = os.path.join(subdir_dir, fn)
    with tarfile.open(out, "w:bz2") as tf:
        # files first, then info (conda convention)
        for f in rel_files:
            tf.add(os.path.join(build_dir, f), arcname=f)
        tf.add(info_dir, arcname="info")

    data = open(out, "rb").read()
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    rec = {**info, "size": len(data), "sha256": sha256, "md5": md5, "fn": fn,
           "subdir": SUBDIR}
    repodata = {
        "info": {"subdir": SUBDIR},
        "packages": {fn: rec},
        "packages.conda": {},
        "repodata_version": 1,
    }
    json.dump(repodata, open(os.path.join(subdir_dir, "repodata.json"), "w"), indent=2)
    json.dump({"info": {"subdir": "noarch"}, "packages": {}, "packages.conda": {},
               "repodata_version": 1},
              open(os.path.join(CHANNEL, "noarch", "repodata.json"), "w"), indent=2)
    print("wrote", out, len(data), "bytes")
    print("sha256", sha256)

if __name__ == "__main__":
    main()
