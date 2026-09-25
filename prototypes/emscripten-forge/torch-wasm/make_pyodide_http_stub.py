#!/usr/bin/env python3
"""Ship a no-op `pyodide-http` override into the local channel.

Upstream conda-forge `pyodide-http 0.2.2` crashes the xeus-python kernel at
import time: `xeus_python_shell.shell` calls `pyodide_http.patch_urllib()`,
whose `_streaming` module calls `to_js(..., dict_converter=...)`, a kwarg the
`pyjs-rt 3.2.0` runtime on emscripten-forge-dev does not accept -> TypeError ->
kernel never becomes ready ("XKernel is already registered" on the retry).

Because the local channel is listed first and channel priority is strict, a
`pyodide-http` package published here is selected over conda-forge's, replacing
`patch_urllib`/`patch_requests` with safe no-ops so the kernel boots. This does
not affect torch itself; it only disables the (browser-only) urllib->fetch shim.
"""
import json, os, tarfile, hashlib, time, io

CHANNEL = "/workspace/.torch-channel"
NAME, VERSION, BUILD = "pyodide-http", "0.2.2", "100"

STUB = '''"""No-op pyodide-http override (see make_pyodide_http_stub.py)."""
__version__ = "0.2.2"

def patch_urllib(*args, **kwargs):
    return None

def patch_requests(*args, **kwargs):
    return None

def patch(*args, **kwargs):
    return None

def should_patch():
    return False
'''

def main():
    noarch = os.path.join(CHANNEL, "noarch")
    os.makedirs(noarch, exist_ok=True)

    members = {
        "site-packages/pyodide_http/__init__.py": STUB.encode(),
    }
    index = {
        "name": NAME, "version": VERSION, "build": f"pyh_{BUILD}",
        "build_number": int(BUILD), "subdir": "noarch", "noarch": "python",
        "depends": ["python >=3.10"], "license": "MIT",
        "timestamp": int(time.time() * 1000),
    }
    link = {"noarch": {"type": "python"}, "package_metadata_version": 1}
    paths = {"paths_version": 1, "paths": [
        {"_path": p, "path_type": "hardlink",
         "sha256": hashlib.sha256(b).hexdigest(), "size_in_bytes": len(b)}
        for p, b in members.items()
    ]}

    fn = f"{NAME}-{VERSION}-pyh_{BUILD}.tar.bz2"
    out = os.path.join(noarch, fn)
    with tarfile.open(out, "w:bz2") as tf:
        def add_bytes(arc, data):
            ti = tarfile.TarInfo(arc); ti.size = len(data); ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(data))
        for p, b in members.items():
            add_bytes(p, b)
        add_bytes("info/index.json", json.dumps(index, indent=2).encode())
        add_bytes("info/paths.json", json.dumps(paths, indent=2).encode())
        add_bytes("info/link.json", json.dumps(link, indent=2).encode())
        add_bytes("info/about.json", json.dumps({"summary": "no-op pyodide-http"}).encode())

    data = open(out, "rb").read()
    rec = {**index, "size": len(data),
           "sha256": hashlib.sha256(data).hexdigest(),
           "md5": hashlib.md5(data).hexdigest(), "fn": fn}

    repo_path = os.path.join(noarch, "repodata.json")
    if os.path.exists(repo_path):
        repo = json.load(open(repo_path))
    else:
        repo = {"info": {"subdir": "noarch"}, "packages": {}, "packages.conda": {},
                "repodata_version": 1}
    repo["packages"][fn] = rec
    json.dump(repo, open(repo_path, "w"), indent=2)
    print("wrote", out, len(data), "bytes")

if __name__ == "__main__":
    main()
