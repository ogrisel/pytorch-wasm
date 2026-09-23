#!/usr/bin/env python3
"""Bundle the TabICL pretrained checkpoint into the JupyterLite site content.

The checkpoint (`tabicl-classifier-v2-*.ckpt`, ~110 MB) exceeds GitHub's 100 MB
per-file limit, so it is NOT committed. Instead we fetch it once (HuggingFace
``jingang/TabICL``) and copy it into ``jupyterlite/content/`` at build time.
jupyterlite-xeus then packs it into the site's virtual filesystem, where it is
served to the kernel on demand. The demo notebook points TabICL at this local
path with ``allow_auto_download=False`` so nothing is fetched at runtime.

Usage:
  python bundle_checkpoint.py [--python /path/to/host/python-with-hf]

If the checkpoint is already in the HuggingFace cache it is reused; otherwise it
is downloaded (requires network + `huggingface_hub`). The `.ckpt` under
`content/` is git-ignored via `jupyterlite/content/.gitignore`.
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO_ID = "jingang/TabICL"
CKPT = "tabicl-classifier-v2-20260212.ckpt"
CONTENT = os.path.join(os.path.dirname(__file__), "jupyterlite", "content")


def find_cached():
    import glob
    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/**/" + CKPT), recursive=True)
    hits += glob.glob("/workspace/.hf/**/" + CKPT, recursive=True)
    for h in hits:
        if os.path.exists(h):
            return os.path.realpath(h)
    return None


def download(python):
    code = (
        "from huggingface_hub import hf_hub_download;"
        f"p=hf_hub_download(repo_id='{REPO_ID}', filename='{CKPT}');"
        "print(p)"
    )
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/workspace/.hf")
    out = subprocess.check_output([python, "-c", code], env=env, text=True)
    return out.strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    src = find_cached()
    if not src:
        print("checkpoint not in cache; downloading via", a.python)
        src = download(a.python)
    print("checkpoint source:", src, os.path.getsize(src), "bytes")

    os.makedirs(CONTENT, exist_ok=True)
    dst = os.path.join(CONTENT, CKPT)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        print("already bundled:", dst)
    else:
        shutil.copy2(src, dst)
        print("bundled ->", dst, os.path.getsize(dst), "bytes")

    gi = os.path.join(CONTENT, ".gitignore")
    with open(gi, "w") as f:
        f.write("# The 110 MB TabICL checkpoint is bundled at build time, not committed.\n")
        f.write("*.ckpt\n")
    print("wrote", gi)


if __name__ == "__main__":
    main()
