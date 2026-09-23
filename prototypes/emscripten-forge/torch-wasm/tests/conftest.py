"""Pytest conftest for running upstream PyTorch tests against the reduced
wasm32-emscripten torch build.

It skips ONLY tests that are fundamentally inapplicable to this runtime
(GPU/distributed/subprocess/threads/disabled-backends/cpp-extension/numpy-bridge/
OS-filesystem/slow-large-memory/profiler), driven by ``skip_manifest.json``.

Skip decisions are gated by *runtime capability probes* so the exact same
conftest is correct under both:

* the real wasm runtime (``sys.platform == 'emscripten'``), where the probes
  report the reduced build's true (missing) capabilities, and
* a host CPU torch used to validate the harness, where the probes report the
  host's capabilities (so almost nothing is skipped and we get a real baseline).

Set ``TORCH_WASM_SIMULATE=1`` to force the wasm capability values on a host
interpreter. This does NOT execute anything on wasm; it only reports how the
manifest *classifies* each test for the wasm runtime (skipped-inapplicable vs
applicable), which is what we commit as the skip ledger.

Genuine reduced-build failures are NOT handled here; record them in
``logs/known_failures.md``.
"""
from __future__ import annotations

import json
import os
import re
import sys
import collections
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_MANIFEST = json.loads((_HERE / "skip_manifest.json").read_text())

IS_WASM = sys.platform == "emscripten" or "wasm" in getattr(sys, "implementation", type("", (), {})()).__dict__.get("_multiarch", "")
FORCE_WASM = os.environ.get("TORCH_WASM_SIMULATE") == "1"
WASM_MODE = IS_WASM or FORCE_WASM


def _probe_host():
    """Detect the *host* capabilities (best-effort; failures => absent)."""
    caps = {}
    try:
        import torch
    except Exception:
        torch = None

    def _try(fn, default=False):
        try:
            return bool(fn())
        except Exception:
            return default

    caps["no_gpu"] = not (
        _try(lambda: torch.cuda.is_available())
        or _try(lambda: torch.backends.mps.is_available())
        or _try(lambda: torch.xpu.is_available())
    )
    caps["no_distributed"] = not _try(lambda: torch.distributed.is_available(), default=True)
    caps["no_subprocess"] = False
    caps["single_threaded"] = False
    # host CPU wheel ships mkldnn + a quantized engine (fbgemm/qnnpack)
    caps["no_backend_engines"] = not (
        _try(lambda: torch.backends.mkldnn.is_available())
        or _try(lambda: len(torch.backends.quantized.supported_engines) > 1)
    )
    caps["no_host_compiler"] = False
    # numpy bridge: works on host wheel, raises when torch built USE_NUMPY=0
    caps["no_numpy_bridge"] = not _try(lambda: (torch.zeros(1).numpy() is not None))
    caps["no_os_fs"] = False
    caps["constrained_heap"] = False
    caps["no_profiler"] = not _try(lambda: torch.autograd.profiler.profile is not None, default=True)
    return caps


# Under the wasm runtime every listed capability is absent by construction.
_WASM_CAPS = {
    "no_gpu": True,
    "no_distributed": True,
    "no_subprocess": True,
    "single_threaded": True,
    "no_backend_engines": True,
    "no_host_compiler": True,
    "no_numpy_bridge": True,
    "no_os_fs": True,
    "constrained_heap": True,
    "no_profiler": True,
}

CAPS = dict(_WASM_CAPS) if WASM_MODE else _probe_host()

# Precompile category matchers.
_CATS = []
for c in _MANIFEST["categories"]:
    rx = re.compile(c["nodeid_regex"]) if c.get("nodeid_regex") else None
    _CATS.append((c, rx))

_skip_tally = collections.Counter()
_skip_examples = collections.defaultdict(list)


def _match_category(nodeid: str):
    """Return the first active category that classifies this nodeid as
    inapplicable, or ``None``."""
    for c, rx in _CATS:
        if rx is None or not rx.search(nodeid):
            continue
        cap = c.get("capability")
        cap_absent = CAPS.get(cap, False) if cap else True
        gate_ok = (not c.get("wasm_only", False)) or WASM_MODE
        if cap_absent and gate_ok:
            return c
    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        cat = _match_category(item.nodeid)
        if cat is None:
            continue
        _skip_tally[cat["id"]] += 1
        if len(_skip_examples[cat["id"]]) < 5:
            _skip_examples[cat["id"]].append(item.nodeid)
        item.add_marker(
            pytest.mark.skip(
                reason=f"[wasm-inapplicable:{cat['id']}] {cat['reason']}"
            )
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    tr = terminalreporter
    tr.write_sep("=", "wasm-inapplicable skip classification")
    mode = "WASM" if IS_WASM else ("SIMULATED-WASM" if FORCE_WASM else "HOST-baseline")
    tr.write_line(f"mode: {mode}   platform: {sys.platform}")
    total = sum(_skip_tally.values())
    for cid, n in sorted(_skip_tally.items(), key=lambda kv: -kv[1]):
        tr.write_line(f"  {n:5d}  {cid}")
    tr.write_line(f"  {total:5d}  TOTAL classified-inapplicable")

    out = os.environ.get("TORCH_WASM_SKIPREPORT")
    if out:
        report = {
            "mode": mode,
            "platform": sys.platform,
            "capabilities": CAPS,
            "skipped_by_category": dict(_skip_tally),
            "total_classified_inapplicable": total,
            "examples": {k: v for k, v in _skip_examples.items()},
        }
        Path(out).write_text(json.dumps(report, indent=2))
        tr.write_line(f"skip report -> {out}")
