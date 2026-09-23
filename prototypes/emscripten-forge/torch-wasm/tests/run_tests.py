#!/usr/bin/env python3
"""Driver that runs the upstream PyTorch core test files against the reduced
torch build, applying the wasm-inapplicable skip manifest (via conftest.py),
and reports per-file passed / skipped / failed / error counts.

Modes (env ``TORCH_WASM_SIMULATE``):
  * unset  -> HOST baseline: run on host CPU torch; only genuinely-absent host
              capabilities (e.g. GPU) are skipped. Proves the harness + the
              upstream tests actually execute end-to-end.
  * "1"    -> SIMULATED-WASM classification: force the wasm capability values so
              the manifest skips every test the reduced wasm runtime cannot run;
              the remaining "applicable" subset is what a real wasm run targets.

Usage:
  python run_tests.py [--files test_optim.py ...] [--out logs/results.json]

The same runner is what the in-wasm harness (see run_pytest_wasm.js /
RUN_IN_WASM.md) invokes under xeus-python; there ``sys.platform`` is
``emscripten`` so the skips apply automatically without simulation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
UPSTREAM = HERE / "upstream"
DEFAULT_FILES = [
    "test_type_promotion.py",
    "test_optim.py",
    "test_autograd.py",
    "test_nn.py",
    "test_torch.py",
    "test_ops.py",
]


def run_one(pyexe: str, f: str, timeout: int):
    junit = HERE / "logs" / f"junit-{f}.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        pyexe, "-m", "pytest", str(UPSTREAM / f),
        "-p", "no:cacheprovider", "-q", "--no-header",
        "--tb=line", "-rN",
        f"--junitxml={junit}",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True,
                              text=True, timeout=timeout)
        tail = proc.stdout[-1500:] + proc.stderr[-800:]
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        tail = (e.stdout or "")[-1500:] + "\n[TIMEOUT]"
        rc = -9

    counts = {"passed": 0, "skipped": 0, "failed": 0, "error": 0, "total": 0}
    if junit.exists():
        try:
            root = ET.parse(junit).getroot()
            suites = root.findall(".//testsuite") or [root]
            for s in suites:
                counts["total"] += int(s.get("tests", 0))
                counts["failed"] += int(s.get("failures", 0))
                counts["error"] += int(s.get("errors", 0))
                counts["skipped"] += int(s.get("skipped", 0))
            counts["passed"] = (counts["total"] - counts["failed"]
                                - counts["error"] - counts["skipped"])
        except Exception as ex:
            counts["parse_error"] = str(ex)
    return rc, counts, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default=str(HERE / "logs" / "results.json"))
    args = ap.parse_args()

    import os
    mode = "SIMULATED-WASM" if os.environ.get("TORCH_WASM_SIMULATE") == "1" else "HOST-baseline"
    if sys.platform == "emscripten":
        mode = "WASM"

    results = {}
    print(f"# mode={mode} python={args.python}")
    for f in args.files:
        if not (UPSTREAM / f).exists():
            print(f"!! missing {f}"); continue
        print(f"\n==== {f} ====", flush=True)
        rc, counts, tail = run_one(args.python, f, args.timeout)
        results[f] = {"rc": rc, **counts}
        print(f"   {counts}")
        (HERE / "logs" / f"out-{f}.log").write_text(tail)

    agg = {"passed": 0, "skipped": 0, "failed": 0, "error": 0, "total": 0}
    for c in results.values():
        for k in agg:
            agg[k] += c.get(k, 0)

    summary = {
        "mode": mode,
        "generated": datetime.now(timezone.utc).isoformat(),
        "python": args.python,
        "platform": sys.platform,
        "per_file": results,
        "aggregate": agg,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n==== AGGREGATE ====")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
