"""In-wasm pytest driver — runs inside the xeus-python wasm32 kernel.

This is the cell body executed by the JupyterLite notebook / the
``run_pytest_wasm.js`` Playwright harness. Under the wasm runtime
``sys.platform == 'emscripten'``, so ``conftest.py`` activates every
wasm-inapplicable skip automatically (no TORCH_WASM_SIMULATE needed).

It runs one upstream test file at a time (memory-bounded) and prints a
machine-parseable summary line the harness greps for.

Expected FS layout inside the kernel (populated by the JupyterLite build, see
RUN_IN_WASM.md):
    /drive/tests/conftest.py
    /drive/tests/skip_manifest.json
    /drive/tests/upstream/<test files + optim/ + autograd/>
"""
import json
import os
import sys

TESTS = os.environ.get("WASM_TESTS_DIR", "/drive/tests")
DEFAULT = ["test_type_promotion.py", "test_optim.py", "test_torch.py"]


def run(files=None):
    import pytest  # shipped via environment.yml

    files = files or DEFAULT
    os.chdir(TESTS)
    results = {}
    for f in files:
        target = os.path.join("upstream", f)
        rc = pytest.main(
            [target, "-q", "-p", "no:cacheprovider", "--no-header",
             "--tb=line", "-rN", f"--junitxml=/tmp/junit-{f}.xml"]
        )
        results[f] = {"pytest_rc": int(rc)}
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(f"/tmp/junit-{f}.xml").getroot()
            s = (root.findall(".//testsuite") or [root])[0]
            t = int(s.get("tests", 0)); fa = int(s.get("failures", 0))
            er = int(s.get("errors", 0)); sk = int(s.get("skipped", 0))
            results[f].update(total=t, failed=fa, error=er, skipped=sk,
                              passed=t - fa - er - sk)
        except Exception as e:
            results[f]["parse_error"] = str(e)
        print(f"WASM_PYTEST_FILE {f} {json.dumps(results[f])}", flush=True)

    print("WASM_PYTEST_SUMMARY " + json.dumps({
        "platform": sys.platform,
        "torch": __import__("torch").__version__,
        "results": results,
    }), flush=True)
    return results


if __name__ == "__main__":
    run(sys.argv[1:] or None)
