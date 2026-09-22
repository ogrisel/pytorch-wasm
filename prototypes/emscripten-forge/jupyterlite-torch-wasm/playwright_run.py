"""Headless-browser end-to-end test of the JupyterLite site.

Opens the demo notebook in the WASM xeus-python kernel, runs all cells (select-all +
Shift+Enter), asserts the training output appears, and saves a screenshot artifact.
Terminal-driven; no computer-use executor required.
"""
import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/lab/index.html?path=mlp_training_demo.ipynb"
SHOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/cursor/artifacts/jupyterlite_mlp_demo.png"
TIMEOUT_S = 260


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": 1400, "height": 1100})
        page = ctx.new_page()
        logs = []
        page.on("console", lambda m: logs.append(f"[{m.type}] {m.text[:200]}"))
        page.on("pageerror", lambda e: logs.append(f"[pageerror] {str(e)[:300]}"))

        print(f"goto {URL}")
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector(".jp-Notebook .jp-CodeCell", timeout=120_000)
        print("notebook cells rendered")

        # Wait for the xeus-python WASM kernel to connect and go idle.
        kernel_ready = False
        t0 = time.time()
        while time.time() - t0 < 120:
            status = page.evaluate(
                "() => { const e = document.querySelector('.jp-Notebook-ExecutionIndicator');"
                " return e ? e.getAttribute('data-status') : null; }"
            )
            if status == "idle":
                kernel_ready = True
                break
            time.sleep(2)
        print(f"kernel idle: {kernel_ready} after {time.time()-t0:.0f}s")

        def run_all():
            page.click(".jp-Notebook", timeout=10_000)
            page.keyboard.press("Escape")
            page.keyboard.press("Control+a")
            page.keyboard.press("Shift+Enter")

        run_all()
        print("issued select-all + Shift+Enter")

        deadline = time.time() + TIMEOUT_S
        got = False
        last_reissue = time.time()
        while time.time() < deadline:
            body = page.inner_text("body")
            if "correlation" in body and "final loss" in body:
                got = True
                print("training outputs detected")
                break
            # Re-issue occasionally in case the first run predated kernel readiness.
            if time.time() - last_reissue > 40:
                try:
                    run_all()
                    print("re-issued run-all")
                except Exception as e:  # noqa
                    logs.append(f"[reissue-error] {str(e)[:120]}")
                last_reissue = time.time()
            time.sleep(3)

        body = page.inner_text("body")
        try:
            page.screenshot(path=SHOT, full_page=True)
            print(f"screenshot -> {SHOT}")
        except Exception as e:  # noqa
            print("screenshot error", e)

        for key in ["microtorch 0", "running on", "final loss", "correlation", "epoch 199"]:
            for line in body.splitlines():
                if key in line:
                    print("OUT:", line.strip())
                    break

        print("\n--- last 20 console logs ---")
        for l in logs[-20:]:
            print(l)

        browser.close()
        if not got:
            print("\nRESULT: outputs NOT fully detected")
            sys.exit(2)
        print("\nRESULT: OK - notebook trained an MLP in the WASM kernel")


if __name__ == "__main__":
    main()
