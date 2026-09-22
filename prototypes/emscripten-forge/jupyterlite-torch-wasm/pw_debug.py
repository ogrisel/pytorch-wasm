import time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/lab/index.html?path=mlp_training_demo.ipynb"
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--no-sandbox"])
    pg = b.new_page()
    logs = []
    pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text[:200]}"))
    pg.on("pageerror", lambda e: logs.append(f"[pageerror] {str(e)[:300]}"))
    pg.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    time.sleep(45)
    keys = pg.evaluate("() => Object.keys(window).filter(k => /jup|lab|lite|app/i.test(k))")
    print("WINDOW KEYS:", keys)
    print("TITLE:", pg.title())
    print("HAS CODECELL:", pg.query_selector(".jp-CodeCell") is not None)
    print("BODY SNIPPET:", pg.inner_text("body")[:500].replace("\n", " | "))
    pg.screenshot(path="/tmp/pw_debug.png", full_page=False)
    print("--- console (last 30) ---")
    for l in logs[-30:]:
        print(l)
    b.close()
