// Headless-Chrome harness that runs the upstream PyTorch test files INSIDE the
// xeus-python wasm32 kernel in JupyterLite, via wasm_pytest_driver.py.
//
// Mirrors ../jupyterlite/test/run_torch.js. Prerequisites (see RUN_IN_WASM.md):
//   * environment.yml adds `pytest`, `expecttest`, `hypothesis`, `numpy` next
//     to `torch`, and the built wasm `torch` conda package is in the channel.
//   * tests/ (conftest.py, skip_manifest.json, upstream/**, wasm_pytest_driver.py)
//     is copied into the JupyterLite content FS under `tests/`.
//   * a notebook `pytest_wasm.ipynb` whose single cell runs:
//         import runpy, sys; sys.argv=['','test_type_promotion.py']
//         runpy.run_path('tests/wasm_pytest_driver.py', run_name='__main__')
//
// NOTE: this harness is committed but was NOT executed in the cloud run that
// produced RESULTS.md, because the wasm `torch/_C.so` module could not be built
// within the time budget on that VM. It is the intended path to real wasm
// results once the module is available.
const { chromium } = require('playwright');
const fs = require('fs');
const BASE = process.env.BASE || 'http://127.0.0.1:8123';
const NB = process.env.NB || 'pytest_wasm.ipynb';
const URL = `${BASE}/notebooks/index.html?path=${NB}`;
const OUT = process.env.OUT || '/workspace/prototypes/emscripten-forge/torch-wasm/tests/logs/wasm-pytest-run.log';
const line = (s) => { fs.appendFileSync(OUT, s + '\n'); process.stdout.write(s + '\n'); };

(async () => {
  fs.writeFileSync(OUT, '');
  const browser = await chromium.launch({
    executablePath: process.env.CHROME || '/usr/local/bin/google-chrome',
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
           '--js-flags=--max-old-space-size=8192'],
  });
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 1100 } });
  const page = await ctx.newPage();
  page.on('console', m => line(`[c.${m.type()}] ${m.text()}`));
  page.on('pageerror', e => line(`[pageerror] ${e.message}`));
  await page.goto(URL, { waitUntil: 'load', timeout: 120000 });
  await page.waitForTimeout(8000);
  await page.reload({ waitUntil: 'load', timeout: 120000 });
  await page.waitForSelector('.jp-Notebook', { timeout: 120000 });
  line('UI loaded; polling kernel status');
  let ready = false;
  for (let i = 0; i < 80; i++) {
    await page.waitForTimeout(3000);
    const st = await page.evaluate(() => document.querySelector('.jp-Notebook-ExecutionIndicator')?.getAttribute('data-status'));
    if (st === 'idle') { ready = true; line('kernel idle at poll ' + i); break; }
  }
  line('kernel ready=' + ready);
  await page.evaluate(async () => { if (window.jupyterapp?.commands) await window.jupyterapp.commands.execute('notebook:run-all-cells'); });
  const deadline = Date.now() + 1800000; // pytest in wasm is slow: allow 30 min
  let last = '';
  while (Date.now() < deadline) {
    await page.waitForTimeout(5000);
    const state = await page.evaluate(() => Array.from(document.querySelectorAll('.jp-Notebook .jp-CodeCell'))
      .map(c => c.querySelector('.jp-OutputArea')?.innerText?.trim() || '').join('\n=====\n'));
    if (state !== last) { line('--- @' + new Date().toISOString() + ' ---\n' + state); last = state; }
    if (/WASM_PYTEST_SUMMARY/.test(state)) { line('*** WASM PYTEST DONE ***'); break; }
    if (/Traceback|abort\(|Aborted|Dynamic linking error/.test(state)) { line('*** ERROR ***'); break; }
  }
  await page.screenshot({ path: OUT.replace(/\.log$/, '.png'), fullPage: true });
  await browser.close();
  process.exit(0);
})().catch(e => { line('FATAL ' + e.stack); process.exit(1); });
