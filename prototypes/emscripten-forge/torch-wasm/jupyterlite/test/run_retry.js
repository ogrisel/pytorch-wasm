// Usage: node run_retry.js <port> <notebookPath> <successRegex> [maxAttempts] [logfile]
const { chromium } = require('playwright');
const fs = require('fs');
const PORT = process.argv[2];
const NB = process.argv[3];
const SUCCESS = new RegExp(process.argv[4] || 'DIAG_DONE');
const MAX = parseInt(process.argv[5] || '8', 10);
const LOGF = process.argv[6] || '/tmp/run_retry.log';
const URL = `http://127.0.0.1:${PORT}/notebooks/index.html?path=${NB}`;
const line = s => { fs.appendFileSync(LOGF, s + '\n'); process.stdout.write(s + '\n'); };
(async () => {
  fs.writeFileSync(LOGF, '');
  const browser = await chromium.launch({ executablePath: '/usr/local/bin/google-chrome', headless: true, args: ['--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--js-flags=--max-old-space-size=8192'] });
  for (let attempt = 1; attempt <= MAX; attempt++) {
    const ctx = await browser.newContext({ viewport: { width: 1400, height: 1100 } });
    const p = await ctx.newPage();
    let xkernel = false;
    p.on('pageerror', e => { if (/XKernel/.test(e.message)) xkernel = true; });
    try {
      await p.goto(URL, { waitUntil: 'load', timeout: 120000 });
      await p.waitForSelector('.jp-Notebook', { timeout: 120000 });
      let ready = false;
      for (let i = 0; i < 45; i++) {
        await p.waitForTimeout(2000);
        const st = await p.evaluate(() => document.querySelector('.jp-Notebook-ExecutionIndicator')?.getAttribute('data-status'));
        if (st === 'idle') { ready = true; break; }
        if (xkernel) break;
      }
      if (!ready || xkernel) {
        line(`attempt ${attempt}: boot FAILED (xkernel=${xkernel} ready=${ready}) -> retry`);
        await ctx.close();
        continue;
      }
      line(`attempt ${attempt}: CLEAN BOOT (kernel idle, no XKernel). Running cells...`);
      try { await p.click('.jp-Notebook .jp-Cell', { timeout: 5000 }); } catch (e) {}
      const ran = await p.evaluate(async () => { if (window.jupyterapp?.commands) { await window.jupyterapp.commands.execute('notebook:run-all-cells'); return 'ran'; } return 'no'; });
      line('runAll cmd: ' + ran);
      // menu fallback to guarantee execution
      try { await p.click('text=Run', { timeout: 5000 }); await p.waitForTimeout(400); await p.click('text=Run All Cells', { timeout: 5000 }); line('menu ran'); } catch (e) { line('menu fallback skip: ' + e.message.split('\n')[0]); }
      const dl = Date.now() + parseInt(process.env.CELL_DEADLINE_MS || '300000', 10); let last = '';
      while (Date.now() < dl) {
        await p.waitForTimeout(4000);
        const s = await p.evaluate(() => Array.from(document.querySelectorAll('.jp-Notebook .jp-CodeCell')).map(c => (c.querySelector('.jp-InputPrompt')?.innerText?.trim()||'') + ' => ' + (c.querySelector('.jp-OutputArea')?.innerText?.trim()||'')).join('\n=====\n'));
        if (s !== last) { line('CELLS:\n' + s); last = s; }
        if (SUCCESS.test(s)) { line('*** SUCCESS MATCH ***'); break; }
      }
      const SHOT = '/workspace/prototypes/emscripten-forge/torch-wasm/logs/pw-retry.png';
      try { await p.screenshot({ path: SHOT, fullPage: true }); line('screenshot ' + SHOT); } catch (e) {}
      await ctx.close();
      await browser.close();
      process.exit(0);
    } catch (e) {
      line(`attempt ${attempt}: EXC ${e.message}`);
      await ctx.close();
    }
  }
  line('ALL ATTEMPTS EXHAUSTED — no clean boot');
  await browser.close();
  process.exit(2);
})().catch(e => { line('FATAL ' + e.stack); process.exit(1); });
