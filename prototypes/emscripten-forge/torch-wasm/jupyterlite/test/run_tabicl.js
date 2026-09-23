const { chromium } = require('playwright');
const fs = require('fs');
const BASE = 'http://127.0.0.1:8123';
const URL = `${BASE}/notebooks/index.html?path=tabicl_demo.ipynb`;
const SHOTS = '/workspace/prototypes/emscripten-forge/torch-wasm/logs';
const LOGF = `${SHOTS}/50-tabicl-run.log`;
const line = (s) => { fs.appendFileSync(LOGF, s + '\n'); process.stdout.write(s + '\n'); };
(async () => {
  fs.writeFileSync(LOGF, '');
  const browser = await chromium.launch({ executablePath: '/usr/local/bin/google-chrome', headless: true, args: ['--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--js-flags=--max-old-space-size=8192'] });
  const ctx = await browser.newContext({ viewport: { width: 1400, height: 1400 } });
  const page = await ctx.newPage();
  page.on('console', m => { const t=m.text(); if(/collides|duplicated|Uploader|scratchpad|removed socket/.test(t))return; line(`[c.${m.type()}] ${t}`); });
  page.on('pageerror', e => line(`[pageerror] ${e.message}`));
  page.on('worker', w => line(`[worker] ${w.url()}`));
  // Single load, NO reload (a reload re-instantiates the xeus kernel in the
  // persisted SW-controlled worker -> "XKernel is already registered").
  line('goto (single load, no reload)');
  await page.goto(URL, { waitUntil: 'load', timeout: 120000 });
  await page.waitForSelector('.jp-Notebook', { timeout: 120000 });
  line('UI loaded; polling kernel status (long boot allowed)');
  let ready=false;
  for (let i=0;i<120;i++){
    await page.waitForTimeout(3000);
    const st = await page.evaluate(() => document.querySelector('.jp-Notebook-ExecutionIndicator')?.getAttribute('data-status'));
    if (i%5===0) line(`kstatus[${i}] ind=${st}`);
    if (st === 'idle') { ready=true; line('kernel idle at poll '+i); break; }
  }
  line('kernel ready=' + ready);
  const ran = await page.evaluate(async () => { if (window.jupyterapp?.commands){ await window.jupyterapp.commands.execute('notebook:run-all-cells'); return 'ran';} return 'no'; });
  line('runAll: ' + ran);
  if (!ran.startsWith('ran')) { try{ await page.click('text=Run'); await page.waitForTimeout(400); await page.click('text=Run All Cells'); line('menu ran'); }catch(e){ line('menu fail '+e.message);} }
  const deadline = Date.now() + 1200000; let last='';
  while (Date.now() < deadline) {
    await page.waitForTimeout(5000);
    const state = await page.evaluate(() => Array.from(document.querySelectorAll('.jp-Notebook .jp-CodeCell')).map(c => (c.querySelector('.jp-InputPrompt')?.innerText?.trim()||'')+' => '+(c.querySelector('.jp-OutputArea')?.innerText?.trim()||'')).join('\n=====\n'));
    if (state!==last){ line('--- cells @'+new Date().toISOString()+' ---\n'+state); last=state; }
    if (/TABICL SUCCESS/i.test(state)){ line('*** TABICL SUCCESS ***'); break; }
    if (/TABICL BLOCKER/i.test(state)){ line('*** TABICL BLOCKER ***'); break; }
    if (/abort\(|Aborted\(|what\(\):|c10::Error/i.test(state)){ line('*** HARD ERROR in cell ***'); break; }
  }
  await page.screenshot({ path: `${SHOTS}/pw-08-tabicl.png`, fullPage: true });
  line('DONE'); await browser.close(); process.exit(0);
})().catch(e => { line('FATAL '+e.stack); process.exit(1); });
