const { chromium } = require('playwright');
const fs=require('fs');
const URL='http://127.0.0.1:8124/notebooks/index.html?path=torch_check.ipynb';
const LOGF='/workspace/prototypes/emscripten-forge/torch-wasm/logs/52-torchcheck.log';
const line=s=>{fs.appendFileSync(LOGF,s+'\n');process.stdout.write(s+'\n');};
(async()=>{fs.writeFileSync(LOGF,'');
const b=await chromium.launch({executablePath:'/usr/local/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--js-flags=--max-old-space-size=8192']});
const p=await(await b.newContext({viewport:{width:1200,height:900}})).newPage();
p.on('pageerror',e=>line('[pageerror] '+e.message));
await p.goto(URL,{waitUntil:'load',timeout:120000});
await p.waitForSelector('.jp-Notebook',{timeout:120000});
let ready=false;for(let i=0;i<80;i++){await p.waitForTimeout(3000);const st=await p.evaluate(()=>document.querySelector('.jp-Notebook-ExecutionIndicator')?.getAttribute('data-status'));if(st==='idle'){ready=true;line('kernel idle '+i);break;}}
line('ready '+ready);
const ran = await p.evaluate(async () => { if (window.jupyterapp?.commands){ await window.jupyterapp.commands.execute('notebook:run-all-cells'); return 'ran';} return 'no'; });
line('runAll: ' + ran);
if (!ran.startsWith('ran')) { try{ await p.click('text=Run'); await p.waitForTimeout(400); await p.click('text=Run All Cells'); line('menu ran'); }catch(e){ line('menu fail '+e.message);} }
const dl=Date.now()+420000;let last='';
while(Date.now()<dl){await p.waitForTimeout(4000);const s=await p.evaluate(()=>Array.from(document.querySelectorAll('.jp-Notebook .jp-CodeCell')).map(c=>(c.querySelector('.jp-OutputArea')?.innerText?.trim()||'')).join('\n---\n'));if(s!==last){line('CELLS:\n'+s);last=s;}if(/TORCH_WASM_OK/.test(s)){line('*** OK ***');break;}if(/Traceback|Error/.test(s)){line('*** ERR ***');break;}}
await p.screenshot({path:'/workspace/prototypes/emscripten-forge/torch-wasm/logs/pw-09-torchcheck.png',fullPage:true});
await b.close();process.exit(0);})().catch(e=>{line('FATAL '+e.message);process.exit(1);});
