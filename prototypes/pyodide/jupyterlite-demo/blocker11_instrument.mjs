import { readFileSync, writeFileSync } from "node:fs";
const f = "node_modules/pyodide/pyodide.asm.js";
let s = readFileSync("node_modules/pyodide/pyodide.asm.js.bak", "utf8");

// (1) Record GOT.mem / GOT.func symbols requested by side modules.
const a1 = '"GOT.mem":new Proxy({},GOTHandler),"GOT.func":new Proxy({},GOTHandler),"env":proxy';
const a1p = '"GOT.mem":new Proxy({},{get(o,p){if(typeof p==="string"){(globalThis.__gm||(globalThis.__gm=new Set())).add(p);}return GOTHandler.get(o,p);}}),"GOT.func":new Proxy({},{get(o,p){if(typeof p==="string"){(globalThis.__gf||(globalThis.__gf=new Set())).add(p);}return GOTHandler.get(o,p);}}),"env":proxy';

// (2) Throw-site stack (first throw during the cpu phase).
const a2 = "function ___cxa_throw(ptr,type,destructor){ptr>>>=0;type>>>=0;destructor>>>=0;";
const a2p = a2 + 'if(globalThis.__phase==="cpu"&&!globalThis.__tt){globalThis.__tt=1;console.error("### THROW STACK (first c10 throw during libtorch_cpu load) ###");console.error(new Error().stack);}';

// (3) Read the raw args + memory window passed to torchInternalAssertFail.
const a3 = 'if(prop in wasmImports&&!wasmImports[prop].stub){return wasmImports[prop]}';
const a3p = 'if(prop in wasmImports&&!wasmImports[prop].stub){var __f=wasmImports[prop];if(globalThis.__phase==="cpu"&&prop==="_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_NS0_22CompileTimeEmptyStringE"&&!globalThis.__aa){globalThis.__aa=1;var __w=wasmImports[prop];wasmImports[prop]=function(a,b,c,d){try{var win=function(p){var o="";for(var i=-8;i<52;i++){var ch=Module.HEAPU8[p+i];o+=(i===0?"[":"")+(ch>=32&&ch<127?String.fromCharCode(ch):".")+(i===0?"]":"");}return o;};console.error("### torchInternalAssertFail RAW ARGS ###");console.error("  func = "+JSON.stringify(Module.UTF8ToString(a)));console.error("  file = "+JSON.stringify(Module.UTF8ToString(b)));console.error("  line = "+c);console.error("  cond = "+JSON.stringify(Module.UTF8ToString(d)));console.error("### MEMORY WINDOW around each pointer ([]=pointer target) ###");console.error("  file@"+b+": "+win(b));console.error("  cond@"+d+": "+win(d));console.error("  func@"+a+": "+win(a));}catch(e){console.error("read err",String(e).slice(0,150));}return __w(a,b,c,d);};return wasmImports[prop];}return __f;}';

for (const [a, ap, name] of [[a1, a1p, "GOT recorders"], [a2, a2p, "throw stack"], [a3, a3p, "assert args"]]) {
  if (!s.includes(a)) { console.error("anchor missing:", name); process.exit(1); }
  s = s.replace(a, ap);
}
writeFileSync(f, s);
console.log("patched: GOT recorders + throw stack + assert args");
