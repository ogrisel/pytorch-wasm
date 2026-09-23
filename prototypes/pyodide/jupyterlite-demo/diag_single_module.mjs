// Diagnose the std::bad_alloc thrown while loading the single combined torch/_C.so.
// Hooks ___cxa_throw to capture the throw stack + exception type, and wraps malloc
// to report any absurd allocation size requested during static initialization.
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";

const soPath = process.argv[2];
globalThis.__catch = true; // arm the patched pyodide.asm.js throw-stack hook
const py = await loadPyodide();
const M = py._module;
await py.loadPackage(["micropip"]);
const micropip = py.pyimport("micropip");
await micropip.install(["typing-extensions", "sympy", "mpmath", "networkx", "jinja2", "fsspec", "filelock"]);
const SP = py.runPython("import sys; f'/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'");

const soBytes = readFileSync(soPath);
py.FS.mkdirTree(`${SP}/torch`);
py.FS.writeFile(`${SP}/torch/_C.cpython-312-wasm32-emscripten.so`, new Uint8Array(soBytes));

// Hook ___cxa_throw to capture the first C++ throw during load.
let firstThrow = null;
const origThrow = M.___cxa_throw;
M.___cxa_throw = function (ptr, type, destructor) {
  if (!firstThrow) {
    let typeName = "?";
    try { typeName = M.getExceptionMessage(ptr)[0]; } catch (_) {}
    firstThrow = { stack: new Error().stack, typeName };
    console.error("### FIRST C++ THROW ###  type:", typeName);
    console.error(firstThrow.stack.split("\n").slice(1, 30).join("\n"));
  }
  return origThrow.apply(this, arguments);
};

// Wrap malloc to catch absurd sizes (a mis-read size field => huge request).
let bigAlloc = null;
const origMalloc = M._malloc;
if (origMalloc) {
  M._malloc = function (n) {
    if (n >>> 0 > 0x20000000 && !bigAlloc) { // > 512 MB
      bigAlloc = { n, stack: new Error().stack };
      console.error(`### HUGE malloc request: ${n} bytes (${(n/1048576).toFixed(1)} MB) ###`);
      console.error(bigAlloc.stack.split("\n").slice(1, 25).join("\n"));
    }
    return origMalloc.apply(this, arguments);
  };
}

try {
  await py._api.loadDynlib(`${SP}/torch/_C.cpython-312-wasm32-emscripten.so`, true, [`${SP}/torch`]);
  console.log("LOADED OK");
} catch (e) {
  let d = typeof e === "number" ? (() => { try { const m = M.getExceptionMessage(e); return `type=${m[0]} msg=${m[1]}`; } catch (_) { return `num ${e}`; } })() : String(e).slice(0, 2000);
  console.error("LOAD FAILED:", d);
}
