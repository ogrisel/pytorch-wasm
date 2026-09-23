// SINGLE-MODULE verifier: load ONE combined torch/_C.*.so side module (which
// whole-includes c10 + torch_cpu + torch + torch_python + deps) in Node+Pyodide,
// then import torch and train a tiny MLP. Tests whether the blocker #11
// static-initializer abort (mis-offset MEMORY_ADDR relocations) disappears once
// everything lives in a single module (no cross-.so relocations).
//
// Usage: node verify_single_module_node.mjs /abs/path/to/_C.*.so [/abs/path/to/torch-*.whl]
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";

const soPath = process.argv[2];
const wheelPath = process.argv[3];
if (!soPath) { console.error("usage: node verify_single_module_node.mjs <_C.so> [wheel]"); process.exit(2); }
Error.stackTraceLimit = 40;
const log = (...a) => console.log("[single]", ...a);

const py = await loadPyodide();
const M = py._module;
log("pyodide", py.version);

// Decode an Emscripten/C++ error: numbers are C++ exception pointers.
function describeErr(e) {
  if (typeof e === "number") {
    try { const m = M.getExceptionMessage(e); return `C++ exception: type=${m[0]} msg=${m[1]}`; }
    catch (_) { return `numeric error ${e} (getExceptionMessage failed)`; }
  }
  return String(e).slice(0, 6000);
}
await py.loadPackage(["micropip"]);
const micropip = py.pyimport("micropip");
await micropip.install(["typing-extensions", "sympy", "mpmath", "networkx", "jinja2", "fsspec", "filelock"]);
log("python deps installed");

const SP = py.runPython("import sys; f'/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'");

// If a wheel is given, unpack its Python tree so `import torch` finds the package.
if (wheelPath) {
  const bytes = readFileSync(wheelPath);
  py.unpackArchive(bytes.buffer, "whl", { extractDir: SP });
  log("wheel unpacked to", SP);
}

// Copy the freshly-linked single module into site-packages/torch/ (overwrites the
// wheel's _C stub if present).
const soBytes = readFileSync(soPath);
py.FS.mkdirTree(`${SP}/torch`);
py.FS.writeFile(`${SP}/torch/_C.cpython-312-wasm32-emscripten.so`, new Uint8Array(soBytes));
log("single module written:", soBytes.length, "bytes");

// Load the single side module. This runs ALL its C++ static initializers — the
// exact point where blocker #11 aborted for the multi-module build.
try {
  await py._api.loadDynlib(`${SP}/torch/_C.cpython-312-wasm32-emscripten.so`, true, [`${SP}/torch`, `${SP}/torch/lib`]);
  log("SINGLE MODULE LOADED OK — static initializers ran without abort");
} catch (e) {
  console.error("LOADLIB FAILED:", describeErr(e));
  if (e && e.stack) console.error(e.stack.split("\n").slice(0, 20).join("\n"));
  process.exit(4);
}

try {
  await py.runPythonAsync("import torch; print('[py] import torch OK', torch.__version__)");
  log("import torch OK");
} catch (e) { console.error("IMPORT FAILED:", describeErr(e)); process.exit(5); }

const code = `
import torch, torch.nn as nn
torch.manual_seed(0)
N=256
c0=torch.randn(N,2)*0.6+torch.tensor([-1.5,-1.5])
c1=torch.randn(N,2)*0.6+torch.tensor([ 1.5, 1.5])
X=torch.cat([c0,c1]); y=torch.cat([torch.zeros(N),torch.ones(N)]).long()
model=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,2))
loss_fn=nn.CrossEntropyLoss(); opt=torch.optim.SGD(model.parameters(),lr=0.1)
first=last=None
for e in range(60):
    opt.zero_grad(); out=model(X); loss=loss_fn(out,y); loss.backward(); opt.step()
    if e==0: first=float(loss)
    last=float(loss)
acc=(model(X).argmax(1)==y).float().mean().item()
print(f"first_loss={first:.4f} last_loss={last:.4f} acc={acc:.3f}")
assert last < first, "loss did not decrease"
print("OK: MLP trained on wasm torch")
`;
try {
  await py.runPythonAsync(code);
  log("training OK");
} catch (e) { console.error("TRAIN FAILED:", String(e).slice(0, 6000)); process.exit(6); }
