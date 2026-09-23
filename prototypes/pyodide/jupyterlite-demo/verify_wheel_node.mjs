// Load Pyodide 0.27.x in Node, install the local torch wasm wheel, and train a
// tiny MLP with autograd + SGD. Headless equivalent of the notebook, used as
// CI-style evidence that the wheel imports and trains.
//
// Usage: node verify_wheel_node.mjs /abs/path/to/torch-*.whl
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";

const wheelPath = process.argv[2];
if (!wheelPath) { console.error("usage: node verify_wheel_node.mjs <wheel>"); process.exit(2); }
Error.stackTraceLimit = 30;
const log = (...a) => console.log("[verify]", ...a);

const py = await loadPyodide();
log("pyodide", py.version);
await py.loadPackage(["micropip"]);
const micropip = py.pyimport("micropip");

// torch's pure-Python runtime deps (served from the Pyodide CDN).
await micropip.install(["typing-extensions", "sympy", "mpmath", "networkx", "jinja2", "fsspec", "filelock"]);
log("python deps installed");

// Unpack the torch wheel into site-packages, then load its wasm shared libraries
// in dependency order (micropip's auto-loader does not topologically sort them).
const SP = py.runPython("import sys; f'/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'");
const bytes = readFileSync(wheelPath);
py.unpackArchive(bytes.buffer, "whl", { extractDir: SP });
log("wheel unpacked to", SP);

const LIBS = [
  "torch/lib/libc10.so",
  "torch/lib/libtorch_cpu.so",
  "torch/lib/libtorch.so",
  "torch/lib/libshm.so",
  "torch/lib/libtorch_python.so",
  "torch/lib/libtorch_global_deps.so",
  "torch/_C.cpython-312-wasm32-emscripten.so",
  "functorch/_C.cpython-312-wasm32-emscripten.so",
];
const SEARCH = [`${SP}/torch/lib`, `${SP}/torch`, `${SP}/functorch`];
for (const rel of LIBS) {
  const p = `${SP}/${rel}`;
  try {
    await py._api.loadDynlib(p, true, SEARCH);
    log("loaded", rel);
  } catch (e) { console.error("LOADLIB FAILED", rel, String(e).slice(0, 3000)); process.exit(4); }
}

try {
  await py.runPythonAsync("import torch; print('[py] import torch OK', torch.__version__)");
  log("import torch OK");
} catch (e) { console.error("IMPORT FAILED:", String(e).slice(0, 4000)); process.exit(5); }

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
} catch (e) { console.error("TRAIN FAILED:", String(e).slice(0, 4000)); process.exit(6); }
