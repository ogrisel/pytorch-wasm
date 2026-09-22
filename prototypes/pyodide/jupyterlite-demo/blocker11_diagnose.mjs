// Blocker #11 diagnostic harness (Pyodide 0.27.8, Node). Loads the torch wasm side
// modules and, when libtorch_cpu aborts, prints: the throw-site stack, the raw arguments
// to torchInternalAssertFail with a memory window, and the unresolved GOT.func/GOT.mem
// symbol sets. Requires the instrumentation applied by _patch.mjs.
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
const py = await loadPyodide();
const M = py._module;
await py.loadPackage(["micropip"]);
const micropip = py.pyimport("micropip");
await micropip.install(["typing-extensions", "sympy", "mpmath", "networkx", "jinja2", "fsspec", "filelock"]);
const SP = py.runPython("import sys; f'/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'");
py.unpackArchive(readFileSync(process.argv[2]).buffer, "whl", { extractDir: SP });
const SEARCH = [`${SP}/torch/lib`, `${SP}/torch`, `${SP}/functorch`];
const load = async (rel) => {
  try { await py._api.loadDynlib(`${SP}/${rel}`, true, SEARCH); console.log("loaded", rel); return true; }
  catch (e) {
    console.error("FAILED to load", rel);
    if (typeof e === "number") { const m = M.getExceptionMessage(e); console.error("  exception type:", m[0]); }
    return false;
  }
};
await load("torch/lib/libc10.so");
globalThis.__phase = "cpu";
await load("torch/lib/libtorch_cpu.so");

const GOT = M.GOT;
const classify = (set, label) => {
  const un = [];
  for (const k of set || []) { try { if (GOT[k] && GOT[k].value === 0 && GOT[k].required) un.push(k); } catch (_) {} }
  console.error(`### ${label}: requested=${set ? set.size : 0} unresolved=${un.length} ###`);
  if (un.length) console.error(un.join("\n"));
};
classify(globalThis.__gm, "GOT.mem (data symbols)");
classify(globalThis.__gf, "GOT.func (function pointers)");
