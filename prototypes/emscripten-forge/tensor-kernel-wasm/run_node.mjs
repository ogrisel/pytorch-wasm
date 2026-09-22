// Load the WASM module and run one dense layer, comparing against a pure-JS
// reference so the run self-verifies. Exits non-zero on mismatch.
import createModule from "./tensor_kernel.mjs";

const rows = 2;
const inF = 3;
const outF = 2;

// x: [2,3], W: [2,3], b: [2]
const x = [1, 2, 3, -1, 0, 4];
const w = [0.5, -0.5, 1.0, -1.0, 2.0, 0.0];
const b = [0.1, -0.2];

function refLinearRelu(x, w, b, rows, inF, outF) {
  const y = new Array(rows * outF).fill(0);
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < outF; o++) {
      let acc = b[o];
      for (let k = 0; k < inF; k++) acc += x[r * inF + k] * w[o * inF + k];
      y[r * outF + o] = Math.max(0, acc);
    }
  }
  return y;
}

const Module = await createModule();
const got = Module.linearRelu(x, w, b, rows, inF, outF);
const want = refLinearRelu(x, w, b, rows, inF, outF);

const gotArr = [];
for (let i = 0; i < got.size ? got.size() : got.length; i++) {
  gotArr.push(got.get ? got.get(i) : got[i]);
}
// convertJSArrayToNumberVector round-trips as a JS array via push(), so `got`
// is already a plain array here; normalize just in case of embind vector return.
const result = Array.isArray(got) ? got : gotArr;

console.log("wasm linearRelu ->", result);
console.log("js   reference  ->", want);

let ok = result.length === want.length;
for (let i = 0; i < want.length && ok; i++) {
  if (Math.abs(result[i] - want[i]) > 1e-6) ok = false;
}

if (!ok) {
  console.error("MISMATCH between WASM kernel and JS reference");
  process.exit(1);
}
console.log("OK: WASM tensor kernel matches reference (PyTorch-style dense layer runs in WebAssembly)");
