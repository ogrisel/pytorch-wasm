// Minimal "one dense layer" tensor kernel, compiled to wasm32-emscripten.
//
// This is deliberately dependency-free: it proves that a numerical C++ kernel of
// the kind that lives at the bottom of PyTorch/ExecuTorch (matmul + bias + ReLU)
// cross-compiles to WebAssembly and runs correctly under a JS runtime. It is the
// smallest honest "PyTorch on WASM" hello-world before pulling in the real runtime.

#include <cstddef>
#include <vector>
#include <emscripten/bind.h>

using emscripten::val;

// y = relu(x @ W^T + b)
//   x: [rows, in], W: [out, in], b: [out]  ->  y: [rows, out]
static std::vector<float> linear_relu(const std::vector<float>& x,
                                      const std::vector<float>& w,
                                      const std::vector<float>& b,
                                      int rows, int in_features,
                                      int out_features) {
  std::vector<float> y(static_cast<size_t>(rows) * out_features, 0.0f);
  for (int r = 0; r < rows; ++r) {
    for (int o = 0; o < out_features; ++o) {
      float acc = b[o];
      const float* xr = &x[static_cast<size_t>(r) * in_features];
      const float* wo = &w[static_cast<size_t>(o) * in_features];
      for (int k = 0; k < in_features; ++k) acc += xr[k] * wo[k];
      y[static_cast<size_t>(r) * out_features + o] = acc > 0.0f ? acc : 0.0f;
    }
  }
  return y;
}

// Thin glue: accept/return JS arrays so the demo is trivial to call from Node.
static val linear_relu_js(val x_in, val w_in, val b_in, int rows,
                          int in_features, int out_features) {
  const auto x = emscripten::convertJSArrayToNumberVector<float>(x_in);
  const auto w = emscripten::convertJSArrayToNumberVector<float>(w_in);
  const auto b = emscripten::convertJSArrayToNumberVector<float>(b_in);
  const auto y = linear_relu(x, w, b, rows, in_features, out_features);
  val out = val::array();
  for (size_t i = 0; i < y.size(); ++i) out.call<void>("push", y[i]);
  return out;
}

EMSCRIPTEN_BINDINGS(tensor_kernel) {
  emscripten::function("linearRelu", &linear_relu_js);
}
