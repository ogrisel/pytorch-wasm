"""Runtime shims so TabICL runs on the reduced wasm32 torch build.

Import this BEFORE importing tabicl:

    import tabicl_wasm_shim  # noqa: F401  (applies psutil + numpy-bridge shims)
    from tabicl import TabICLClassifier

Two gaps in the reduced xeus-python wasm environment are bridged:

1. ``psutil`` is a compiled C extension that is not available for
   ``emscripten-wasm32``. TabICL imports it at module load
   (``tabicl._model.inference``) only to read
   ``psutil.virtual_memory().available/.total`` for out-of-core batch sizing.
   We install a tiny pure-Python stub reporting the browser JS heap budget so
   the import succeeds and small in-memory inference is used.

2. The wasm torch is built ``USE_NUMPY=0`` (no C++ numpy bridge), so
   ``torch.from_numpy`` and ``Tensor.numpy()`` raise. TabICL's sklearn wrapper
   converts every batch via ``torch.from_numpy(X).float()`` and returns
   ``out.cpu().numpy()``. We reimplement both through ``.tolist()`` round-trips,
   which do NOT use the numpy bridge. This is slower but correct for the small
   (hundreds of rows) datasets targeted here.

Idempotent and side-effect-guarded; safe to import twice.
"""
from __future__ import annotations

import sys
import types

_APPLIED = "_tabicl_wasm_shim_applied"


def _install_psutil_stub(mem_bytes: int = 1_800_000_000) -> None:
    if "psutil" in sys.modules:
        return
    try:
        import psutil  # noqa: F401  (real one present -> nothing to do)
        return
    except Exception:
        pass
    m = types.ModuleType("psutil")

    class _VM(tuple):
        # mimic psutil's svmem namedtuple surface used by tabicl
        @property
        def total(self):
            return mem_bytes

        @property
        def available(self):
            return mem_bytes

        @property
        def percent(self):
            return 0.0

        @property
        def used(self):
            return 0

        @property
        def free(self):
            return mem_bytes

    def virtual_memory():
        return _VM()

    def cpu_count(logical=True):
        return 1

    class Process:  # minimal
        def __init__(self, *a, **k):
            pass

        def memory_info(self):
            ni = types.SimpleNamespace(rss=0, vms=0)
            return ni

    m.virtual_memory = virtual_memory
    m.cpu_count = cpu_count
    m.Process = Process
    m.__version__ = "0.0.0-wasm-stub"
    sys.modules["psutil"] = m


def _patch_numpy_bridge() -> None:
    import numpy as np
    import torch

    if getattr(torch, _APPLIED, False):
        return

    _orig_from_numpy = getattr(torch, "from_numpy", None)

    def from_numpy(a):
        # Avoid the (disabled) C++ numpy bridge: go through Python lists.
        arr = np.asarray(a)
        t = torch.tensor(arr.tolist())
        # preserve integer/float kind roughly
        if arr.dtype.kind in ("i", "u"):
            t = t.to(torch.int64)
        elif arr.dtype.kind == "b":
            t = t.to(torch.bool)
        else:
            t = t.to(torch.float32)
        return t

    torch.from_numpy = from_numpy

    def _numpy(self, *a, **k):
        return np.array(self.detach().cpu().tolist())

    try:
        torch.Tensor.numpy = _numpy
    except Exception as e:  # pragma: no cover
        print("tabicl_wasm_shim: could not patch Tensor.numpy:", e)

    setattr(torch, _APPLIED, True)


def apply() -> None:
    _install_psutil_stub()
    try:
        _patch_numpy_bridge()
    except Exception as e:  # torch may not import in some contexts
        print("tabicl_wasm_shim: numpy-bridge patch deferred/failed:", e)


apply()
