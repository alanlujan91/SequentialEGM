"""Numba compilation flags shared by the ENGINE kernels and the ``ssj`` kernels."""

from __future__ import annotations

# fastmath minus ``nnan``/``ninf``, so NaN/Inf sentinels stay reliable: full
# ``fastmath=True`` may fold ``np.isnan(out[i])`` to ``False``, so ``code/ssj/``
# compiles with these flags too.
SAFE_FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}
KERNEL_JIT = {
    "fastmath": SAFE_FASTMATH,
    "nogil": True,
    "cache": True,
    "error_model": "numpy",
}
