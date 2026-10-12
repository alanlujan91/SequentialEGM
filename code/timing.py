"""Wall-clock primitives shared by the benchmark and validation scripts.

Each helper takes a zero-argument or plain-call function and fixes only how it
is timed: the estimator built on top (best of N, mean with a confidence
interval, batched mean) stays with the caller.
"""

from __future__ import annotations

import time


def timed(fn, *args, **kwargs):
    """``(fn(*args, **kwargs), wall seconds)``."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def mean_call_time(fn, n=1):
    """Mean wall seconds per call of the zero-argument ``fn`` over one block of
    ``n`` back-to-back calls.
    """
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def call_times(fn, n):
    """Wall seconds of each of ``n`` separate calls of the zero-argument ``fn``."""
    return [timed(fn)[1] for _ in range(n)]
