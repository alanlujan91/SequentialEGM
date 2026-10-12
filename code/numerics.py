"""Numerical helpers shared by the solvers' benchmarks, tests and figures."""

import numpy as np


def rel_diff(x, ref):
    """Elementwise relative difference ``|x - ref| / |ref|``.

    Consumption is strictly positive, so there the ratio is always defined. An
    asset policy can sit exactly at its zero borrowing limit, where the ratio is
    ``0 / 0`` when ``x`` is there too: equal entries score ``0``, the exact
    agreement they are. A reference at zero with ``x`` off it scores ``inf``,
    since no finite relative error describes a disagreement over whether the
    constraint binds.
    """
    gap = np.abs(x - ref)
    with np.errstate(divide="ignore"):
        return np.divide(gap, np.abs(ref), out=np.zeros_like(gap), where=gap != 0)


def median_p95_max(r):
    """Median, 95th percentile and maximum of ``r``, as floats."""
    return float(np.median(r)), float(np.percentile(r, 95)), float(np.max(r))


def require_finite(**arrays) -> None:
    """Raise ``RuntimeError`` naming the first keyword array that holds a NaN or
    an infinity, with its count of such entries (``assert`` vanishes under ``-O``).
    """
    for name, arr in arrays.items():
        n_bad = int(np.sum(~np.isfinite(arr)))
        if n_bad:
            msg = f"{name} contains NaN/Inf: {n_bad} bad values"
            raise RuntimeError(msg)
