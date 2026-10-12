"""Regression: the in-loop interpolator harness shares the compact kernel.

The ENGINE configuration of ``solve_interp`` must reproduce
``ssj.benchmark.solve_compact`` (the harness's Phase 1 and Phase 3 are pinned to
the production kernel, so they cannot drift); the structured curvilinear
interpolant converges and agrees, and the Delaunay triangulation extrapolates off its
hull without producing NaN.
"""

from __future__ import annotations

import functools

import numpy as np
from numerics import rel_diff
from ssj.benchmark import calibration, solve_compact
from ssj.interp_loop import solve_interp

GRID = (24, 20)


@functools.cache
def interp_solved(interp: str) -> dict:
    """``solve_interp`` with ``interp`` at ``GRID``, solved once per session."""
    return solve_interp(calibration(*GRID), interp)


def test_engine_config_reproduces_solve_compact() -> None:
    """ENGINE harness == production kernel, BIT-identical (gate against any
    Phase 1/Phase 3 drift; a tolerance here once let sub-1e-9 drift pass).
    """
    eng = interp_solved("engine")
    ref = solve_compact(calibration(*GRID))
    assert eng["converged"]
    assert eng["iter"] == ref["iter"]
    for k in ("a", "b", "c", "Va", "Vb"):
        assert np.array_equal(eng[k], ref[k]), f"{k} drift from solve_compact"


def test_curvilinear_converges_and_agrees() -> None:
    """The structured curvilinear interpolant drives the same solve to agreement."""
    cur = interp_solved("curvilinear")
    eng = interp_solved("engine")
    assert cur["converged"]
    for k in ("a", "c"):
        d = np.median(rel_diff(cur[k], eng[k]))
        assert d < 1e-2, f"{k} median rel {d:.2e}"


def test_delaunay_stays_finite_off_hull() -> None:
    """The Delaunay realign extrapolates off its hull instead of returning NaN,
    so a short run stops only at its iteration cap. The full run never
    converges on this model, which bench_interp records.
    """
    out = solve_interp(calibration(*GRID), "delaunay", maxit=20)
    assert out["reason"] == "no_convergence"
    assert out["iter"] == 20
    assert np.isfinite(out["a"]).all()


def test_engine_reversed_converges_and_agrees() -> None:
    """The reversed pass order (warped axis first) still converges; with the Euler
    correction its converged policy agrees with the forward order closely (the order
    governs the worst-case guarantee, not converged accuracy; Section 4).
    """
    rev = interp_solved("engine_reversed")
    eng = interp_solved("engine")
    assert rev["converged"]
    d = np.median(rel_diff(rev["c"], eng["c"]))
    assert d < 1e-2, f"reversed vs forward median rel {d:.2e}"
