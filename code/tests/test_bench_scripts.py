"""Regressions for the benchmark scripts' row builders and instruments."""

from __future__ import annotations

import pytest
from ssj.bench_euler_passes import csv_row
from ssj.bench_interp import residual_log10
from ssj.benchmark import calibration, solve_compact


def test_euler_passes_row_reads_the_convergence_flag() -> None:
    """A solve that stopped on a non-finite step below the iteration cap is
    labeled unconverged from its ``converged`` flag.
    """
    cal = calibration(24, 20)
    out = {**solve_compact(cal), "converged": False, "iter": 5}
    assert csv_row("Coarse", 24, 20, "K1", out, cal)[4] == "no"


def test_residual_log10_scores_every_query_or_raises() -> None:
    """The off-grid instrument raises on a non-finite error and on a policy
    interpolated outside the grid.
    """
    cal = calibration(24, 20)
    out = solve_compact(cal)
    nan_c = out["c"].copy()
    nan_c[0, 5, 5] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        residual_log10(cal, {**out, "c": nan_c})
    with pytest.raises(RuntimeError, match="grid domain"):
        residual_log10(cal, {**out, "b": out["b"] + cal["b"][-1]})
