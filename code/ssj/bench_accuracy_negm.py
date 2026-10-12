r"""Accuracy comparison behind the NEGM paragraph of Section 3.

Solves the two-asset household block with EGM$^n$ and with the NEGM
single-difference comparator at matched grids and tolerance (``SOLVE_TOL``, both
solvers' default), then records how far apart the converged policies sit and
how each scores against the model's own liquid Euler equation on the regular
grid. It writes ``accuracy_negm.csv``, and ``ssj.bench_timing``
(``timing_negm.csv``) times the same comparison.

Distances are median relative differences over the regular $(b, a)$ grid and
every income state. The Euler columns are median robust consumption-unit
residuals, which compare consumption against the smaller of the Euler-implied
level and the level at the binding liquid constraint, so no node is excluded:
``euler_median_*`` at the solver's own regular grid nodes, and
``euler_offgrid_median_log10_*`` (in $\\log_{10}$) at the off-grid interior
midpoints of ``ssj.bench_interp``, the reading behind Section 3's
within-fifteen-percent sentence. The ``(192, 160)`` row solves both methods
at the policy-benchmark grid itself; ``family`` labels each grid's role
(diagnostic, timing, or benchmark).

Run: ``PYTHONPATH=code uv run python -m ssj.bench_accuracy_negm``.
"""

from __future__ import annotations

import logging

from cli import start_logging
from engine.utilities import empirical_ipo_alpha

from ssj.bench_common import (
    INTERP_GRIDS,
    TIMING_GRIDS,
    log_done,
    write_csv,
)
from ssj.bench_interp import residual_log10
from ssj.benchmark import (
    assert_mesh_fold_free,
    assert_mesh_ipm,
    calibration,
    policy_distance,
    robust_euler_stats,
    solve_compact,
    solve_negm,
    stage0_mesh,
)

logger = logging.getLogger(__name__)

ACCURACY_GRIDS = [
    *(("diagnostic", nB, nA) for _, nB, nA in INTERP_GRIDS),
    *TIMING_GRIDS,
]


_HEADER = [
    "family",
    "nB",
    "nA",
    "iters_egmn",
    "iters_negm",
    "dist_b_median",
    "dist_a_median",
    "euler_median_egmn",
    "euler_median_negm",
    "euler_offgrid_median_log10_egmn",
    "euler_offgrid_median_log10_negm",
    "alpha_hat_max",
]


def solve_pair(grid):
    """``(cal, egmn, negm)``: both converged solves at ``grid``."""
    cal = calibration(*grid)
    return cal, solve_compact(cal), solve_negm(cal)


def measure(family, grid):
    """One CSV row at ``grid``."""
    return accuracy_row(family, grid, *solve_pair(grid))


def accuracy_row(family, grid, cal, egmn, negm):
    """One CSV row from the converged ``solve_pair(grid)`` solves."""
    mesh = stage0_mesh(egmn, cal)
    # The paper's four-corner and IPM claims cover every reported two-asset
    # mesh, the (192, 160) benchmark included.
    assert_mesh_fold_free(mesh["b_endo"], mesh["a_endo"])
    assert_mesh_ipm(mesh["b_endo"], mesh["a_endo"])
    b_endo = mesh["b_endo"]
    row = [
        family,
        *grid,
        egmn["iter"],
        negm["iter"],
        *(f"{policy_distance(egmn, cal, negm, cal, key):.6f}" for key in ("b", "a")),
        f"{robust_euler_stats(egmn, cal)[0]:.3e}",
        f"{robust_euler_stats(negm, cal)[0]:.3e}",
        f"{residual_log10(cal, egmn)[0]:.4f}",
        f"{residual_log10(cal, negm)[0]:.4f}",
        # Max bracket displacement of EGMn's own Stage-0 mesh over income
        # states, the paper's ex-post alpha-hat, measured by the same helper
        # as the health warp sweep.
        max(empirical_ipo_alpha(b) for b in b_endo),
    ]
    logger.info("%s %s: %s", family, grid, dict(zip(_HEADER[3:], row[3:], strict=True)))
    return row


def main() -> None:
    start_logging()
    rows = [measure(family, (n_b, n_a)) for family, n_b, n_a in ACCURACY_GRIDS]
    write_csv("accuracy_negm.csv", _HEADER, rows)
    log_done("accuracy_negm.csv")


if __name__ == "__main__":
    main()
