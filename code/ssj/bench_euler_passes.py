r"""Regular-grid Euler inversion pass sweep: regenerates ``euler_passes_abrs.csv``.

Reproduces the numbers Online Appendix B reports for the Stage-0 regular-grid
Euler inversion, the pass that recovers liquid consumption at the regular
$(\bRat, \aRat)$ grid by inverting the Euler equation against the interpolated
continuation marginal value (see ``ssj.hh_egm``). The inversion is the defining
EGM operation applied at the regular grid; the pass count $K$ is how many times
the closed-form map is applied against the $\cRat \leftrightarrow \bRat'$
coupling (one by default). $K = 0$ (skip the inversion, read consumption off the
interpolated policy) is the shortcut EGM exists to avoid and is never a production
configuration; it is solved here only to document why the inversion is necessary.

For the ENGINE configuration this solves $K \in \{0, 1, 2\}$ and the nested
method (NEGM) at three grids, and reports two residual families of the converged
policy:

- on-grid (``ssj.benchmark.robust_euler_stats``): the robust consumption-unit
  Euler residual at the solver's own grid nodes;
- off-grid (``ssj.bench_interp.residual_log10``): the same robust residual at
  the dense interior test set that governs @tab-solvetime-abrs, reported as
  $\log_{10}$ (median, p95, max).

The paper's readings follow directly from the CSV: one pass already reaches the
off-grid accuracy the grid allows, matching NEGM to within about a factor of
$1.5$ (the curvilinear interpolant is covered by ``ssj.bench_interp``); the
second pass drives the on-grid residual to about $10^{-6}$, below both the
off-grid interpolation floor and the grid-resolution accuracy floor. The $K = 0$
row documents why the inversion is necessary: skipping it fails to converge at
the coarse grid.

Run: ``PYTHONPATH=code uv run python -m ssj.bench_euler_passes``.
"""

from __future__ import annotations

import logging

from cli import start_logging

from ssj.bench_common import (
    INTERP_GRIDS,
    SOLVE_TOL,
    log_done,
    solve_egm_k,
    write_csv,
)
from ssj.bench_interp import residual_log10
from ssj.benchmark import (
    MAXIT,
    backward_args,
    calibration,
    robust_euler_stats,
    solve_negm,
)

logger = logging.getLogger(__name__)

_PASSES = (0, 1, 2)
_HEADER = [
    "resolution",
    "nB",
    "nA",
    "config",
    "converged",
    "iters",
    "ongrid_median",
    "ongrid_p95",
    "ongrid_max",
    "offgrid_log10_median",
    "offgrid_log10_p95",
    "offgrid_log10_max",
]


def solve_passes(cal, k, tol=SOLVE_TOL, maxit=MAXIT, require=True):
    """EGM$^n$ (ENGINE regrid) with the regular-grid Euler inversion iterated ``k``
    times. ``k = 0`` skips the inversion (the shortcut EGM avoids); ``require``
    as in ``ssj.benchmark.solve_backward``.
    """
    return solve_egm_k(
        backward_args(cal),
        k,
        cal,
        tol=tol,
        maxit=maxit,
        require=require,
    )


def csv_row(label, nB, nA, config, out, cal):
    """One CSV row: convergence, iterations, and both residual families."""
    converged = out["converged"]
    og_med, og_p95, og_mx = robust_euler_stats(out, cal)  # on-grid (nodes)
    ff_med, ff_p95, ff_mx = residual_log10(cal, out)  # off-grid (log10)
    logger.info(
        "%-7s %-5s converged=%-5s it=%4d | on-grid med=%.2e | off-grid log10 "
        "med=%.2f p95=%.2f max=%.2f",
        label,
        config,
        converged,
        out["iter"],
        og_med,
        ff_med,
        ff_p95,
        ff_mx,
    )
    return [
        label.lower(),
        nB,
        nA,
        config,
        "yes" if converged else "no",
        out["iter"],
        f"{og_med:.3e}",
        f"{og_p95:.3e}",
        f"{og_mx:.3e}",
        f"{ff_med:.3f}",
        f"{ff_p95:.3f}",
        f"{ff_mx:.3f}",
    ]


def measure():
    """Convergence and residual rows for every (grid, configuration)."""
    rows = []
    for label, nB, nA in INTERP_GRIDS:
        cal = calibration(nB, nA)
        # K = 0 fails to converge at the coarse grid, and its row records the failure.
        for k in _PASSES:
            out = solve_passes(cal, k, require=k > 0)
            rows.append(csv_row(label, nB, nA, f"K{k}", out, cal))
        rows.append(csv_row(label, nB, nA, "negm", solve_negm(cal), cal))
    return rows


def main() -> None:
    start_logging()
    rows = measure()
    write_csv("euler_passes_abrs.csv", _HEADER, rows)
    log_done("euler_passes_abrs.csv")


if __name__ == "__main__":
    main()
