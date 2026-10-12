"""NEGM inner-tolerance robustness line behind Section 3's timing claim.

The matched NEGM comparator root-finds each node's illiquid first-order
condition with an Illinois bracket at relative tolerance ``1e-11``, two
orders tighter than the outer ``1e-9`` convergence norm. This module
re-times serial NEGM with the inner tolerance matched to the outer and
reports whether the Section 3 serial factor survives: the two grids with
measured serial NEGM cells in the timing table, one thread, best-of-three
after a warm solve (the loosened leg runs the kernel the tight leg already
compiled, so it skips its own warm solve). It also records the
iteration counts and the consumption-policy distance between the two
inner-tolerance variants, so the reading shows what the tighter bracket
buys as well as what it costs. An iteration count at the 8,000 cap means
the solve never met the outer norm: with the inner bracket loosened to the
outer tolerance, per-node root-find noise keeps the fixed-point iteration
from converging, which is the measured reason the comparator's bracket
sits two orders below the outer norm.

Run: ``PYTHONPATH=code uv run python -m ssj.bench_negm_innertol``.
"""

from __future__ import annotations

import logging
from functools import partial

import numba
from cli import start_logging

from ssj.bench_common import (
    SOLVE_TIME_GRIDS,
    SOLVE_TOL,
    best_of,
    best_time,
    log_done,
    write_csv,
)
from ssj.benchmark import (
    backward_args,
    calibration,
    policy_distance,
    solve_backward,
    solve_compact,
)
from ssj.hh_negm import NEGM_INNER_REL_TOL, backward_negm

logger = logging.getLogger(__name__)

_REPEATS = 3


_HEADER = [
    "nB",
    "nA",
    "egmn_serial_s",
    "negm_tol11_serial_s",
    "negm_tol9_serial_s",
    "factor_tol11",
    "factor_tol9",
    "iters_negm_tol11",
    "iters_negm_tol9",
    "dist_c_tol9_vs_tol11",
]


def solve_negm_inner(cal, inner_rel_tol, require=True):
    """NEGM solve with an explicit inner bracket tolerance (mirrors solve_negm);
    ``require`` as in ``solve_backward``.
    """
    return solve_backward(
        cal,
        backward_negm,
        *backward_args(cal),
        inner_rel_tol,
        require=require,
    )


def measure(grid):
    cal = calibration(*grid)
    t_e, _ = best_of(solve_compact, cal, _REPEATS)
    t11, out11 = best_of(
        partial(solve_negm_inner, inner_rel_tol=NEGM_INNER_REL_TOL),
        cal,
        _REPEATS,
    )
    # A bracket loosened to the outer tolerance is expected to hit the iteration
    # cap (Section 3), so this leg records its unconverged solve. Its kernel is
    # the one the tight leg just compiled, so it skips the warm solve.
    t9, outs9 = best_time(
        partial(solve_negm_inner, cal, SOLVE_TOL, require=False),
        _REPEATS,
        warm=False,
    )
    out9 = outs9[-1]
    dist = policy_distance(out9, cal, out11, cal, "c")
    row = [
        *grid,
        f"{t_e:.4f}",
        f"{t11:.4f}",
        f"{t9:.4f}",
        f"{t11 / t_e:.1f}",
        f"{t9 / t_e:.1f}",
        out11["iter"],
        out9["iter"],
        f"{dist:.2e}",
    ]
    logger.info("%s: %s", grid, dict(zip(_HEADER[2:], row[2:], strict=True)))
    return row


def main() -> None:
    start_logging()
    numba.set_num_threads(1)
    rows = [measure(g) for g in SOLVE_TIME_GRIDS[:2]]
    write_csv("negm_innertol_abrs.csv", _HEADER, rows)
    log_done("negm_innertol_abrs.csv")


if __name__ == "__main__":
    main()
