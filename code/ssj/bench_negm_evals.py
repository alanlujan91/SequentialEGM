"""NEGM illiquid-FOC evaluations per node, a supplementary reading of Section 3's
NEGM inner search.

Runs the production NEGM solve at the three timing grids (``backward_negm`` with
``NEGM_INNER_REL_TOL``, outer tolerance ``SOLVE_TOL``) and, at every backward
iteration, feeds the same ``(Va_p, Vb_p)`` to an instrumented copy of
``negm_deliver``. The copy reuses the production Stage 1, ``foc``,
``feasible_upper``, ``negm_cell`` and ``store_cell`` and replaces only the
Illinois loop with one that counts. Every iteration asserts that the copy's six
outputs equal the production step's bit for bit.

Each call of ``foc`` counts as one evaluation: a residual ``F(a')`` from two
bilinear envelope lookups. A node at the lower corner ``a' = a_grid[0]`` takes
1 evaluation, a node at the upper edge 2, and an interior node 2 plus its
Illinois iterations. Because the ``feasible_upper`` budget bisection evaluates
only the closed-form budget, the CSV reports it apart, as the share of nodes
whose bracket top falls below the grid.

Run: ``PYTHONPATH=code uv run python -m ssj.bench_negm_evals``.
"""

from __future__ import annotations

import logging

import numpy as np
from cli import start_logging
from engine.core import SAFE_FASTMATH
from numba import njit, prange

from ssj.bench_common import SOLVE_TIME_GRIDS, log_done, write_csv
from ssj.benchmark import backward_args, calibration, iterate
from ssj.hh_egm import kernel_tuples, policy_arrays, split_outputs, store_cell
from ssj.hh_negm import (
    NEGM_INNER_REL_TOL,
    backward_negm,
    feasible_upper,
    foc,
    negm_cell,
    negm_stage1,
)

logger = logging.getLogger(__name__)

CSV_NAME = "negm_foc_evals_abrs.csv"
HEADER = [
    "nB",
    "nA",
    "iterations",
    "evals_mean_final",
    "evals_mean_all_steps",
    "evals_median_final",
    "evals_max_final",
    "interior_share_final",
    "bisection_share_final",
]
OUTPUT_NAMES = ("Va", "Vb", "a", "b", "c", "uce")


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def illinois_count(
    x0,
    f0,
    x1,
    f1,
    ag,
    m,
    v1z_e,
    v1ap_e,
    l_grid,
    a_grid,
    Ra,
    chi0,
    chi1,
    chi2,
    tol,
):
    """``hh_negm.illinois`` with the same control flow, also returning the
    number of ``foc`` calls.
    """
    n = 0
    for _ in range(100):
        if abs(x1 - x0) < tol:
            break
        denom = f1 - f0
        if denom == 0.0:
            break
        x2 = x1 - f1 * (x1 - x0) / denom
        lo = min(x1, x0)
        hi = x0 + x1 - lo
        if x2 <= lo or x2 >= hi:
            x2 = 0.5 * (lo + hi)
        f2, _ = foc(x2, ag, m, v1z_e, v1ap_e, l_grid, a_grid, Ra, chi0, chi1, chi2)
        n += 1
        if f2 == 0.0:
            return x2, n
        if (f2 > 0.0) == (f1 > 0.0):
            x1, f1 = x2, f2
            f0 *= 0.5
        else:
            x0, f0 = x1, f1
            x1, f1 = x2, f2
    return 0.5 * (x0 + x1), n


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def choice_count(
    ag,
    mres,
    v1z_e,
    v1ap_e,
    l_grid,
    a_grid,
    Ra,
    chi0,
    chi1,
    chi2,
    inner_rel_tol,
):
    """``hh_negm.negm_choice``, also returning ``(n_foc, interior, bisected)``."""
    a_lo = a_grid[0]
    f_lo, _ = foc(a_lo, ag, mres, v1z_e, v1ap_e, l_grid, a_grid, Ra, chi0, chi1, chi2)
    if f_lo <= 0.0:
        return a_lo, 1, 0, 0
    a_hi = feasible_upper(a_lo, ag, mres, a_grid[-1], Ra, chi0, chi1, chi2)
    bisected = 1 if a_hi < a_grid[-1] else 0
    f_hi, _ = foc(a_hi, ag, mres, v1z_e, v1ap_e, l_grid, a_grid, Ra, chi0, chi1, chi2)
    if f_hi >= 0.0:
        return a_hi, 2, 0, bisected
    a_e, n = illinois_count(
        a_lo,
        f_lo,
        a_hi,
        f_hi,
        ag,
        mres,
        v1z_e,
        v1ap_e,
        l_grid,
        a_grid,
        Ra,
        chi0,
        chi1,
        chi2,
        inner_rel_tol * (1.0 + a_hi),
    )
    return a_e, 2 + n, 1, bisected


@njit(parallel=True, cache=True, fastmath=SAFE_FASTMATH)
def deliver_count(c_s1, v1z, v1ap, grids, e_grid, params, inner_rel_tol):
    """``hh_negm.negm_deliver`` line for line, plus the per-node counts
    ``(n_foc, interior, bisected)`` stacked on ``(3, e, b, a)``.
    """
    a_grid, b_grid, l_grid, z_grid = grids
    _beta, _eis, Ra, Rb, chi0, chi1, chi2 = params
    nE, nB, nA = c_s1.shape[0], b_grid.shape[0], a_grid.shape[0]
    outs = policy_arrays(nE, nB, nA)
    counts = np.zeros((3, nE, nB, nA), dtype=np.int64)
    for p in prange(nE * nB):
        e, ib = p // nB, p % nB
        for ia in range(nA):
            ag = a_grid[ia]
            mres = z_grid[e] + Rb * b_grid[ib] + Ra * ag
            a_e, n, interior, bisected = choice_count(
                ag,
                mres,
                v1z[e],
                v1ap[e],
                l_grid,
                a_grid,
                Ra,
                chi0,
                chi1,
                chi2,
                inner_rel_tol,
            )
            counts[0, e, ib, ia] = n
            counts[1, e, ib, ia] = interior
            counts[2, e, ib, ia] = bisected
            c_e, b_e, cb, psi2 = negm_cell(a_e, ag, mres, c_s1[e], grids, params)
            store_cell(outs, e, ib, ia, a_e, b_e, cb, c_e, psi2, e_grid[e], params)
    return split_outputs(outs), counts


def counted_solve(cal):
    """The production NEGM solve at ``cal`` with the per-node counts of every
    backward step. Raises if a step's instrumented outputs differ from the
    production step's. Returns ``(out, counts per step)``.
    """
    args = backward_args(cal)
    grids, params = kernel_tuples(*args[:4], *args[5:])
    steps = []

    def step(Va_p, Vb_p):
        ref = backward_negm(Va_p, Vb_p, *args, NEGM_INNER_REL_TOL)
        stage1 = negm_stage1(Va_p, Vb_p, cal["b"], args[2], cal["beta"], cal["eis"])
        outs, counts = deliver_count(
            *stage1,
            grids,
            cal["e"],
            params,
            NEGM_INNER_REL_TOL,
        )
        for name, x, y in zip(OUTPUT_NAMES, ref, outs, strict=True):
            if not np.array_equal(x, y, equal_nan=True):
                msg = f"step {len(steps) + 1}: instrumented NEGM {name} differs"
                raise AssertionError(msg)
        steps.append(counts)
        return ref[:5]

    return iterate(step, cal), steps


def measure(grid):
    """One CSV row: the converged solve's iteration count and its FOC-evaluation
    statistics at the final step and over all steps. Raises unless it converges.
    """
    out, steps = counted_solve(calibration(*grid))
    if not out["converged"]:
        msg = f"NEGM solve at {grid} did not converge"
        raise RuntimeError(msg)
    n_final, interior, bisected = steps[-1]
    row = [
        *grid,
        out["iter"],
        f"{n_final.mean():.3f}",
        f"{np.mean([s[0].mean() for s in steps]):.3f}",
        f"{np.median(n_final):.0f}",
        int(n_final.max()),
        f"{interior.mean():.5f}",
        f"{bisected.mean():.5f}",
    ]
    logger.info("%s: %s", grid, dict(zip(HEADER[2:], row[2:], strict=True)))
    return row


def main() -> None:
    start_logging()
    write_csv(CSV_NAME, HEADER, [measure(g) for g in SOLVE_TIME_GRIDS])
    log_done(CSV_NAME)


if __name__ == "__main__":
    main()
