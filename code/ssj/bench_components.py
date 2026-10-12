"""Component-by-component runtime table: regenerates ``tab_components_abrs.md``.

Attributes the reported gains to the components of the backward step. This
times each component of the EGM$^n$ backward
step separately, accumulated over a full converged solve (tolerance ``SOLVE_TOL``),
using the full-solve diagnostic's building blocks, whose ENGINE configuration
reproduces the production kernel exactly (the suite's bit-identity gate):

- expectation step: the Markov update of the continuation marginals;
- stage inversions: Stage 1 kinked consumption EGM plus the closed-form
  Stage 0 inversion, producing the endogenous mesh (Phase 1);
- mesh-to-grid interpolation: the ENGINE regrid of the illiquid choice;
- regular-grid inversion and envelopes: the corner selection, the seeded
  Euler inversion, and the envelope updates (Phase 3).

Times are serial (one numba thread), the cleanest attribution: parallel
timings divide each component by its own scaling factor. Writes the raw data
to ``components_abrs.csv``.

Run: ``PYTHONPATH=code uv run python -m ssj.bench_components``.
"""

from __future__ import annotations

import logging
import time

import numba
from cli import start_logging

from ssj.bench_common import (
    FIG_DIR,
    SOLVE_TIME_GRIDS,
    SOLVE_TOL,
    grid_label,
    log_done,
    write_csv,
    write_latex_table,
)
from ssj.benchmark import (
    MAXIT,
    calibration,
    expect,
    marginal_change,
    regular_grid_inversion,
    require_converged,
    stage_inversions,
)
from ssj.interp_loop import initial_state, regrid_with

logger = logging.getLogger(__name__)

_PARTS = (
    "expectation",
    "stage inversions",
    "mesh-to-grid interpolation",
    "regular-grid inversion",
)


def instrumented_solve(cal, maxit=MAXIT):
    """Converged solve with per-component accumulated wall time."""
    Va, Vb, l_grid, ap_choice = initial_state(cal)
    acc = dict.fromkeys(_PARTS, 0.0)
    it = 0
    converged = False
    for it in range(1, maxit + 1):
        t0 = time.perf_counter()
        Va_p, Vb_p = expect(cal["Pi"], Va), expect(cal["Pi"], Vb)
        t1 = time.perf_counter()
        c_s1, bp_s1, a_endo, b_endo = stage_inversions(Va_p, Vb_p, cal, l_grid)
        t2 = time.perf_counter()
        a_pol = regrid_with(
            "engine",
            b_endo,
            a_endo,
            ap_choice,
            cal["b"],
            cal["a"],
        )
        t3 = time.perf_counter()
        Va_n, Vb_n, _a, _b, _c, _uce = regular_grid_inversion(
            a_pol,
            c_s1,
            bp_s1,
            Vb_p,
            cal,
            l_grid,
        )
        t4 = time.perf_counter()
        acc["expectation"] += t1 - t0
        acc["stage inversions"] += t2 - t1
        acc["mesh-to-grid interpolation"] += t3 - t2
        acc["regular-grid inversion"] += t4 - t3
        d = marginal_change(Va_n, Va, Vb_n, Vb)
        Va, Vb = Va_n, Vb_n
        if d < SOLVE_TOL:
            converged = True
            break
    require_converged(
        converged,
        f"instrumented solve failed to converge after {it} iterations",
    )
    return acc, it


def measure():
    numba.set_num_threads(1)
    out = {}
    for g in SOLVE_TIME_GRIDS:
        cal = calibration(*g)
        instrumented_solve(cal)  # warm (njit compile + cache)
        acc, it = instrumented_solve(cal)
        out[g] = (acc, it)
        total = sum(acc.values())
        logger.info(
            "(%3d,%3d) serial, %d iters, %.2fs: %s",
            *g,
            it,
            total,
            "  ".join(f"{k} {v:.2f}s ({100 * v / total:.0f}%)" for k, v in acc.items()),
        )
    return out


def write_table(m) -> None:
    cap = (
        "Runtime in seconds by component of the EGM$^n$ backward solve, summed over "
        "a converged one-thread solve (shares in parentheses). Stage inversions are "
        "Stage 1's consumption EGM and the Stage 0 inversion; mesh-to-grid "
        "interpolation is the ENGINE regrid; regular-grid inversion is corner "
        "selection, Euler inversion, and envelope updates."
    )
    head = [
        r"Grid & Expectation & Stage inversions & Mesh-to-grid & Regular-grid inversion \\",
    ]
    rows = []
    for g in SOLVE_TIME_GRIDS:
        acc, _it = m[g]
        total = sum(acc.values())
        cells = [f"{acc[k]:.2f} ({100 * acc[k] / total:.0f}\\%)" for k in _PARTS]
        rows.append(f"{grid_label(g)} & " + " & ".join(cells) + r" \\")
    table = (cap, "tab-components-abrs", r"\small", "5pt", "@{}lrrrr@{}")
    write_latex_table(FIG_DIR / "tab_components_abrs.md", table, head, rows)


def write_csvs(m) -> None:
    data = []
    for g in SOLVE_TIME_GRIDS:
        acc, it = m[g]
        for k in _PARTS:
            data.append([*g, k.replace(" ", "_"), f"{acc[k]:.4f}", it])
    header = ["nB", "nA", "component", "seconds", "iters"]
    write_csv("components_abrs.csv", header, data)


def main() -> None:
    start_logging()
    m = measure()
    write_table(m)
    write_csvs(m)
    log_done("tab_components_abrs.md", "components_abrs.csv")


if __name__ == "__main__":
    main()
