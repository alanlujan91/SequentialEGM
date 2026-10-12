"""Timing benchmark: regenerates ``tab_timing_negm.md``.

Times EGM$^n$ (the parallel njit kernel) against NEGM (the matched root-find
comparator) to a converged household solution at matched grids,
tolerance (``SOLVE_TOL``, every solver's default), and linear interpolation,
best of three from a warm start. Both are timed at 1 and 16 numba threads
(``numba.set_num_threads``); every cell is measured, including the
minutes-long serial NEGM solve at $(160,240)$.

Alongside the table this writes the raw per-cell data to ``timing_negm.csv``. The
Euler-residual comparison across solvers lives in ``ssj.bench_euler_passes``
(``euler_passes_abrs.csv``).

Run: ``PYTHONPATH=code uv run python -m ssj.bench_timing``.
"""

from __future__ import annotations

import logging

import numba
from cli import start_logging

from ssj.bench_common import (
    FIG_DIR,
    SOLVE_TIME_GRIDS,
    best_of,
    grid_label,
    log_done,
    require_threads,
    write_csv,
    write_latex_table,
)
from ssj.benchmark import calibration, solve_compact, solve_negm

logger = logging.getLogger(__name__)


def measure():
    """Best-of timings per (solver, grid, threads) and iteration counts per
    (solver, grid), read off the timed solves (iteration counts are
    thread-invariant).
    """
    times = {}
    iters = {}
    for nt in (1, 16):
        numba.set_num_threads(nt)
        for g in SOLVE_TIME_GRIDS:
            cal = calibration(*g)
            times["egm", g, nt], egm_out = best_of(solve_compact, cal)
            iters["egm", g] = egm_out["iter"]
            logger.info("egm %s @%dt: %.3fs", g, nt, times["egm", g, nt])
            times["negm", g, nt], negm_out = best_of(solve_negm, cal)
            iters["negm", g] = negm_out["iter"]
            logger.info("negm %s @%dt: %.3fs", g, nt, times["negm", g, nt])
    return times, iters


def write_negm(t) -> None:
    cap = (
        "Time in seconds to a converged two-asset solution, NEGM versus EGM$^n$, at "
        "matched grids, tolerance, interpolation, and thread counts. Parentheses give "
        "the speed ratio over NEGM at the same thread count."
    )
    head = [
        r"Grid & \multicolumn{2}{c}{1 thread} & \multicolumn{2}{c}{16 threads} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"& NEGM & EGM$^n$ & NEGM & EGM$^n$ \\",
    ]
    rows = []
    for g in SOLVE_TIME_GRIDS:
        e1, e16 = t["egm", g, 1], t["egm", g, 16]
        n1, n16 = t["negm", g, 1], t["negm", g, 16]
        rows.append(
            f"{grid_label(g)} & {n1:.2f} & {e1:.2f} ({n1 / e1:.1f}x) & {n16:.2f} & "
            f"{e16:.2f} ({n16 / e16:.1f}x) " + r"\\",
        )
    table = (cap, "tab-timing-negm", r"\small", "5pt", "@{}lrrrr@{}")
    write_latex_table(FIG_DIR / "tab_timing_negm.md", table, head, rows)


def write_csvs(t, iters) -> None:
    """Raw per-cell data behind the table, one row per (grid, solver, threads)."""
    rows = [
        [*g, solver, nt, f"{t[key, g, nt]:.4f}", iters[key, g]]
        for g in SOLVE_TIME_GRIDS
        for key, solver in (("negm", "negm"), ("egm", "egmn"))
        for nt in (1, 16)
    ]
    header = ["nB", "nA", "solver", "threads", "seconds", "iters"]
    write_csv("timing_negm.csv", header, rows)


def main() -> None:
    start_logging()
    require_threads(16, "timing protocol compares 1 and 16 matched threads")
    t, iters = measure()
    write_negm(t)
    write_csvs(t, iters)
    log_done("tab_timing_negm.md", "timing_negm.csv")


if __name__ == "__main__":
    main()
