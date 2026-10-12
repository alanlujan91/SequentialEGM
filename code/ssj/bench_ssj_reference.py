"""Reference-solver benchmark: regenerates ``tab_ssj_reference.md``.

The online-appendix counterpart of the paper's timing table: EGM$^n$ against
the two-asset block's native solver (the purpose-built hybrid distributed with
the sequence-jacobian package, pinned in the replication archive), at matched
grids, tolerance (``SOLVE_TOL``, both solvers' default), and linear interpolation, best
of three from a warm start. The reference implementation is serial, so it has
one time column; EGM$^n$ is timed at 1 and 16 numba threads. Beside the times
sit backward iterations to the shared tolerance and the off-grid robust
Euler median of each converged policy, the accuracy reading behind the
online appendix's "matches its off-grid robust Euler accuracy".

Alongside the table this writes the raw data to ``ssj_reference_abrs.csv``.

Run: ``PYTHONPATH=code uv run python -m ssj.bench_ssj_reference``.
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
from ssj.bench_interp import residual_log10
from ssj.benchmark import calibration, solve_compact, solve_shipped

logger = logging.getLogger(__name__)


def measure():
    """Timings, iteration counts and accuracy, the latter two read once off the
    serial timed solves (both are deterministic and thread-invariant).
    """
    out = {}
    refs = {}
    for g in SOLVE_TIME_GRIDS:
        out["t_ref", g], refs[g] = best_of(solve_shipped, calibration(*g))
        logger.info("reference %s (serial): %.3fs", g, out["t_ref", g])
    for nt in (1, 16):
        numba.set_num_threads(nt)
        for g in SOLVE_TIME_GRIDS:
            cal = calibration(*g)
            out["t_egm", g, nt], egm = best_of(solve_compact, cal)
            logger.info("egm %s @%dt: %.3fs", g, nt, out["t_egm", g, nt])
            if nt == 1:
                out["iters", g] = (egm["iter"], refs[g]["iter"])
                out["acc", g] = (
                    residual_log10(cal, egm)[0],
                    residual_log10(cal, refs[g])[0],
                )
    return out


def write_table(m) -> None:
    cap = (
        "Time in seconds and converged-policy accuracy: the two-asset block's native solver "
        "versus EGM$^n$, at matched grids, tolerance, and linear interpolation. The "
        "native solver is serial; parentheses give its time over EGM$^n$'s at sixteen "
        "threads. Iterations are backward steps to the shared tolerance; accuracy is "
        "the off-grid robust Euler median ($\\log_{10}$), EGM$^n$ first."
    )
    head = [
        r"Grid & Native, serial & \multicolumn{2}{c}{EGM$^n$} & Iterations & Accuracy \\",
        r"\cmidrule(lr){3-4}",
        r"& & 1 thread & 16 threads & EGM$^n$/native & EGM$^n$/native \\",
    ]
    rows = []
    for g in SOLVE_TIME_GRIDS:
        tr, e1, e16 = m["t_ref", g], m["t_egm", g, 1], m["t_egm", g, 16]
        ie, ir = m["iters", g]
        ae, ar = m["acc", g]
        # Display accuracy from the CSV-persisted precision so the table is
        # reproducible from ssj_reference_abrs.csv alone.
        ae_d, ar_d = float(f"{ae:.4f}"), float(f"{ar:.4f}")
        rows.append(
            f"{grid_label(g)} & {tr:.2f} & {e1:.2f} & {e16:.2f} ({tr / e16:.1f}x) & "
            f"{ie}/{ir} & ${ae_d:.2f}$/${ar_d:.2f}$ " + r"\\",
        )
    table = (cap, "tab-ssj-reference", r"\footnotesize", "4pt", "@{}lrrrcc@{}")
    write_latex_table(FIG_DIR / "tab_ssj_reference.md", table, head, rows)


def write_csvs(m) -> None:
    data = []
    for g in SOLVE_TIME_GRIDS:
        ie, ir = m["iters", g]
        ae, ar = m["acc", g]
        data.append(
            [
                *g,
                "reference",
                1,
                f"{m['t_ref', g]:.4f}",
                ir,
                f"{ar:.4f}",
            ],
        )
        for nt in (1, 16):
            data.append(
                [
                    *g,
                    "egmn",
                    nt,
                    f"{m['t_egm', g, nt]:.4f}",
                    ie,
                    f"{ae:.4f}",
                ],
            )
    header = ["nB", "nA", "solver", "threads", "seconds", "iters", "offgrid_log10_med"]
    write_csv("ssj_reference_abrs.csv", header, data)


def main() -> None:
    start_logging()
    require_threads(16, "protocol times EGM^n at 16 threads")
    m = measure()
    write_table(m)
    write_csvs(m)
    log_done("tab_ssj_reference.md", "ssj_reference_abrs.csv")


if __name__ == "__main__":
    main()
