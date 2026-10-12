"""Shared plumbing for the wall-clock/table benchmark scripts.

``bench_ssj_reference``, ``bench_timing``, and ``bench_interp`` each regenerate
one paper table from best-of-``reps`` warmed solves at matched grids: the
figure output directory, the CSV writer, the best-of timing loop, grid-label
formatting for a table cell, and the thread-count guard every ``main`` enforces
before timing.
"""

from __future__ import annotations

import functools
import logging

import numba
from repo_paths import FIG_DIR, write_rows
from timing import timed

from ssj.benchmark import MAXIT, SOLVE_TOL, solve_backward
from ssj.hh_egm import backward_step

# The published solve-time grids ``(nB, nA)`` of the Section 3 timing tables.
SOLVE_TIME_GRIDS = [(40, 60), (80, 120), (160, 240)]

# The policy-benchmark grid ``(nB, nA)``: the refinement anchor.
BENCHMARK_GRID = (192, 160)

# The timing grids plus the paper's benchmark grid, labeled by family: shared by
# every table that reports a household solve at the published resolutions
# (``bench_accuracy_negm`` prepends its own diagnostic grids).
TIMING_GRIDS = [("timing", *g) for g in SOLVE_TIME_GRIDS] + [
    ("benchmark", *BENCHMARK_GRID),
]

# The labeled coarse/medium/fine grids of the interpolation-accuracy tables.
INTERP_GRIDS = [("Coarse", 24, 20), ("Medium", 48, 40), ("Fine", 96, 80)]


logger = logging.getLogger(__name__)


def log_done(*names) -> None:
    """Log the artifacts a benchmark ``main`` wrote to ``FIG_DIR``."""
    logger.info("DONE: wrote %s to %s", ", ".join(names), FIG_DIR)


def write_csv(name, header, rows) -> None:
    write_rows(FIG_DIR / name, header, rows)


def write_latex_table(path, table, head, body) -> None:
    r"""Write ``path`` as a MyST raw-LaTeX booktabs table (callers pass their own
    ``FIG_DIR / name``, so a test redirecting a module's ``FIG_DIR`` holds).

    ``table`` is ``(caption, label, size, colsep, colspec)``; the ``head`` rows sit
    between ``\\toprule`` and ``\\midrule``, the ``body`` rows before ``\\bottomrule``.
    The fence is the econ-ark-myst ``latex`` directive, which writes the body into
    the LaTeX export as is and keeps ``\\setlength`` out of the copy MyST parses for
    the site and the Typst PDF.
    """
    caption, label, size, colsep, colspec = table
    rows = [
        "```{latex}",
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        size,
        f"\\setlength{{\\tabcolsep}}{{{colsep}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *head,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "```",
    ]
    path.write_text("\n".join(rows) + "\n")


def best_time(fn, reps=3, warm=True):
    """Best wall time of ``reps`` timed calls to the zero-argument ``fn``, after
    one untimed warm (compile) call when ``warm``, and every call's output in
    order (the warm call's first).
    """
    outs = [fn()] if warm else []
    best = float("inf")
    for _ in range(reps):
        out, seconds = timed(fn)
        outs.append(out)
        best = min(best, seconds)
    return best, outs


def best_of(fn, cal, reps=3):
    """``(best, out)``: the ``best_time`` of ``fn(cal)`` after its own warm call,
    and the last timed repetition's output. ``fn`` is a ``ssj.benchmark`` solver,
    which raises unless its solve converges, so no unconverged solve is timed.
    """
    best, outs = best_time(functools.partial(fn, cal), reps)
    return best, outs[-1]


def grid_label(g) -> str:
    return f"$({g[0]}, {g[1]})$"


def solve_egm_k(step_args, k, cal, tol=SOLVE_TOL, maxit=MAXIT, require=True):
    """Iterate ``ssj.hh_egm.backward_step`` with the regular-grid Euler
    inversion applied ``k`` times, from ``step_args``
    (``ssj.benchmark.backward_args(cal)``, or an equivalent tuple built
    around a caller-supplied ``l_grid``); ``require`` as in ``solve_backward``.
    """
    return solve_backward(
        cal,
        backward_step,
        *step_args,
        k,
        tol=tol,
        maxit=maxit,
        require=require,
    )


def require_threads(threads, protocol) -> None:
    """Hard-fail if numba exposes fewer than ``threads`` cores: the published
    ``protocol`` compares matched thread counts, so a smaller machine would
    silently publish a different measurement.
    """
    if threads > numba.config.NUMBA_NUM_THREADS:
        msg = f"the published {protocol}; this machine exposes {numba.config.NUMBA_NUM_THREADS}."
        raise RuntimeError(msg)
