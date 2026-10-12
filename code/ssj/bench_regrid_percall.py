"""Per-call Stage-0 regrid timings: ENGINE against the curvilinear comparator.

Times one full realign of the regular $(b, a)$ target grid from the converged
Stage-0 mesh, per interpolator, at the interpolation diagnostic's three grids
(the solve-time table's grid family): the direct reading behind Section 4's
accounting of where the full-solve gap comes from. Both realign paths are
serial by construction (the ENGINE call here is the serial regrid kernel; the
sector walk is a single-thread loop), which is also why the per-call reading
reconciles with the threaded diagnostic's per-iteration difference. The curvilinear timing includes
its per-call interpolant construction because the solve loop rebuilds it every
backward step, exactly as ``ssj.interp_loop`` does. Best of five repetitions
after one warm-up call per path, so compilation is excluded.

Run: ``PYTHONPATH=code uv run python -m ssj.bench_regrid_percall``.
"""

from __future__ import annotations

import logging
from functools import partial

from cli import start_logging

from ssj.bench_common import INTERP_GRIDS, best_time, log_done, write_csv
from ssj.benchmark import calibration, solve_compact, stage0_mesh
from ssj.interp_loop import a_choice_mesh, regrid_with

logger = logging.getLogger(__name__)

_REPEATS = 5


_HEADER = [
    "label",
    "nB",
    "nA",
    "engine_percall_s",
    "curvilinear_percall_s",
    "ratio",
]


def measure(label, n_b, n_a):
    cal = calibration(n_b, n_a)
    mesh = stage0_mesh(solve_compact(cal), cal)
    ap_choice = a_choice_mesh(cal, mesh["l_grid"])
    args = (mesh["b_endo"], mesh["a_endo"], ap_choice, cal["b"], cal["a"])
    t_e = best_time(partial(regrid_with, "engine", *args), _REPEATS)[0]
    t_c = best_time(partial(regrid_with, "curvilinear", *args), _REPEATS)[0]
    row = [label, n_b, n_a, f"{t_e:.6e}", f"{t_c:.6e}", f"{t_c / t_e:.1f}"]
    logger.info(
        "%s (%d, %d): engine %.3e s, curvilinear %.3e s, ratio %.1f",
        label,
        n_b,
        n_a,
        t_e,
        t_c,
        t_c / t_e,
    )
    return row


def main() -> None:
    start_logging()
    rows = [measure(*g) for g in INTERP_GRIDS]
    write_csv("regrid_percall_abrs.csv", _HEADER, rows)
    log_done("regrid_percall_abrs.csv")


if __name__ == "__main__":
    main()
