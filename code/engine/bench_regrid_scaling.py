"""Scaling benchmark: ENGINE dense regrid wall-clock versus mesh size.

Backs the scaling sentence of Section 4: a dense regrid onto a P x Q target
grid costs at most O(P Q log J log K) when each target is searched afresh, but
sorted target axes let each bracket search resume where the previous one ended,
so the work is the number of targets plus their total bracket displacement, and
the measured cost falls far below that bound. The benchmark times the
``regrid`` kernels on the synthetic warped EGM mesh used by the test suite at
matched refinement (P = J, Q = K), doubling J = K from 32 to 1024, and fits
the log-log slope of best-of-``_REPS`` warm times against N = J K. The paper
reports the ``serial`` slope (N^0.98) and its time per output value at the
largest mesh (about ten nanoseconds). An N log N law would have effective
slope about 1.10 over this range, and its time per output would grow like
log N instead of holding constant.

The timed quantity is the public ``regrid`` call, which includes output
allocation and writes. Those costs grow linearly too, but page faults on
multi-megabyte outputs can push the large-N slope a few hundredths up, so the
fitted slope describes the whole call, kernel and memory traffic together.
Wall-time scaling also depends on machine load; run on a quiescent machine.

The ``serial`` and ``ipo`` kernel variants get a fitted slope. The
``parallel`` variant is timed as well, but its fixed thread-pool dispatch cost
dominates the small-N end, so no slope is fitted for it. Each mesh's largest
bracket displacement between adjacent rows (``alpha_hat``) is recorded beside
the timings. On this mesh ``alpha_hat`` grows with size, since the additive
warp shifts each slice by more than the local spacing, while the time per
output is constant: the kernel walks each bracket path once, and on these
monotone row shifts that path's total variation is at most J whatever
``alpha_hat`` is. The mesh is a harder case than a small-alpha production mesh.

Writes ``content/figures/regrid_scaling.csv`` (raw timings) and logs the
fitted slopes. Run from the repo root::

    PYTHONPATH=code uv run python code/engine/bench_regrid_scaling.py
"""

from __future__ import annotations

import logging

import numba
import numpy as np
from cli import start_logging
from repo_paths import FIG_DIR, write_rows
from timing import mean_call_time

from engine.engine2d import Engine2DInterp, warmup_jit
from engine.utilities import empirical_ipo_alpha, ipm_violations, make_warped_egm_grid

logger = logging.getLogger(__name__)

_OUT = FIG_DIR / "regrid_scaling.csv"
_SIZES = [32, 64, 128, 256, 512, 1024]
_REPS = 7
# Inner-call batching target: repeat the regrid until one timed sample takes
# at least this long, so per-call Python overhead does not flatten the
# small-N end of the fit.
_MIN_SAMPLE_S = 2e-3
_EST_PROBES = 3
_FIT_MODES = ("serial", "ipo")
_ALL_MODES = ("serial", "ipo", "parallel")
_THREADS = 8  # modest cap for the parallel variant; serial/ipo ignore it


def interior_axis(lo: float, hi: float, n: int) -> np.ndarray:
    """Sorted axis of ``n`` points on [lo, hi] shrunk by 5 percent of its
    length at each end.

    Applied to the sides of ``inscribed_box``, the two axes keep every
    target pair inside the mesh image.
    """
    pad = 0.05 * (hi - lo)
    return np.linspace(lo + pad, hi - pad, n)


def inscribed_box(x, y):
    """``(x_lo, x_hi, y_lo, y_hi)``: a coordinate box inside an IPM mesh's
    image. ``x`` runs from the largest ``x`` on the ``j = 0`` line to the
    smallest on the ``j = J - 1`` line; ``y`` runs from the largest ``y`` on
    the ``k = 0`` line to the smallest on the ``k = K - 1`` line.

    Each point of the image's four boundary polylines breaks one of the
    box's strict inequalities: no boundary point lies in the open box. A
    connected set off the boundary lies wholly inside the image or wholly
    outside it, and the box lies inside once one of its points does. The
    bounding box instead reaches past a warped image: on
    ``make_warped_egm_grid`` meshes 9 to 12 percent of its target pairs fall
    outside and take ENGINE's extrapolation branch.
    """
    return x[0].max(), x[-1].min(), y[:, 0].max(), y[:, -1].min()


def warped_regrid_case(s: int):
    """The ``s x s`` warped EGM mesh with its interpolant and targets inside
    its image, ``(x_endog, interp, x_new, y_new)``.
    """
    x_endog, y_endog, values = make_warped_egm_grid(s, s)
    # The regrid kernels require IPM (def-ipm); raise so a future
    # mesh-generator change cannot silently invalidate the timing.
    if any(ipm_violations(x_endog, y_endog)):
        msg = f"warped mesh at J=K={s} violates IPM; benchmark invalid"
        raise RuntimeError(msg)
    x_lo, x_hi, y_lo, y_hi = inscribed_box(x_endog, y_endog)
    return (
        x_endog,
        Engine2DInterp([values], x_endog, y_endog),
        interior_axis(x_lo, x_hi, s),
        interior_axis(y_lo, y_hi, s),
    )


def time_regrid(
    interp: Engine2DInterp,
    x_new,
    y_new,
    mode: str,
    reps: int = _REPS,
    min_sample_s: float = _MIN_SAMPLE_S,
) -> float:
    """Best-of-``reps`` warm per-call time for one regrid variant, batching inner
    calls until one timed sample takes at least ``min_sample_s``.

    The batch size doubles until the best of ``_EST_PROBES`` samples reaches
    the target; the best-of guards against a one-off scheduling stall ending
    the doubling early.
    """

    def regrid():
        return interp.regrid(x_new, y_new, mode=mode)

    def best_sample_s(n: int) -> float:
        return n * min(mean_call_time(regrid, n) for _ in range(_EST_PROBES))

    n_inner = 1
    while best_sample_s(n_inner) < min_sample_s:
        n_inner *= 2
    return min(mean_call_time(regrid, n_inner) for _ in range(reps))


def main() -> None:
    start_logging()
    warmup_jit()
    numba.set_num_threads(_THREADS)
    rows = []
    for s in _SIZES:
        x_endog, interp, x_new, y_new = warped_regrid_case(s)
        alpha_hat = empirical_ipo_alpha(x_endog)
        for mode in _ALL_MODES:
            t = time_regrid(interp, x_new, y_new, mode)
            rows.append(
                {
                    "mode": mode,
                    "J": s,
                    "K": s,
                    "N": s * s,
                    "alpha_hat": alpha_hat,
                    "time_s": t,
                    "ns_per_output": 1e9 * t / (s * s),
                },
            )
            logger.info(
                "J=K=%4d  N=%7d  alpha_hat=%d  %-8s  %.3e s  (%.1f ns/output)",
                s,
                s * s,
                alpha_hat,
                mode,
                t,
                1e9 * t / (s * s),
            )

    for mode in _FIT_MODES:
        pts = [(r["N"], r["time_s"]) for r in rows if r["mode"] == mode]
        log_n = np.log([n for n, _ in pts])
        log_t = np.log([t for _, t in pts])
        slope = np.polyfit(log_n, log_t, 1)[0]
        per_out_ratio = (pts[-1][1] / pts[-1][0]) / (pts[0][1] / pts[0][0])
        logger.info(
            "%-8s log-log slope of time vs N: %.3f  per-output growth ratio: %.2f",
            mode,
            slope,
            per_out_ratio,
        )

    header = ["mode", "J", "K", "N", "alpha_hat", "time_s", "ns_per_output"]
    write_rows(_OUT, header, [[r[k] for k in header] for r in rows])
    logger.info("wrote %s", _OUT)


if __name__ == "__main__":
    main()
