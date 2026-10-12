r"""In-loop interpolator diagnostic: regenerates ``tab_solvetime_abrs.md``.

Solves the compact EGM$^n$ two-asset block (``ssj.interp_loop.solve_interp``) with
each Stage-0 realign interpolator (ENGINE, curvilinear, Delaunay) at three grids,
holding every other component fixed, and reports the best-of-three warmed full
solve time and the robust consumption-unit Euler error of the converged policy on
a dense off-grid interior test set (median, 95th percentile, max of $\\log_{10}$).
An interpolator that does not converge (the Delaunay triangulation runs to the
iteration cap on this model) is recorded as "fails" in the CSV and the run log
but excluded from the paper table, which reports converged solves only. ``main`` also computes the auxiliary
readings the appendix prose cites,
both logging them and persisting each to a CSV alongside the table: the SSJ
reference anchor under the identical metric (``ssj_anchor_abrs.csv``), the
refinement check against a high-resolution benchmark and the grid-bound
insensitivity check (``refinement_bounds_abrs.csv``), the reversed-pass-order
readings (``order_reading_abrs.csv``), the high-wealth extrapolation-band
residuals (``extrap_band_abrs.csv``), the fraction of Stage-0 targets outside
the converged endogenous-mesh image (``extrap_frequency_abrs.csv``), the
policy-refinement table (``tab_refinement_bounds_abrs.md``), and the raw
per-cell table data (``solvetime_abrs.csv``).

Run: ``PYTHONPATH=code uv run python -m ssj.bench_interp``.
``--band-only`` regenerates only ``extrap_band_abrs.csv``. ``--validation-only``
regenerates the deterministic refinement, bounds, and extrapolation-frequency
artifacts without disturbing the published wall-clock table.
"""

from __future__ import annotations

import argparse
import functools
import logging

import numba
import numpy as np
from cli import start_logging
from numerics import median_p95_max, rel_diff

from ssj.bench_common import (
    BENCHMARK_GRID,
    FIG_DIR,
    INTERP_GRIDS,
    best_time,
    grid_label,
    log_done,
    require_threads,
    write_csv,
    write_latex_table,
)
from ssj.benchmark import (
    assert_mesh_fold_free,
    assert_mesh_ipm,
    calibration,
    grid_interp,
    liquid_euler_gap,
    node_mesh,
    policy_rel_diffs,
    require_converged,
    solve_compact,
    solve_shipped,
    stage0_mesh,
)
from ssj.interp_loop import solve_interp

logger = logging.getLogger(__name__)

_INTERP = [
    ("ENGINE", "engine"),
    ("Curvilinear", "curvilinear"),
    ("Delaunay", "delaunay"),
]
_INTERIOR_FRAC = 0.85
_REPEATS = 3
_THREADS = 16


def midpoint_mesh(cal):
    """Both states' midpoints between adjacent solver nodes, crossed into a
    ``(b, a)`` mesh: the shared query-point universe ``query_points`` and
    ``query_points_band`` each filter by ``_INTERIOR_FRAC``.
    """
    b, a = cal["b"], cal["a"]
    bm = 0.5 * (b[:-1] + b[1:])
    am = 0.5 * (a[:-1] + a[1:])
    return np.meshgrid(bm, am, indexing="ij")


def query_points(cal):
    """Off-grid interior query points: midpoints between adjacent solver nodes,
    truncated below ``_INTERIOR_FRAC`` of each axis maximum, crossed over both states.
    """
    b, a = cal["b"], cal["a"]
    bq, aq = midpoint_mesh(cal)
    keep = (bq < _INTERIOR_FRAC * b[-1]) & (aq < _INTERIOR_FRAC * a[-1])
    return bq[keep].ravel(), aq[keep].ravel()


def query_points_band(cal):
    """High-wealth band the interior truncation drops: axis midpoints with either
    coordinate at or above ``_INTERIOR_FRAC`` of its axis maximum, crossed over
    both states. Still inside the regular grid (so scoring never extrapolates),
    but in the region whose converged values the realign step reaches only
    through its boundary-slope extrapolation.
    """
    b, a = cal["b"], cal["a"]
    bq, aq = midpoint_mesh(cal)
    keep = (bq >= _INTERIOR_FRAC * b[-1]) | (aq >= _INTERIOR_FRAC * a[-1])
    return bq[keep].ravel(), aq[keep].ravel()


def index_ring(z):
    """Values of ``z`` along its four outer index lines, once around from node
    ``(0, 0)``.
    """
    return np.concatenate([z[0, :], z[1:, -1], z[-1, -2::-1], z[-2:0:-1, 0]])


def mesh_boundary(x, y):
    """Outer polygon of a tensor-indexed, fold-free mesh."""
    return np.column_stack([index_ring(x), index_ring(y)])


def points_in_polygon(points, polygon):
    """Classify points in a polygon, counting boundary points as inside."""
    px, py = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    on_boundary = np.zeros(points.shape[0], dtype=bool)
    scale = max(float(np.ptp(polygon[:, 0])), float(np.ptp(polygon[:, 1])), 1.0)
    tol = 1e-10 * scale
    for i in range(polygon.shape[0]):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % polygon.shape[0]]
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy
        if length_sq == 0.0:
            continue
        length = np.sqrt(length_sq)
        cross = dx * (py - y0) - dy * (px - x0)
        projection = (px - x0) * dx + (py - y0) * dy
        on_boundary |= (
            (np.abs(cross) <= tol * length)
            & (projection >= -tol * length)
            & (projection <= length_sq + tol * length)
        )
        if dy != 0.0:
            crosses = (y0 > py) != (y1 > py)
            x_cross = x0 + (py - y0) * dx / dy
            inside ^= crosses & (px < x_cross)
    return inside | on_boundary


def extrapolation_counts(cal, out):
    """Count Stage-0 regular-grid targets outside the converged mesh image.

    A fold-free bilinear mesh is homeomorphic to its tensor-index rectangle, so
    its interpolation region is the polygon bounded by the four outer index
    lines. Targets outside that polygon invoke the interpolator's boundary-slope
    extension. Counts are returned by income state and pooled across states.
    """
    mesh = stage0_mesh(out, cal)
    assert_mesh_fold_free(mesh["b_endo"], mesh["a_endo"])
    assert_mesh_ipm(mesh["b_endo"], mesh["a_endo"])
    bq, aq = node_mesh(cal)
    points = np.column_stack([bq.ravel(), aq.ravel()])
    rows = []
    n_outside_total = 0
    for ie in range(cal["Pi"].shape[0]):
        boundary = mesh_boundary(mesh["b_endo"][ie], mesh["a_endo"][ie])
        n_outside = int(np.sum(~points_in_polygon(points, boundary)))
        n_outside_total += n_outside
        rows.append(
            {
                "income_state": ie,
                "n_queries": points.shape[0],
                "n_outside": n_outside,
                "outside_fraction": n_outside / points.shape[0],
            },
        )
    n_queries_total = points.shape[0] * cal["Pi"].shape[0]
    pooled = {
        "income_state": "all",
        "n_queries": n_queries_total,
        "n_outside": n_outside_total,
        "outside_fraction": n_outside_total / n_queries_total,
    }
    return rows, pooled


def require_policies_in_domain(out, cal, query) -> None:
    """Raise unless the policies ``(b', a')`` linearly interpolated at ``query``
    lie inside the regular grid, so scoring never extrapolates the continuation.
    The node policies themselves can leave it at the high-wealth corner.
    """
    b, a = cal["b"], cal["a"]
    pts = np.column_stack(query)
    for ie in range(cal["Pi"].shape[0]):
        bp, ap = (grid_interp(out[k][ie], cal)(pts) for k in "ba")
        oob = max(bp.max() - b[-1], b[0] - bp.min(), ap.max() - a[-1], a[0] - ap.min())
        if oob > 0:
            msg = f"interpolated policy leaves the grid domain by {oob:.3e}"
            raise RuntimeError(msg)


def residual_log10(cal, out, query=None):
    """Median, p95, max of ``log10`` of the robust consumption-unit Euler error
    ``|liquid_euler_gap|`` at ``query`` (default: the off-grid interior set).
    ``c_implied = min(Euler-implied, constraint-bound)``, so the error is defined
    at every query and every query is scored: an exact zero scores ``-inf`` and
    a non-finite error raises.
    """
    query = query_points(cal) if query is None else query
    require_policies_in_domain(out, cal, query)
    err = np.abs(liquid_euler_gap(out, cal, out["Vb"], query))
    if not np.isfinite(err).all():
        msg = "non-finite Euler error at a query point"
        raise RuntimeError(msg)
    with np.errstate(divide="ignore"):
        return median_p95_max(np.log10(err))


def time_solve(cal, interp):
    """Best-of-three warmed full solve time and converged output for one interpolator."""
    solve = functools.partial(solve_interp, cal, interp)
    out = solve()  # warm
    if not out["converged"]:
        return np.inf, out
    best, outs = best_time(solve, _REPEATS, warm=False)
    require_converged(
        all(o["converged"] for o in outs),
        f"timed repetition failed to converge ({interp}); "
        "the warm solve converged, so the run is not reproducible",
    )
    return best, outs[-1]


def measure():
    """One record per ``(grid label, interpolator key)``: the best-of-three solve
    ``time``, the solve ``out`` and, when it converged, the interior off-grid
    residual ``stats``. The readings downstream reuse these solves.
    """
    records = {}
    for label, nB, nA in INTERP_GRIDS:
        cal = calibration(nB, nA)
        for name, key in _INTERP:
            t, out = time_solve(cal, key)
            rec = records[label, key] = {"time": t, "out": out, "stats": None}
            if out["converged"]:
                rec["stats"] = residual_log10(cal, out)
                logger.info(
                    "%-7s %-12s %.3fs it=%d med=%.2f p95=%.2f max=%.2f",
                    label,
                    name,
                    t,
                    out["iter"],
                    *rec["stats"],
                )
            else:
                logger.info(
                    "%-7s %-12s fails (%s, it=%d)",
                    label,
                    name,
                    out["reason"],
                    out["iter"],
                )
    return records


def persisted_cell(rec):
    """A converged cell's ``(time, median, p95, max)`` strings at the precision
    ``solvetime_abrs.csv`` stores; the table displays these same values, so it
    regenerates from the CSV exactly.
    """
    return [f"{rec['time']:.4f}", *(f"{s:.3f}" for s in rec["stats"])]


def write_solvetime_table(records) -> None:
    """Write the paper table. Only converged solves appear as rows; an
    interpolator that does not converge (Delaunay reaches the iteration cap) is
    recorded in ``solvetime_abrs.csv`` and the run log, not displayed as a row of
    failure cells.
    """
    cap = (
        "Full solve time and converged-policy accuracy with ENGINE or the curvilinear "
        "method of \\citet{White2015} as the Stage-0 regrid of an otherwise identical "
        "EGM$^n$ solve. ``Solve (s)'' is the best of three warmed solves to tolerance "
        "$10^{-9}$. Residuals are the median, 95th percentile, and maximum of the "
        "$\\log_{10}$ robust consumption-unit Euler error at off-grid midpoints below 85 "
        "percent of each grid maximum."
    )
    head = [
        r"Grid & Interpolator & Solve (s) & \multicolumn{3}{c}{$\log_{10}$ Euler error} \\",
        r"\cmidrule(l){4-6}",
        r"& & & Median & p95 & Max \\",
    ]
    lines = []
    for label, nB, nA in INTERP_GRIDS:
        glabel = f"{label} (${nB} \\times {nA}$)"
        for name, key in _INTERP:
            rec = records[label, key]
            if not rec["out"]["converged"]:
                continue
            time, med, p95, mx = map(float, persisted_cell(rec))
            lines.append(
                f"{glabel} & {name} & {time:.3f} & ${med:.2f}$ & ${p95:.2f}$ "
                f"& ${mx:.2f}$ " + r"\\",
            )
    table = (cap, "tab-solvetime-abrs", r"\footnotesize", "4pt", "@{}llrrrr@{}")
    write_latex_table(FIG_DIR / "tab_solvetime_abrs.md", table, head, lines)


def require_structured_converged(records) -> None:
    """Both structured interpolants must converge at every grid; Delaunay's
    non-convergence at the iteration cap is an expected, recorded outcome.
    Called after the CSV write so a failure leaves its diagnostics on disk.
    """
    missing = [
        f"{label}/{name}"
        for label, _, _ in INTERP_GRIDS
        for name, key in _INTERP[:2]
        if not records[label, key]["out"]["converged"]
    ]
    delaunay = [
        (label, records[label, "delaunay"]["out"]["reason"])
        for label, _, _ in INTERP_GRIDS
    ]
    unexpected = [
        f"{label}/Delaunay ({reason})"
        for label, reason in delaunay
        if reason not in {"", "no_convergence"}
    ]
    if missing or unexpected:
        raise RuntimeError(
            "unexpected solver outcome: "
            + ", ".join(missing + unexpected)
            + " (diagnostics recorded in solvetime_abrs.csv; only Delaunay's"
            " non-convergence at the iteration cap is expected)",
        )


def write_solvetime_csv(records) -> None:
    """Raw per-cell data behind ``tab_solvetime_abrs.md``, one row per
    (grid, interpolator). A failed cell persists its failure mechanism
    (``no_convergence``, ``nan_policy`` or ``nan_value``) and the iteration it
    stopped at, so the archive tells non-convergence from a non-finite abort.
    """
    rows = []
    for label, nB, nA in INTERP_GRIDS:
        for _name, key in _INTERP:
            rec = records[label, key]
            out = rec["out"]
            if out["converged"]:
                time, *stats = persisted_cell(rec)
                cell = ["ok", time, out["iter"], *stats]
            else:
                cell = [f"fails_{out['reason']}", "", out["iter"], "", "", ""]
            rows.append([label.lower(), nB, nA, key, *cell])
    write_csv(
        "solvetime_abrs.csv",
        [
            "resolution",
            "nB",
            "nA",
            "method",
            "status",
            "solve_seconds",
            "iters",
            "log10_median",
            "log10_p95",
            "log10_max",
        ],
        rows,
    )


def write_refinement_bounds_table(rows) -> None:
    """Write the raw-LaTeX table generated from refinement and bounds rows."""
    cap = (
        "Consumption-policy refinement and upper-bound sensitivity, as relative "
        "differences in percent over a common set of coarse-grid interior midpoints "
        "and all income states. "
        "Refinement rows compare each grid with the "
        f"${BENCHMARK_GRID[0]} \\times {BENCHMARK_GRID[1]}$ benchmark; the "
        "bounds row compares the medium grid with both state-grid maxima scaled by $1.5$."
    )
    lines = []
    for row in rows:
        exercise, _label, nB, nA, med, p95, mx = row[:7]
        exercise_label = (
            "Refinement" if exercise == "refinement" else "Bounds $\\times 1.5$"
        )
        lines.append(
            f"{exercise_label} & {grid_label((nB, nA))} & {100 * float(med):.2f} & "
            f"{100 * float(p95):.2f} & {100 * float(mx):.2f} " + r"\\",
        )
    table = (cap, "tab-refinement-bounds-abrs", r"\small", "5pt", "@{}llrrr@{}")
    head = [r"Exercise & Grid & Median & p95 & Maximum \\"]
    write_latex_table(FIG_DIR / "tab_refinement_bounds_abrs.md", table, head, lines)


def refinement_bounds_reading(solutions=None):
    """Persist refinement/bounds readings and return the refinement solutions.

    ``solutions`` maps each grid label to its converged compact solve (the
    ENGINE run of ``measure``, bit-identical to ``solve_compact``); grids it
    lacks are solved here.
    """
    rb_rows = []
    solutions = dict(solutions or {})
    bench = calibration(*BENCHMARK_GRID)
    cb = solve_compact(bench)
    pts = np.column_stack(query_points(calibration(*INTERP_GRIDS[0][1:])))
    for label, nB, nA in INTERP_GRIDS:
        cal = calibration(nB, nA)
        if label not in solutions:
            solutions[label] = solve_compact(cal)
        out = solutions[label]
        med, p95, mx = median_p95_max(policy_rel_diffs(out, cal, cb, bench, pts))
        rb_rows.append(
            [
                "refinement",
                label.lower(),
                nB,
                nA,
                f"{med:.8f}",
                f"{p95:.8f}",
                f"{mx:.8f}",
                out["iter"],
                cb["iter"],
            ],
        )
        logger.info(
            "refinement %s vs %dx%d: med=%.2f%% max=%.2f%%",
            label,
            *BENCHMARK_GRID,
            100 * med,
            100 * mx,
        )

    base_label, nB, nA = INTERP_GRIDS[1]
    base = calibration(nB, nA)
    ext = calibration(nB, nA)
    ext["b"] = ext["b"] * 1.5
    ext["a"] = ext["a"] * 1.5
    ob, oe = solutions[base_label], solve_compact(ext)
    pts = np.column_stack(query_points(base))
    med, p95, mx = median_p95_max(policy_rel_diffs(oe, ext, ob, base, pts))
    rb_rows.append(
        [
            "bounds",
            f"{base_label.lower()}_x1.5",
            nB,
            nA,
            f"{med:.8f}",
            f"{p95:.8f}",
            f"{mx:.8f}",
            oe["iter"],
            ob["iter"],
        ],
    )
    logger.info(
        "bounds x1.5 medium: med=%.2f%% max=%.2f%%",
        100 * med,
        100 * mx,
    )
    write_csv(
        "refinement_bounds_abrs.csv",
        [
            "exercise",
            "label",
            "nB",
            "nA",
            "rel_median",
            "rel_p95",
            "rel_max",
            "iters",
            "iters_ref",
        ],
        rb_rows,
    )
    write_refinement_bounds_table(rb_rows)
    return solutions


def write_extrapolation_table(rows) -> None:
    """Write pooled Stage-0 extrapolation frequencies as a raw-LaTeX table."""
    cap = (
        "Stage-0 extrapolation frequency at the converged fixed point: regular $(b,a)$ "
        "targets outside the endogenous mesh, pooled over income states. Boundary "
        "points count as interpolation."
    )
    lines = [
        f"{grid_label((nB, nA))} & {pooled['n_outside']} / {pooled['n_queries']} & "
        f"{100 * pooled['outside_fraction']:.3f} " + r"\\"
        for _label, nB, nA, pooled in rows
    ]
    table = (cap, "tab-extrap-frequency-abrs", r"\small", "6pt", "@{}lrr@{}")
    head = [r"Grid & Outside / total & Outside (percent) \\"]
    write_latex_table(FIG_DIR / "tab_extrap_frequency_abrs.md", table, head, lines)


def extrapolation_frequency_reading(solutions) -> None:
    """Persist per-income and pooled Stage-0 extrapolation frequencies of the
    converged compact ``solutions``, keyed by grid label.
    """
    csv_rows = []
    table_rows = []
    for label, nB, nA in INTERP_GRIDS:
        cal = calibration(nB, nA)
        out = solutions[label]
        by_income, pooled = extrapolation_counts(cal, out)
        for row in [*by_income, pooled]:
            csv_rows.append(
                [
                    label.lower(),
                    nB,
                    nA,
                    row["income_state"],
                    row["n_queries"],
                    row["n_outside"],
                    f"{row['outside_fraction']:.8f}",
                    out["iter"],
                ],
            )
        table_rows.append((label, nB, nA, pooled))
        logger.info(
            "Stage-0 extrapolation %s: %d/%d = %.3f%%",
            label,
            pooled["n_outside"],
            pooled["n_queries"],
            100 * pooled["outside_fraction"],
        )
    write_csv(
        "extrap_frequency_abrs.csv",
        [
            "resolution",
            "nB",
            "nA",
            "income_state",
            "n_queries",
            "n_outside",
            "outside_fraction",
            "iters",
        ],
        csv_rows,
    )
    write_extrapolation_table(table_rows)


def aux_readings(records) -> None:
    """Persist speed, SSJ-anchor, refinement, bounds, and extrapolation readings,
    reusing ``measure``'s ENGINE solves as the compact solutions. Runs after
    ``require_structured_converged``, so both structured cells converged.
    """
    for label, _, _ in INTERP_GRIDS:
        logger.info(
            "speed %s: ENGINE %.2fx faster than curvilinear",
            label,
            records[label, "curvilinear"]["time"] / records[label, "engine"]["time"],
        )
    anchor_rows = []
    for label, nB, nA in INTERP_GRIDS:
        cal = calibration(nB, nA)
        med, p95, mx = residual_log10(cal, solve_shipped(cal))
        anchor_rows.append(
            [label.lower(), nB, nA, f"{med:.3f}", f"{p95:.3f}", f"{mx:.3f}"],
        )
        logger.info("SSJ anchor %s: med=%.2f p95=%.2f max=%.2f", label, med, p95, mx)
    write_csv(
        "ssj_anchor_abrs.csv",
        ["resolution", "nB", "nA", "log10_median", "log10_p95", "log10_max"],
        anchor_rows,
    )

    solutions = refinement_bounds_reading(
        {label: records[label, "engine"]["out"] for label, _, _ in INTERP_GRIDS},
    )
    extrapolation_frequency_reading(solutions)


def order_reading(records) -> None:
    """Archive the reversed-pass-order solve (Section 4's order subsection): the
    reversal cost relative to the forward order, and the converged-policy distance
    to ENGINE and to curvilinear. With the Euler correction all pass orders agree
    closely, so the order governs the worst-case guarantee, not converged accuracy.
    Persisted to ``order_reading_abrs.csv``. The forward-order and curvilinear
    solves (and the forward time) come from ``measure``'s ``records``, already
    checked converged by ``require_structured_converged``.
    """
    out = []
    for label, nB, nA in INTERP_GRIDS:
        cal = calibration(nB, nA)
        te, e = records[label, "engine"]["time"], records[label, "engine"]["out"]
        c = records[label, "curvilinear"]["out"]
        tr, r = time_solve(cal, "engine_reversed")
        require_converged(
            r["converged"],
            f"order reading: engine_reversed failed to converge at {label}",
        )
        de = float(np.median(rel_diff(r["c"], e["c"])))
        dc = float(np.median(rel_diff(r["c"], c["c"])))
        med = residual_log10(cal, r)[0]
        out.append(
            [
                label.lower(),
                nB,
                nA,
                f"{tr / te:.3f}",
                f"{med:.3f}",
                f"{de:.3e}",
                f"{dc:.3e}",
            ],
        )
        logger.info(
            "order %s: reversal cost %.2fx, rev residual med=%.2f, policy dist engine=%.1e curvilinear=%.1e",
            label,
            tr / te,
            med,
            de,
            dc,
        )
    write_csv(
        "order_reading_abrs.csv",
        [
            "resolution",
            "nB",
            "nA",
            "reversal_cost",
            "rev_log10_median",
            "policy_dist_engine",
            "policy_dist_curvilinear",
        ],
        out,
    )


def band_reading(records=None) -> None:
    """Robust Euler residual of the converged policy on the high-wealth band the
    interior truncation excludes (the appendix's extrapolation-band reading), for
    the two structured interpolators. Persisted to ``extrap_band_abrs.csv``,
    self-contained: the CSV also records the same solve's interior statistics, so the
    band-vs-interior and cross-method comparisons never depend on another run's CSV.
    ``residual_log10`` scores every query or raises, so each scored count equals
    its expected count. Reuses ``measure``'s ``records`` (solves and interior
    statistics) when given, else solves and scores each method here.
    """
    out_rows = []
    for label, nB, nA in INTERP_GRIDS:
        cal = calibration(nB, nA)
        band = query_points_band(cal)
        n_states = cal["Pi"].shape[0]
        n_band = band[0].size * n_states
        n_interior = query_points(cal)[0].size * n_states
        for name, key in _INTERP[:2]:
            if records is None:
                sol = solve_interp(cal, key)
                require_converged(
                    sol["converged"],
                    f"band reading: {name} failed to converge at {label}",
                )
                interior = residual_log10(cal, sol)
            else:
                sol, interior = records[label, key]["out"], records[label, key]["stats"]
            bmed, bp95, bmax = residual_log10(cal, sol, query=band)
            imed, ip95, imax = interior
            out_rows.append(
                [
                    label.lower(),
                    nB,
                    nA,
                    key,
                    n_band,
                    n_band,
                    f"{bmed:.4f}",
                    f"{bp95:.4f}",
                    f"{bmax:.4f}",
                    n_interior,
                    n_interior,
                    f"{imed:.4f}",
                    f"{ip95:.4f}",
                    f"{imax:.4f}",
                ],
            )
            logger.info(
                "band %-7s %-12s n=%d med=%.2f p95=%.2f max=%.2f | "
                "interior n=%d med=%.2f max=%.2f",
                label,
                name,
                n_band,
                bmed,
                bp95,
                bmax,
                n_interior,
                imed,
                imax,
            )
    write_csv(
        "extrap_band_abrs.csv",
        [
            "resolution",
            "nB",
            "nA",
            "method",
            "band_n_scored",
            "band_n_expected",
            "band_log10_median",
            "band_log10_p95",
            "band_log10_max",
            "interior_n_scored",
            "interior_n_expected",
            "interior_log10_median",
            "interior_log10_p95",
            "interior_log10_max",
        ],
        out_rows,
    )


def main(band_only: bool = False, validation_only: bool = False) -> None:
    start_logging()
    if band_only or validation_only:
        # These artifacts are deterministic and thread-invariant; clamp so a
        # replicator with fewer cores can still regenerate them.
        numba.set_num_threads(min(_THREADS, numba.config.NUMBA_NUM_THREADS))
    else:
        require_threads(
            _THREADS,
            f"timing protocol requires {_THREADS} threads (--band-only and "
            "--validation-only regenerate the thread-invariant artifacts)",
        )
        numba.set_num_threads(_THREADS)
    if band_only:
        band_reading()
        log_done("extrap_band_abrs.csv")
        return
    if validation_only:
        solutions = refinement_bounds_reading()
        extrapolation_frequency_reading(solutions)
        log_done(
            "refinement_bounds_abrs.csv",
            "tab_refinement_bounds_abrs.md",
            "extrap_frequency_abrs.csv",
            "tab_extrap_frequency_abrs.md",
        )
        return
    records = measure()
    write_solvetime_csv(records)
    require_structured_converged(records)
    write_solvetime_table(records)
    order_reading(records)
    aux_readings(records)
    band_reading(records)
    log_done(
        "tab_solvetime_abrs.md",
        "solvetime_abrs.csv",
        "ssj_anchor_abrs.csv",
        "refinement_bounds_abrs.csv",
        "tab_refinement_bounds_abrs.md",
        "extrap_frequency_abrs.csv",
        "tab_extrap_frequency_abrs.md",
        "order_reading_abrs.csv",
        "extrap_band_abrs.csv",
    )


if __name__ == "__main__":
    _p = argparse.ArgumentParser(description=__doc__)
    _mode = _p.add_mutually_exclusive_group()
    _mode.add_argument(
        "--band-only",
        action="store_true",
        help="regenerate only extrap_band_abrs.csv (leaves timing artifacts untouched)",
    )
    _mode.add_argument(
        "--validation-only",
        action="store_true",
        help="regenerate refinement, bounds, and extrapolation-frequency artifacts",
    )
    args = _p.parse_args()
    main(band_only=args.band_only, validation_only=args.validation_only)
