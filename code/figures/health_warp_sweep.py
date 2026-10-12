"""Warpedness sweep on the health model: accuracy and cost vs MEASURED warp.

How do the interpolation methods behave on more warped or squeezed grids, and
how does efficiency relate to grid spacing and anisotropy? This sweep answers
both on economically reachable calibrations of the
health model (the model class curvilinear interpolation was designed for):
the production exponent ``alpha`` sets the curvature of ``g(n)`` and
``DieProbMax`` scales the survival channel, so together they modulate how
violently the investment inversion bends the endogenous mesh.

Per sweep point (alpha, DieProbMax) x interpolator the script reports:

1. MEASURED warpedness of the endogenous ``(m, h)`` mesh: its count of NaN
   nodes, the count of steps violating Index-Preserved Monotonicity (a step
   touching a NaN node included), the empirical bracket-displacement constant
   ``alpha_hat`` when IPM holds (NaN otherwise) (computed by the SAME helper as
   the regrid-scaling bench, so the numbers are commensurable with Section 4's
   constant alpha), and the count of folded cells (any non-positive corner
   Jacobian) in total and among those the off-mesh query grid reaches, with
   the share of the final queries inside them and the largest per-period
   share of the solve's own continuation points inside a later mesh's folds.
2. Truth-anchored accuracy: both FOC residuals of the period-0 policy at
   off-mesh queries, with the post-decision marginals ``w_a, w_H`` rebuilt by
   QUADRATURE through the TRUE continuation ``solution[1]`` (production
   ``calc_exp_next``), not read from interpolated marginals. The consumption
   residual uses the min-with-constraint form (implied c capped at the a >= 0
   budget bound); the investment residual anchors on the intratemporal margin
   ``g'(n) w_H = u'(c)``, which holds in BOTH the interior and the
   liquidity-constrained regime. Infeasible policy outputs (non-positive
   consumption, negative investment, or a budget overspent at all, a < 0)
   are COUNTED in ``n_policy_viol``, never dropped, and ``max_overspend_rel``
   records the largest overspend relative to m; the only exclusions are
   non-finite residuals, reported in ``res_*_nbad``.
3. Completion and wall-clock solve time.

The model is solved for five periods (``cycles=5``) so warp accumulates over
several backward steps, while the residual stays exact: the FOC of
``solution[0]`` is evaluated against its actual continuation ``solution[1]``.

Writes ``content/figures/health_warp_sweep.csv`` and a three-panel figure
``health_warp_sweep.(png|pdf)``. Run from the repo root::

    PYTHONPATH=code uv run python code/figures/health_warp_sweep.py

The sweep is expensive (27 solves); to only regenerate the figure from the
already-written CSV without re-solving anything, pass ``--plot-only``::

    PYTHONPATH=code uv run python code/figures/health_warp_sweep.py --plot-only
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cli import start_logging
from engine.utilities import (
    Delaunay2DInterp,
    bilinear_contains,
    empirical_ipo_alpha,
    ipm_violations,
    nan_nodes,
    reached_folds,
)
from figure_styles import (
    ENGINE_COLORS,
    apply_engine_style,
    format_axes_for_publication,
    save_engine_figure,
)
from HARK.ConsumptionSaving.ConsHealthModel import eval_health_prod
from HARK.interpolation import Curvilinear2DInterp
from HARK.rewards import CRRAutilityP, CRRAutilityP_inv
from HARKmodels.ConsHealthSeparableModel import (
    BasicHealthConsumerType,
    init_basic_health,
    inv_marg_health_prod,
    policy_violations,
    post_decision_marginals,
)
from HARKmodels.health_interp import HealthEngine2DInterp, continuation_queries
from matplotlib.colors import to_rgb
from numerics import median_p95_max
from repo_paths import FIG_DIR, read_rows, write_rows
from timing import call_times

logger = logging.getLogger(__name__)

# solve_s is the best of this many solves after a warm one, kept to 4 significant
# digits: an ENGINE solve here takes about 10 ms.
_TIMING_REPS = 5
_OUT_CSV = os.fspath(FIG_DIR / "health_warp_sweep.csv")
# Sweep: production curvature x mortality scale (logistic(c) = DieProbMax).
_ALPHAS = [0.20, 0.35, 0.50]
_DIEPROB_COEFFS = [-2.0, 0.0, 1.5]  # DieProbMax ~ 0.12 / 0.50 / 0.82
# ENGINE's search margin at the sweep's own max measured bracket-displacement
# constant (16), a speed setting: values are identical at the kernel default of 2.
_INTERPOLATORS = [
    ("curvilinear", Curvilinear2DInterp),
    ("engine", HealthEngine2DInterp),
    ("delaunay", Delaunay2DInterp),
]
_GRID = {"aXtraCount": 25, "hLvlCount": 20, "T_cycle": 1, "cycles": 5}
_M_RANGE = (0.5, 25.0)
_H_RANGE = (0.5, 10.0)

# CSV columns, in order; ``load_rows_from_csv`` restores the integer columns'
# dtype and parses every other non-text column as a float.
_COLUMNS = (
    "alpha",
    "die_prob_max",
    "interpolator",
    "status",
    "solve_s",
    "ipo_alpha_hat",
    "n_nodes",
    "ipm_violations",
    "nan_nodes",
    "off_hull_queries",
    "folds_total",
    "folds_reached",
    "fold_final_query_share",
    "fold_continuation_share",
    "res_c_med",
    "res_c_p95",
    "res_c_max",
    "res_c_nbad",
    "res_n_med",
    "res_n_p95",
    "res_n_max",
    "res_n_nbad",
    "n_policy_viol",
    "max_overspend_rel",
    "failure",
)
_TEXT_COLS = {"interpolator", "status", "failure"}
_LABELS = {"curvilinear": "Curvilinear", "engine": "ENGINE", "delaunay": "Delaunay"}
_INT_COLS = {
    "n_nodes",
    "ipm_violations",
    "nan_nodes",
    "off_hull_queries",
    "folds_total",
    "folds_reached",
    "res_c_nbad",
    "res_n_nbad",
    "n_policy_viol",
}


def sweep_agent(alpha, coeff, interpolator):
    """Sweep-grid health agent at curvature ``alpha`` and mortality ``coeff``."""
    params = {**init_basic_health, **_GRID}
    params["HealthProdExp"] = alpha
    params["DieProbMaxCoeffs"] = [coeff]
    params["interpolator"] = interpolator
    return BasicHealthConsumerType(**params)


def mesh_warp_stats(m_mesh, h_mesh):
    """``(alpha_hat, ipm_violations, nan_nodes)`` of the endogenous mesh.

    ``empirical_ipo_alpha`` is the paper's own ex-post measurement, shared
    with the regrid-scaling bench so the numbers are commensurable with
    Section 4. It presupposes Index-Preserved Monotonicity (``def-ipm``):
    each h-slice of the m mesh strictly increasing along the asset index, and
    each asset-row of the h mesh strictly increasing along the health index.
    ``ipm_violations`` counts the steps breaking it, a step touching a NaN
    node included; with any, the bracket-displacement constant is undefined and
    ``alpha_hat`` is NaN.
    """
    n_nan = nan_nodes(m_mesh, h_mesh)
    n_bad = sum(ipm_violations(m_mesh, h_mesh))
    if n_bad > 0:
        return np.nan, n_bad, n_nan
    return empirical_ipo_alpha(m_mesh), 0, n_nan


def fold_stats(m_mesh, h_mesh, q):
    """``(total, reached, share)`` of the folded cells (any non-positive corner
    Jacobian, ``def-fold-free``) with four finite corners: their count, how many
    contain a point of ``q`` (shape ``(n, 2)``), and the share of ``q`` inside
    any of them. A cell with a NaN corner is undefined and is counted by
    ``nan_nodes``, never as a fold.
    """
    folded, reached = reached_folds(m_mesh, h_mesh, q)
    pts = np.stack([m_mesh, h_mesh], axis=-1)
    inside = np.zeros(len(q), dtype=bool)
    n_folds = n_reached = 0
    for (j, k), hit in zip(folded, reached, strict=True):
        if not np.isfinite(pts[j : j + 2, k : k + 2]).all():
            continue
        n_folds += 1
        if hit:
            n_reached += 1
            inside |= bilinear_contains(
                (pts[j, k], pts[j + 1, k], pts[j, k + 1], pts[j + 1, k + 1]),
                q,
            )
    return n_folds, n_reached, float(inside.mean())


def query_grid(n_pts=60):
    """The sweep's off-mesh ``(m, h)`` query mesh, ``n_pts x (n_pts - 10)``."""
    m = np.linspace(*_M_RANGE, n_pts)
    h = np.linspace(*_H_RANGE, n_pts - 10)
    return np.meshgrid(m, h, indexing="ij")


def foc_residuals(agent, n_pts=60):
    """Truth-anchored FOC residuals of the period-0 policy at off-mesh queries.

    The post-decision marginals are rebuilt by quadrature through the TRUE
    continuation ``agent.solution[1]``, so the only approximation under test
    is the interpolant inside ``solution[0]``. Min-with-constraint discipline:
    the implied consumption is capped at the a >= 0 budget bound, the implied
    investment anchors on the intratemporal margin ``g'(n) w_H = u'(c)``
    (valid at interior investment whether or not the liquidity constraint
    binds; the Inada condition on g keeps investment interior), and
    infeasible policy outputs (c <= 0, n < 0, or a < 0) are counted, never
    dropped: n < 0 and a < 0 are projected onto the feasible set for
    evaluation, and c <= 0, where neither FOC is defined, scores an infinite
    residual (counted in each residual's ``nbad``).
    """
    sol = agent.solution[0]
    sol_next = agent.solution[1]
    rho, alpha, gamma = agent.CRRA, agent.HealthProdExp, agent.HealthProdFac
    R, beta, phi = agent.Rfree[0], agent.DiscFac, agent.DieProbMax[0]
    dstn = agent.ShockDstn[0]

    M, H_q = query_grid(n_pts)
    _, c, n_raw = sol(M, H_q)
    # Count every policy-output violation BEFORE repairing for evaluation.
    violations = policy_violations(M, c, n_raw)
    n = np.maximum(n_raw, 0.0)
    a = M - c - n
    c_pos = c > 0.0
    c_safe = np.where(c_pos, c, 1.0)
    a_q = np.maximum(a, 0.0)  # clip for quadrature; violation still scored
    H_post = H_q + eval_health_prod(n, alpha, gamma)

    # Post-decision marginals by quadrature through the true continuation.
    _, w_a, w_H = post_decision_marginals(
        sol_next,
        dstn,
        a_q,
        H_post,
        (R, rho, alpha, gamma),
        phi,
        beta,
    )

    # Consumption FOC, min-with-constraint: implied c capped at a >= 0 bound.
    # At w_a = 0 the inversion is +inf and the constraint binds, as it should.
    c_implied = np.minimum(CRRAutilityP_inv(w_a, rho), M - n)
    res_c = np.where(c_pos, np.abs(c - c_implied) / c_safe, np.inf)

    # Investment FOC via g'(n) w_H = u'(c), valid whether or not the liquidity
    # constraint binds, compared in levels over 1 + n so n ~ 0 stays bounded.
    # At w_H = 0 the ratio is +inf and the inversion gives its limit n = 0.
    n_implied = np.minimum(
        inv_marg_health_prod(CRRAutilityP(c_safe, rho) / w_H, alpha, gamma),
        np.maximum(M - c, 0.0),
    )
    res_n = np.where(c_pos, np.abs(n - n_implied) / (1.0 + n), np.inf)

    def stats(r):
        finite = np.isfinite(r)
        return (*median_p95_max(r[finite]), int(np.sum(~finite)))

    return stats(res_c), stats(res_n), *violations


def interp_mesh(interp):
    """Endogenous ``(m, h)`` mesh a solution interpolant was built on."""
    for xattr, yattr in (("x_values", "y_values"), ("fx", "fy")):
        if hasattr(interp, xattr):
            return np.asarray(getattr(interp, xattr)), np.asarray(
                getattr(interp, yattr),
            )
    msg = f"{type(interp).__name__} exposes no endogenous mesh"
    raise AttributeError(msg)


def run_sweep():
    """Solve the full (alpha, DieProbMax) x interpolator grid and write the CSV.

    Expensive: 27 solves plus a warmup. Returns the same row structure that
    ``load_rows_from_csv`` reconstructs from the written file, so callers
    can feed either straight into ``plot_sweep``.
    """
    # Warm every interpolator: the first solve under each in a process pays
    # one-time import/JIT costs (the smoke run showed 4.2 s vs a 0.04 s steady
    # state), which would contaminate its first timed row.
    for _, interp_cls in _INTERPOLATORS:
        sweep_agent(_ALPHAS[0], _DIEPROB_COEFFS[0], interp_cls).solve()
    rows = [
        sweep_row(a_exp, coeff, name, interp_cls)
        for a_exp in _ALPHAS
        for coeff in _DIEPROB_COEFFS
        for name, interp_cls in _INTERPOLATORS
    ]
    write_rows(_OUT_CSV, _COLUMNS, [[row.get(c, "") for c in _COLUMNS] for row in rows])
    logger.info("wrote %s", _OUT_CSV)
    return rows


def sweep_row(a_exp, coeff, name, interp_cls):
    """One CSV row, as a dict by column: solve at ``(a_exp, coeff)`` under
    ``interp_cls``, then measure its mesh and FOC residuals; a failed solve
    keeps only the calibration, status and error.
    """
    die_prob = 1.0 / (1.0 + np.exp(-coeff))
    row = {"alpha": a_exp, "die_prob_max": round(die_prob, 3), "interpolator": name}
    agent = sweep_agent(a_exp, coeff, interp_cls)
    try:
        agent.solve()
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        logger.info("alpha=%.2f D=%.2f %-11s FAILED: %s", a_exp, die_prob, name, exc)
        return {**row, "status": "failed", "failure": str(exc)[:80]}
    secs = min(call_times(agent.solve, _TIMING_REPS))
    # Queries the solve sent outside a triangulation's hull (Delaunay only).
    off_hull = sum(getattr(sol, "off_hull_queries", 0) for sol in agent.solution)
    mx, my = interp_mesh(agent.solution[0])
    a_hat, ipm_bad, n_nan = mesh_warp_stats(mx, my)
    final = np.column_stack([a.ravel() for a in query_grid()])
    n_folds, n_reached, final_share = fold_stats(mx, my, final)
    # The continuation points are the same every period; score each later mesh.
    continuation = continuation_queries(agent)
    continuation_share = max(
        fold_stats(*interp_mesh(sol), continuation)[2] for sol in agent.solution[1:]
    )
    stats_c, stats_n, n_infeas, overspend = foc_residuals(agent)
    row.update(
        status="ok",
        solve_s=float(f"{secs:.4g}"),
        ipo_alpha_hat=a_hat,
        n_nodes=mx.size,
        ipm_violations=ipm_bad,
        nan_nodes=n_nan,
        off_hull_queries=off_hull if name == "delaunay" else "",
        folds_total=n_folds,
        folds_reached=n_reached,
        fold_final_query_share=round(final_share, 6),
        fold_continuation_share=round(continuation_share, 6),
        n_policy_viol=n_infeas,
        max_overspend_rel=overspend,
        failure="",
    )
    for policy, stats in (("c", stats_c), ("n", stats_n)):
        *quantiles, nbad = stats
        for key, value in zip(("med", "p95", "max"), quantiles, strict=True):
            row[f"res_{policy}_{key}"] = round(value, 8)
        row[f"res_{policy}_nbad"] = nbad
    logger.info(
        "alpha=%.2f D=%.2f %-11s %6.2fs a_hat=%3s ipm_bad=%2d folds=%d/%d "
        "c[med/p95/max]=%.1e/%.1e/%.1e n=%.1e/%.1e/%.1e bad=%d/%d infeas=%d",
        a_exp,
        die_prob,
        name,
        secs,
        a_hat,
        ipm_bad,
        n_reached,
        n_folds,
        *stats_c[:3],
        *stats_n[:3],
        stats_c[3],
        stats_n[3],
        n_infeas,
    )
    return row


def load_rows_from_csv(csv_path):
    """Sweep rows, as dicts by column, read back from an already-written CSV
    with the numeric dtypes ``plot_sweep`` expects restored. A "failed" row
    keeps empty strings in its numeric columns, as ``run_sweep`` wrote them.
    """
    rows = read_rows(csv_path)
    for row in rows:
        for col, value in row.items():
            if col in _TEXT_COLS or value == "":
                continue
            row[col] = int(value) if col in _INT_COLS else float(value)
    return rows


def lighten_color(hex_color, amount=0.35):
    """Blend a color toward white to desaturate/lighten a comparator series.

    Hue is unchanged (so colorblind-safety is preserved); only lightness
    drops. The ENGINE series, plotted at full saturation, thus carries more
    visual weight than the curvilinear and Delaunay comparators.
    """
    r, g, b = to_rgb(hex_color)
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def swarm_offsets(values, spread=0.4):
    """Small deterministic x-offsets for points sharing a measured
    bracket-displacement constant.

    Several calibrations can share one measured ``alpha_hat``, which would
    otherwise stack into one indistinguishable marker; each such group is spaced evenly over
    ``[-spread, spread]`` around its shared value, isolated points are untouched.
    """
    offsets = np.zeros(len(values))
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            offsets[order[i:j]] = np.linspace(-spread, spread, j - i)
        i = j
    return offsets


def set_log_decades(ax, values, min_decades=2) -> None:
    """Force at least ``min_decades`` labeled decades on a log y-axis.

    Each panel's data spans only 1-1.7 decades, and matplotlib's default
    autoscale can collapse that to a single labeled decade tick. One tick
    gives the reader no sense of scale. Rounding out to whole decades and pinning
    an explicit ``LogLocator``/``LogFormatterSciNotation`` guarantees at
    least two readable ``10^n`` gridlines per panel.
    """
    lo_exp = np.floor(np.log10(np.min(values)))
    hi_exp = np.ceil(np.log10(np.max(values)))
    if hi_exp - lo_exp < min_decades:
        hi_exp = lo_exp + min_decades
    ax.set_ylim(10.0**lo_exp, 10.0**hi_exp)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())


def main() -> None:
    start_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip the (expensive) sweep; replot from the existing CSV only.",
    )
    args = parser.parse_args()
    if args.plot_only:
        rows = load_rows_from_csv(_OUT_CSV)
        logger.info(
            "plot-only: read %d rows from %s (not modified)",
            len(rows),
            _OUT_CSV,
        )
    else:
        rows = run_sweep()
    plot_sweep(rows)


def plot_interpolator_series(axes, panels, panel_values, colors, name, rows) -> None:
    """Scatter one interpolator's residuals across all three panels."""
    ok = [
        r
        for r in rows
        if r["interpolator"] == name and r["status"] == "ok" and r["ipo_alpha_hat"] >= 0
    ]
    if not ok:
        return
    a_hat = np.asarray([r["ipo_alpha_hat"] for r in ok], dtype=float)
    jitter = {"curvilinear": -0.12, "engine": 0.12, "delaunay": 0.0}[name]
    x = a_hat + jitter + swarm_offsets(a_hat)
    label = _LABELS[name]
    for ax, (col, _) in zip(axes, panels, strict=False):
        vals = np.asarray([r[col] for r in ok], dtype=float)
        panel_values[col].append(vals)
        ax.semilogy(
            x,
            vals,
            "o",
            label=label,
            color=colors[name],
            alpha=0.7,
            markersize=6,
        )


def plot_sweep(rows) -> None:
    """Median, p95, and max consumption FOC residual against the measured
    bracket-displacement constant of each run's own endogenous mesh.

    Rows whose mesh violates IPM (NaN nodes included) have no defined
    bracket-displacement constant (``ipo_alpha_hat`` is NaN) and are excluded
    from the panels; the CSV records them via ``ipm_violations`` and
    ``nan_nodes``. Solve times stay in the CSV only: at this grid size they
    are dominated by per-call Python overhead, and the appendix timing
    figure already measures cost properly at scale.
    """
    apply_engine_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    colors = {
        "curvilinear": lighten_color(ENGINE_COLORS["hark"]),
        "engine": ENGINE_COLORS["engine"],
        "delaunay": lighten_color(ENGINE_COLORS["neutral_dark"]),
    }
    panels = [
        ("res_c_med", "(a) Median"),
        ("res_c_p95", "(b) 95th Percentile"),
        ("res_c_max", "(c) Maximum"),
    ]
    panel_values = {col: [] for col, _ in panels}
    for name, _ in _INTERPOLATORS:
        plot_interpolator_series(axes, panels, panel_values, colors, name, rows)
    for ax, (col, title) in zip(axes, panels, strict=False):
        ax.set_xlabel(
            r"measured bracket-displacement constant $\hat{\alpha}$ of the mesh",
        )
        ax.set_title(title, fontsize=11)
        format_axes_for_publication(ax, legend=False)
        set_log_decades(ax, np.concatenate(panel_values[col]))
    axes[0].set_ylabel("consumption Euler residual")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    n_calibrations = len({(r["alpha"], r["die_prob_max"]) for r in rows})
    excluded = Counter(
        r["interpolator"]
        for r in rows
        if r["status"] == "ok" and not r["ipo_alpha_hat"] >= 0
    )
    axes[0].annotate(
        "".join(
            f"{_LABELS[name]}: {k} of {n_calibrations} calibrations excluded\n"
            for name, k in excluded.items()
        )
        + r"$\hat{\alpha}$ undefined (IPM violated)",
        xy=(0.97, 0.40),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7,
        color=ENGINE_COLORS["neutral_medium"],
    )
    plt.tight_layout()
    save_engine_figure(fig, "health_warp_sweep.png")


if __name__ == "__main__":
    main()
