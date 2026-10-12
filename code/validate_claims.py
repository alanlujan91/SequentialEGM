"""Validate Numerical Claims from Paper.

This script validates the numerical claims made in content/paper/ against the code.
Run with: uv run python validate_claims.py
"""

from __future__ import annotations

import logging
import sys
import traceback

import numpy as np
from cli import start_logging
from engine.bench_regrid_scaling import interior_axis, time_regrid, warped_regrid_case
from engine.engine2d import Engine2DInterp, warmup_jit
from engine.utilities import (
    corner_jacobians,
    empirical_ipo_alpha,
    ipm_violations,
    reached_folds,
)
from HARK.interpolation import Curvilinear2DInterp
from HARKmodels.ConsHealthSeparableModel import (
    PAPER_TIMING_CYCLES,
    PAPER_TIMING_GRID,
    SUMMARY_CYCLES,
    SUMMARY_GRIDS,
    BasicHealthConsumerType,
    SequentialHealthConsumerType,
    init_basic_health,
    paper_timing_params,
    post_decision_mesh,
)
from HARKmodels.ConsLaborSeparableModel import (
    LaborSeparableConsumerType,
    init_labor_separable,
)
from HARKmodels.health_interp import (
    HEALTH_CURVATURE_BOUND,
    HealthEngine2DInterp,
    continuation_queries,
)
from repo_paths import FIG_DIR, read_rows
from scipy.interpolate import LinearNDInterpolator
from timing import mean_call_time, timed

logger = logging.getLogger(__name__)

# Track validation results
RESULTS = {"passed": [], "failed": [], "warnings": [], "errors": []}


def recorder(bucket: str, level: int, tag: str):
    """A logger for one outcome: appends ``(claim, details)`` to ``RESULTS[bucket]``
    and logs ``TAG: claim (details)`` at ``level``.
    """

    def record(claim: str, details: str = "") -> None:
        RESULTS[bucket].append((claim, details))
        logger.log(level, "%s: %s%s", tag, claim, f" ({details})" if details else "")

    return record


log_pass = recorder("passed", logging.INFO, "PASS")
# A failure is a claim the data disproves; an error is an environment or
# infrastructure problem (inconclusive, never a claim failure).
log_fail = recorder("failed", logging.ERROR, "FAIL")
log_warn = recorder("warnings", logging.WARNING, "WARN")
log_error = recorder("errors", logging.ERROR, "ERROR")


def log_tiered(
    value: float,
    pass_at: float,
    warn_at: float,
    passed,
    warned,
    failed,
) -> None:
    """Log the ``(claim, details)`` pair ``passed`` when ``value >= pass_at``,
    ``warned`` when ``value >= warn_at``, else ``failed``.
    """
    if value >= pass_at:
        log_pass(*passed)
    elif value >= warn_at:
        log_warn(*warned)
    else:
        log_fail(*failed)


def check_params(checks, source: str = "the agent", tol: float = 1e-6) -> None:
    """Pass or fail each ``(name, actual, expected)``; an ``actual`` missing from
    ``source`` warns.
    """
    for name, actual, expected in checks:
        if actual is None:
            log_warn(f"{name} = {expected}", f"parameter not found in {source}")
        elif abs(float(actual) - float(expected)) < tol:
            log_pass(f"{name} = {expected}", f"actual = {actual}")
        else:
            log_fail(f"{name} = {expected}", f"actual = {actual}")


def get_scalar(val):
    """Extract scalar from a value that may be a list or array.

    Returns None if val is an empty list or array.
    """
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return None
        return val[0]
    return val


# =============================================================================
# Section 2: Consumption-Labor Model Parameters
# =============================================================================
def validate_consumption_labor_params() -> None:
    """Validate consumption-labor model parameters match paper claims.

    Checks the Section 2 figure model (``LaborSeparableConsumerType``) against the
    footnote ``[^consumption-labor-params]``. The portfolio choice was removed in
    the revision, so there are no risky-return parameters to validate.

    The paper's plain leisure-utility coefficient labShare=16 is the code's
    ``LeisureFactor ** (1 - CRRA)`` with ``LeisureFactor=0.5``: 0.5^(1-5) = 16.
    """
    agent = LaborSeparableConsumerType()

    checks = [
        ("CRRA", get_scalar(agent.CRRA), 5.0),
        ("DiscFac", get_scalar(agent.DiscFac), 0.9),
        ("Rfree", get_scalar(agent.Rfree), 1.03),
        ("TranShkStd (wage shock std)", get_scalar(agent.TranShkStd), 0.1),
        (
            "LeisureCRRA (leiShare)",
            get_scalar(init_labor_separable.get("LeisureCRRA", agent.LeisureCRRA)),
            2.0,
        ),
        (
            "labShare (= LeisureFactor^(1-CRRA) in the code)",
            get_scalar(init_labor_separable.get("LeisureFactor", agent.LeisureFactor))
            ** (1.0 - get_scalar(agent.CRRA)),
            16.0,
        ),
    ]

    check_params(checks)


# =============================================================================
# Section 3: Health Model Parameters
# =============================================================================
def validate_health_model_params() -> None:
    """Validate health model parameters match paper claims (footnote [^health-grid-params])."""
    checks = [
        ("CRRA", get_scalar(init_basic_health.get("CRRA")), 0.5),
        ("DiscFac", get_scalar(init_basic_health.get("DiscFac")), 0.95),
        ("Rfree", get_scalar(init_basic_health.get("Rfree")), 1.03),
        (
            "HealthProdExp (α)",
            get_scalar(init_basic_health.get("HealthProdExp")),
            0.35,
        ),
        (
            "DeprRteMean (depreciation center)",
            get_scalar(init_basic_health.get("DeprRteMean")),
            0.05,
        ),
        (
            "DeprRteSpread (depreciation half-width)",
            get_scalar(init_basic_health.get("DeprRteSpread")),
            0.05,
        ),
        (
            "WageRteMean (wage mean)",
            get_scalar(init_basic_health.get("WageRteMean")),
            0.1,
        ),
        (
            "WageRteStd (wage std)",
            get_scalar(init_basic_health.get("WageRteStd")),
            0.1,
        ),
    ]

    check_params(checks, "init_basic_health")


# =============================================================================
# Section 4: ENGINE Speedup Claims
# =============================================================================
def build_warped_test_grid():
    """Build the warped (J, K) test grid shared by the speedup and accuracy checks.

    Each coordinate shifts by a function of the other axis alone, so the mesh
    is fold-free and IPM and every cell is a parallelogram up to rounding.
    The queries fill the interior of the mesh's bounding box.
    """
    J, K = 44, 50
    x_base = np.geomspace(0.1, 100, J)
    y_base = np.linspace(0.1, 100, K)
    xx, yy = np.meshgrid(x_base, y_base, indexing="ij")

    warp = 0.1
    x_endog = xx + warp * np.sin(np.pi * yy / yy.max()) * np.mean(x_base)
    y_endog = yy + warp * np.cos(np.pi * xx / xx.max()) * np.mean(y_base)
    values = np.log(x_endog + y_endog + 1.0)

    n_query = 40
    xx_q, yy_q = np.meshgrid(
        interior_axis(x_endog.min(), x_endog.max(), n_query),
        interior_axis(y_endog.min(), y_endog.max(), n_query),
        indexing="ij",
    )
    return x_endog, y_endog, values, xx_q, yy_q


def time_interp_query(interp, xx_q, yy_q, n_warmup=5, n_timed=20):
    """Warm up ``interp``, then time its mean per-call query cost."""

    def query():
        return interp(xx_q, yy_q)

    for _ in range(n_warmup):
        query()
    elapsed = mean_call_time(query, n_timed)
    result = interp(xx_q, yy_q)[0]
    return elapsed, result


def benchmark_delaunay(x_endog, y_endog, values, xx_q, yy_q):
    """Time Delaunay triangulation setup plus query cost, separately and total."""
    points = np.column_stack([x_endog.ravel(), y_endog.ravel()])

    delaunay_interp, setup = timed(LinearNDInterpolator, points, values.ravel())

    query_points = np.column_stack([xx_q.ravel(), yy_q.ravel()])
    _ = delaunay_interp(query_points)  # warmup
    query = mean_call_time(lambda: delaunay_interp(query_points), 10)
    return setup, query, setup + query


def report_interpolation_accuracy(engine_result, hark_result, truth) -> int:
    """Log the ENGINE-vs-analytic-truth and HARK-outlier claims.

    ENGINE correctness is checked against the known analytic function the
    grid samples, f(x, y) = log(x + y + 1): the real ground truth, where a
    stale JIT cache or wrong-branch execution would blow up the error.

    We do NOT compare against HARK as a reference: on this valid mesh HARK's
    Curvilinear2DInterp overflows at some queries. Its per-cell inversion
    solves a quadratic whose leading coefficient scales with the cell's twist
    (x_00 - x_10 - x_01 + x_11, likewise y). On these parallelogram cells
    the twist is zero or rounding noise (about 1e-16); at the noise cells the
    root it returns is off by many orders of magnitude. ENGINE stays finite
    and accurate, itself a robustness advantage. The worst HARK deviation
    is computed and reported below rather than hard-coded, since its
    magnitude depends on the grid and the installed HARK version.

    Returns the count of HARK outliers (|err| > 1), for the caller's
    staleness check.
    """
    engine_dev = np.abs(engine_result - truth)
    engine_finite = bool(np.all(np.isfinite(engine_dev)))
    engine_mean_err = float(np.mean(engine_dev))
    engine_max_err = float(np.max(engine_dev))
    hark_dev = np.abs(hark_result - truth)
    hark_bad = int(np.sum(~np.isfinite(hark_dev) | (hark_dev > 1.0)))
    hark_finite_dev = hark_dev[np.isfinite(hark_dev)]
    hark_worst = float(hark_finite_dev.max()) if hark_finite_dev.size else float("inf")
    if engine_finite and engine_mean_err <= 1e-2:
        log_pass(
            "ENGINE interpolation accuracy (vs analytic truth)",
            f"mean|err|={engine_mean_err:.2e}, max|err|={engine_max_err:.2e}; "
            f"outliers (|err| > 1): HARK Curvilinear2DInterp {hark_bad} "
            f"(worst |err|={hark_worst:.1e}), ENGINE {int(np.sum(engine_dev > 1.0))}",
        )
    else:
        log_fail(
            "ENGINE interpolation accuracy (vs analytic truth)",
            f"mean|err|={engine_mean_err:.2e} (threshold 1e-2), finite={engine_finite}",
        )
    return hark_bad


def report_speed_vs_hark(speedup_vs_hark: float) -> None:
    """Log the "comparable to HARK" speed claim (within 50%, per the paper)."""
    log_tiered(
        speedup_vs_hark,
        0.8,
        0.5,
        (
            "ENGINE comparable to HARK interpolation speed",
            f"actual speedup = {speedup_vs_hark:.1f}x",
        ),
        (
            "ENGINE vs HARK interpolation",
            f"ratio = {speedup_vs_hark:.2f}x (ENGINE moderately slower)",
        ),
        (
            "ENGINE vs HARK interpolation",
            f"ratio = {speedup_vs_hark:.2f}x (ENGINE significantly slower)",
        ),
    )


def report_speed_vs_delaunay(
    speedup_vs_delaunay_total: float,
    speedup_vs_delaunay_query: float,
) -> None:
    """Log the Delaunay total-cost claim and the informational per-query ratio.

    Per-query speed vs Delaunay is informational, not a paper claim. The
    paper's advantage over Delaunay is total cost: Delaunay's triangulation
    is O(N log N) to build and must be rebuilt whenever the grid changes
    (every backward step in EGM), whereas ENGINE needs no setup
    (content/paper/4_multinterp.md:14,197). Once the triangulation is
    prebuilt, scipy's C barycentric per-query lookup is competitive with
    (often faster than) ENGINE's per-query, so only the ratio is reported
    here; the EGM-relevant win is the total-cost check.
    """
    if speedup_vs_delaunay_total >= 10:
        log_pass(
            "ENGINE faster than Delaunay (with setup)",
            f"actual speedup = {speedup_vs_delaunay_total:.1f}x",
        )
    else:
        log_warn(
            "ENGINE vs Delaunay (total)",
            f"actual speedup = {speedup_vs_delaunay_total:.1f}x",
        )

    query_direction = (
        "ENGINE slower" if speedup_vs_delaunay_query < 1.0 else "ENGINE faster"
    )
    log_pass(
        "ENGINE vs Delaunay (per-query, informational)",
        f"per-query ratio = {speedup_vs_delaunay_query:.2f}x ({query_direction} "
        f"per-query); the paper's claim is total cost (no triangulation rebuild), "
        f"validated above",
    )


def validate_speedup_claims() -> None:
    """Validate ENGINE speedup claims against HARK and Delaunay.

    NOTE: wall-time speedup measurements are sensitive to machine load.
    Run this validator on a quiescent machine for reproducible results.
    The warmup calls in ``main`` and ``time_interp_query`` reduce JIT variance
    but do not eliminate OS noise.
    """
    x_endog, y_endog, values, xx_q, yy_q = build_warped_test_grid()

    engine_interp = Engine2DInterp([values], x_endog, y_endog)
    engine_time, engine_result = time_interp_query(engine_interp, xx_q, yy_q)

    hark_interp = Curvilinear2DInterp([values], x_endog, y_endog)
    hark_time, hark_result = time_interp_query(hark_interp, xx_q, yy_q)

    _delaunay_setup, delaunay_query, delaunay_total = benchmark_delaunay(
        x_endog,
        y_endog,
        values,
        xx_q,
        yy_q,
    )

    speedup_vs_hark = hark_time / engine_time
    speedup_vs_delaunay_total = delaunay_total / engine_time
    speedup_vs_delaunay_query = delaunay_query / engine_time

    truth = np.log(xx_q + yy_q + 1.0)
    hark_bad = report_interpolation_accuracy(engine_result, hark_result, truth)

    # The robustness narrative (ENGINE finite where HARK is not) depends on
    # HARK producing outliers on this grid. Warn if an upstream HARK fix
    # removes them; the claim then needs re-evaluating.
    if hark_bad == 0:
        log_warn(
            "HARK produced no outliers on this grid",
            "ENGINE-vs-HARK robustness narrative may need updating",
        )

    # Paper claims: "comparable speed to optimized sector-walking implementations"
    report_speed_vs_hark(speedup_vs_hark)
    report_speed_vs_delaunay(speedup_vs_delaunay_total, speedup_vs_delaunay_query)


def mean_build_and_solve(AgentClass, interpolator, params, reps=3):
    """Mean seconds to construct and solve one agent, after one warmup solve,
    and the warmup agent.
    """

    def build_and_solve():
        agent = AgentClass(interpolator=interpolator, **params)
        agent.solve()
        return agent

    agent = build_and_solve()
    return mean_call_time(build_and_solve, reps), agent


def validate_full_model_speedup():
    """Validate full model solve speedup (EGM^n + ENGINE vs HARK baseline).

    The paper's fig:performance claims a 2.8-4.7x speedup on the health model.
    Pass requires the published lower bound (2.8x); anything below warns rather
    than passes, since a measurement contradicting the published range must
    never certify it, while wall-time sensitivity to machine load argues
    against a hard failure.

    Returns the solved ENGINE agent, whose health-investment meshes the
    Section 5 checks test.
    """
    params = paper_timing_params()

    hark_time, _ = mean_build_and_solve(
        BasicHealthConsumerType,
        Curvilinear2DInterp,
        params,
    )
    engine_time, engine_agent = mean_build_and_solve(
        SequentialHealthConsumerType,
        HealthEngine2DInterp,
        params,
    )
    speedup = hark_time / engine_time

    # Paper (fig:performance) claims a 2.8-4.7x speedup over the baseline.
    detail = f"actual = {speedup:.2f}x (paper claims 2.8-4.7x)"
    log_tiered(
        speedup,
        2.8,
        1.5,
        ("Full model solve several times faster than HARK baseline", detail),
        ("Full model solve speedup below claimed range", detail),
        ("Full model solve speedup", detail),
    )
    return engine_agent


# =============================================================================
# Section 4: Interpolation Accuracy (O(h²) convergence)
# =============================================================================
def validate_interpolation_convergence() -> None:
    """Validate O(h²) error convergence for ENGINE."""

    def true_func(x, y):
        return np.sin(x) * np.cos(y) + x * y

    # Test with increasing grid resolutions
    grid_sizes = [10, 20, 40, 80]
    errors = []
    spacings = []

    # Fixed query grid (high resolution)
    x_query = np.linspace(0.1, 2.9, 100)
    y_query = np.linspace(0.1, 2.9, 100)
    xx_q, yy_q = np.meshgrid(x_query, y_query, indexing="ij")
    true_vals = true_func(xx_q, yy_q)

    for N in grid_sizes:
        # Create regular grid
        x_grid = np.linspace(0, 3, N)
        y_grid = np.linspace(0, 3, N)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
        values = true_func(xx, yy)

        # Interpolate
        interp = Engine2DInterp([values], xx, yy)
        result = interp(xx_q, yy_q)[0]

        # Calculate error
        max_error = np.max(np.abs(result - true_vals))
        h = 3.0 / (N - 1)

        errors.append(max_error)
        spacings.append(h)

    # Check convergence order
    errors = np.array(errors)
    spacings = np.array(spacings)

    # Fit log(error) vs log(h) to get convergence order
    coeffs = np.polyfit(np.log(spacings), np.log(errors), 1)
    order = coeffs[0]

    for i, N in enumerate(grid_sizes):
        logger.info("  N=%3d: h=%.4f, error=%.2e", N, spacings[i], errors[i])
    logger.info("Fitted order: %.2f", order)

    # Paper claims O(h²), so order should be ~2
    if 1.5 <= order <= 2.5:
        log_pass("O(h²) convergence", f"fitted order = {order:.2f}")
    else:
        log_fail("O(h²) convergence", f"fitted order = {order:.2f}, expected ~2.0")


# =============================================================================
# Section 4: Complexity Scaling
# =============================================================================
def fit_r_squared(x_data, y_data):
    """Compute R² for a linear fit of y vs x; NaN for constant y, where R² is
    undefined (so no model can claim the fit).
    """
    coeffs = np.polyfit(x_data, y_data, 1)
    y_pred = np.polyval(coeffs, x_data)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def validate_complexity_scaling() -> None:
    """Validate sublinear complexity scaling (log K fits vs linear K)."""
    # Test with varying grid sizes: wide range so scaling dominates overhead.
    # Use scalar queries to isolate per-query complexity from batch amortization.
    test_configs = [
        (50, 60),
        (100, 120),
        (200, 250),
        (400, 500),
        (800, 1000),
    ]

    times = []
    K_values = []

    for J, K in test_configs:
        # Create warped grid (not rectilinear, so ENGINE binary search is exercised)
        x_base = np.linspace(0, 10, J)
        y_base = np.linspace(0, 10, K)
        xx, yy = np.meshgrid(x_base, y_base, indexing="ij")
        xx = xx + 0.05 * np.sin(yy)
        yy = yy + 0.05 * np.cos(xx)
        values = np.log(xx + yy + 1)

        interp = Engine2DInterp([values], xx, yy)

        # Scalar queries spread across the domain
        n_scalar = 200
        x_pts = np.linspace(1.0, 9.0, n_scalar)
        y_pts = np.linspace(1.0, 9.0, n_scalar)

        def scalar_queries(interp=interp, x_pts=x_pts, y_pts=y_pts) -> None:
            for xq, yq in zip(x_pts, y_pts, strict=True):
                interp(xq, yq)

        # Warmup
        for qi in range(min(5, n_scalar)):
            interp(x_pts[qi], y_pts[qi])

        times.append(mean_call_time(scalar_queries) / n_scalar)
        K_values.append(K)

    times = np.array(times)
    K_values = np.array(K_values, dtype=float)

    # Fit three complexity models: O(log K), O(log^2 K), and O(K)
    log_K = np.log2(K_values)
    r2_log = fit_r_squared(log_K, times)  # O(log K) model
    r2_log2 = fit_r_squared(log_K**2, times)  # O(log^2 K) model
    r2_linear = fit_r_squared(K_values, times)  # O(K) model

    # Determine best fit
    models = [("O(log K)", r2_log), ("O(log² K)", r2_log2), ("O(K)", r2_linear)]
    best_model, best_r2 = max(models, key=lambda x: x[1])

    logger.info("%12s %16s", "Grid", "Time (us/query)")
    for i, (J, K) in enumerate(test_configs):
        logger.info("  %dx%4d    %14.1f", J, K, times[i] * 1e6)

    # Pass if any sublinear model fits well, or if time growth is clearly
    # sublinear (less than 2x time increase for 16x grid increase).
    time_ratio = times[-1] / times[0]
    grid_ratio = K_values[-1] / K_values[0]

    if r2_log > 0.8 or r2_log2 > 0.8:
        best_sub = "O(log K)" if r2_log > r2_log2 else "O(log² K)"
        best_sub_r2 = max(r2_log, r2_log2)
        log_pass(
            f"Sublinear complexity ({best_sub})",
            f"R² = {best_sub_r2:.4f}",
        )
    elif time_ratio < grid_ratio * 0.5:
        log_pass(
            "Sublinear complexity (empirical)",
            f"time grew {time_ratio:.1f}x for {grid_ratio:.0f}x grid increase",
        )
    else:
        log_fail(
            "Complexity scaling (O(log K) is a central paper claim)",
            f"Best fit: {best_model} (R^2 = {best_r2:.4f}), "
            f"time grew {time_ratio:.1f}x for {grid_ratio:.0f}x grid",
        )


def validate_regrid_linear_scaling() -> None:
    """Validate Section 4's measured regrid scaling.

    The dense regrid's worst case is O(P Q log J log K), but with sorted target
    axes each bracket search resumes where the previous one ended. Section 4
    reports the resulting N^0.98 scaling in N = J K at matched refinement
    (P = J, Q = K), measured by ``code/engine/bench_regrid_scaling.py``. A log-log slope alone
    cannot reject an O(N log N) law over a feasible size range (such a law has
    effective slope about 1.09 here, inside any noise-tolerant slope band), so
    the discriminating statistic is the per-output time ratio between the
    largest and smallest mesh: about 1 under linear scaling, but
    log(N_max)/log(N_min) = 1.67 under N log N over this range. Pass requires
    that ratio at most 1.3 and the slope inside a gross-sanity band
    [0.8, 1.15]. Wall-time scaling depends on machine load; run on a quiescent
    machine for reproducible results.
    """
    sizes = [64, 128, 256, 512, 1024]
    times = []
    for s in sizes:
        _, interp, x_new, y_new = warped_regrid_case(s)
        times.append(
            time_regrid(interp, x_new, y_new, "serial", reps=5, min_sample_s=1e-3),
        )

    n_vals = np.array([s * s for s in sizes], dtype=float)
    times_arr = np.array(times)
    slope = np.polyfit(np.log(n_vals), np.log(times_arr), 1)[0]
    per_output = times_arr / n_vals
    per_output_ratio = per_output[-1] / per_output[0]

    for s, t in zip(sizes, times, strict=True):
        logger.info("  %dx%4d    %16.3f %12.1f", s, s, t * 1e3, 1e9 * t / (s * s))
    logger.info("Log-log slope of time vs N: %.3f", slope)
    logger.info("Per-output time ratio (largest/smallest): %.2f", per_output_ratio)

    if per_output_ratio <= 1.3 and 0.8 <= slope <= 1.15:
        log_pass(
            "Linear sorted-regrid scaling",
            f"per-output ratio = {per_output_ratio:.2f} (<= 1.3; N log N "
            f"predicts 1.67), slope = {slope:.3f}",
        )
    else:
        log_fail(
            "Linear sorted-regrid scaling (Section 4 reports N^0.98 at matched "
            "refinement)",
            f"per-output ratio = {per_output_ratio:.2f} (threshold 1.3), "
            f"slope = {slope:.3f} (band [0.8, 1.15])",
        )


# =============================================================================
# Section 5: Health-mesh diagnostics (the mesh comes from the full-model solve)
# =============================================================================
def validate_ipo_constant(x_endog) -> None:
    """Check that the health search margin covers the mesh's measured
    bracket-displacement constant.

    Section 4 says margins matched to the health meshes' measured constants
    (10 to 16) avoid the fallback searches. The health generators run ENGINE
    at ``HEALTH_CURVATURE_BOUND``, so a measured constant above it means
    fallback searches: extra search time, never a wrong bracket.
    """
    alpha = empirical_ipo_alpha(x_endog)
    if alpha <= HEALTH_CURVATURE_BOUND:
        log_pass(
            f"bracket-displacement constant α <= health search margin "
            f"{HEALTH_CURVATURE_BOUND}",
            f"measured α = {alpha}",
        )
    else:
        log_fail(
            f"bracket-displacement constant exceeds the health search margin "
            f"{HEALTH_CURVATURE_BOUND}",
            f"measured α = {alpha}; windowed searches miss and fall back to a "
            f"full row search, contradicting the matched-margin claim",
        )


def health_meshes(agent):
    """The ``(m, h)`` mesh of every period's health-investment interpolant."""
    return [
        (sol.health_investment.interp.fx, sol.health_investment.interp.fy)
        for sol in agent.solution
    ]


def log_unreached_folds(meshes, folds, queries) -> None:
    """Log, as information, the folded cells no query reaches: their count,
    columns and ``h`` range against the queries' ``h`` range.
    """
    unreached = [
        (y, folded[~reached])
        for (_, y), (folded, reached) in zip(meshes, folds, strict=True)
    ]
    h_corners = np.concatenate(
        [
            y[cells[:, 0] + dj, cells[:, 1] + dk]
            for y, cells in unreached
            for dj in (0, 1)
            for dk in (0, 1)
        ],
    )
    if not h_corners.size:
        return
    columns = sorted({int(k) for _, cells in unreached for k in cells[:, 1]})
    logger.info(
        "  folds outside the reached cells: %d cells over %d meshes, columns "
        "k = %s, h in [%.3g, %.3g]; queries span h in [%.3g, %.3g]",
        sum(len(cells) for _, cells in unreached),
        len(meshes),
        columns,
        h_corners.min(),
        h_corners.max(),
        queries[:, 1].min(),
        queries[:, 1].max(),
    )


def validate_grid_regularity(meshes, queries, kink_cells) -> None:
    """Validate ENGINE's two mesh requirements on the health meshes.

    IPM (``def-ipm``) must hold on every mesh. Fold-freeness
    (``def-fold-free``, all four corners) is required on the first
    ``kink_cells`` rows of cells, those beside the constructed boundary,
    constrained and ``a = 0`` rows (the policy kink lies on the ``a = 0``
    mesh line), and on every cell whose bilinear image contains a query
    point. The validator logs folds elsewhere as information.
    """
    folds = [reached_folds(x, y, queries) for x, y in meshes]
    n_reached = sum(int(reached.sum()) for _, reached in folds)
    n_kink = sum(
        int(np.sum(~(corner_jacobians(x, y)[:, :kink_cells] > 0).all(axis=0)))
        for x, y in meshes
    )
    n_ipm = sum(sum(ipm_violations(x, y)) for x, y in meshes)
    if n_reached == n_kink == n_ipm == 0:
        log_pass(
            "Grid regularity (IPM; four-corner fold-free beside the kink and "
            "on every cell the queries reach)",
            f"{len(meshes)} meshes, cell rows j < {kink_cells} beside the kink, "
            f"{len(queries)} query points",
        )
    else:
        log_fail(
            "Grid regularity",
            f"{n_kink} cells beside the kink fold; {n_reached} reached cells "
            f"fold; {n_ipm} steps break IPM",
        )
    log_unreached_folds(meshes, folds, queries)


def kink_cell_rows(agent) -> int:
    """Rows of cells beside ``endogenous_mesh``'s constructed rows: with the
    liquidity constraint, the cells between the ``m = 0`` boundary row, the
    ``constrained_N`` constrained rows and the ``a = 0`` row on which the
    kink lies, and the cells just above that row. Without the constraint,
    only the boundary cells.
    """
    constrained, _, _ = post_decision_mesh(
        agent.aLvlGrid,
        agent.HLvlGrid,
        agent.ShockDstn[0],
    )
    return agent.constrained_N + 2 if constrained else 1


def validate_health_meshes(agent) -> None:
    """Bracket-displacement and grid-regularity checks on the solved ENGINE
    agent's meshes, against the points its backward step queries.
    """
    meshes = health_meshes(agent)
    validate_ipo_constant(meshes[0][0])
    validate_grid_regularity(
        meshes,
        continuation_queries(agent),
        kink_cell_rows(agent),
    )


# Points just past the health summary's sweeps (OA-A): denser health grids at
# the paper's cycle count, and 20 cycles at the paper's grid.
SUMMARY_IPM_BREAKERS = (
    *(((a, h), PAPER_TIMING_CYCLES) for a, h in ((44, 60), (60, 75), (80, 100))),
    (PAPER_TIMING_GRID, 20),
)


def engine_ipm_break(grid, cycles) -> int:
    """``def-ipm`` violating steps of the first endogenous mesh ENGINE rejects
    when the simultaneous solver runs under it at ``grid`` and ``cycles``, or 0
    when every period's mesh keeps the precondition. The meshes depend on the
    interpolant through the continuation, so ENGINE's own solve is the measure.
    """
    breaks = []

    def engine(values, x, y):
        n_bad = sum(ipm_violations(x, y))
        if n_bad:
            breaks.append(n_bad)
        return HealthEngine2DInterp(values, x, y)

    params = {
        **init_basic_health,
        "aXtraCount": grid[0],
        "hLvlCount": grid[1],
        "cycles": cycles,
        "interpolator": engine,
    }
    try:
        BasicHealthConsumerType(**params).solve()
    except ValueError:
        if not breaks:
            raise
    return breaks[0] if breaks else 0


def validate_summary_sweep_ipm() -> None:
    """OA-A: every point of the health summary's sweeps keeps the endogenous mesh
    inside ENGINE's ``def-ipm`` precondition in every period, and the denser
    health grids and the longer horizon just past them break it.
    """
    sweep = ipm_census(summary_sweep_points())
    past = ipm_census(SUMMARY_IPM_BREAKERS)
    claim = "Health summary sweep stays inside def-ipm, denser grids break it"
    details = f"sweep points {sweep}; past-sweep points {past}"
    if any(sweep.values()) or not all(past.values()):
        log_fail(claim, details)
    else:
        log_pass(claim, details)


def summary_sweep_points():
    """The distinct ``(grid, cycles)`` points of the health summary's two sweeps."""
    cycle_sweep = [(PAPER_TIMING_GRID, c) for c in SUMMARY_CYCLES]
    grid_sweep = [(g, PAPER_TIMING_CYCLES) for g in SUMMARY_GRIDS]
    return list(dict.fromkeys(cycle_sweep + grid_sweep))


def ipm_census(points):
    """``{"JxK@cycles": def-ipm violations over all periods}`` for ``points``."""
    return {f"{g[0]}x{g[1]}@{c}": engine_ipm_break(g, c) for g, c in points}


# =============================================================================
# Artifact Consistency (see validate_artifact_consistency's docstring below)
# =============================================================================
def load_csv_rows(name: str) -> list[dict[str, str]] | None:
    """Read the CSV artifact ``FIG_DIR / name`` as string-valued dict rows, or
    None (logged as a failure) if it is missing.
    """
    csv_path = FIG_DIR / name
    if not csv_path.is_file():
        log_fail(f"Artifact consistency: {name}", f"missing artifact at {csv_path}")
        return None
    return read_rows(csv_path)


def report_artifact(name: str, mismatches: list[str], pass_detail: str) -> None:
    """Fail artifact ``name`` with its mismatches, or pass it with ``pass_detail``."""
    claim = f"Artifact consistency: {name}"
    if mismatches:
        log_fail(claim, "; ".join(mismatches))
    else:
        log_pass(claim, pass_detail)


def check_negm_ratio_targets(seconds, targets, threads, mismatches) -> None:
    """Append a mismatch for each (nB, nA) whose NEGM/EGM^n ratio misses target."""
    for (nB, nA), target in targets.items():
        negm = seconds.get((nB, nA, "negm", threads))
        egmn = seconds.get((nB, nA, "egmn", threads))
        if negm is None or egmn is None:
            mismatches.append(f"({nB},{nA}) {threads}-thread rows missing")
            continue
        ratio = round(negm / egmn, 1)
        if ratio != target:
            mismatches.append(f"({nB},{nA}) {threads}-thread ratio {ratio} != {target}")


def check_timing_negm_csv() -> None:
    """Check timing_negm.csv against the paper's quoted NEGM/EGM^n ratios.

    content/paper/3_multdim_abrs.md quotes serial factors of about thirteen at
    all three grids (12.5, 13.1, 13.1) and 16-thread ratios 3.7 to 7.9;
    6_conclusion.md and the abstract quote the same span.
    """
    rows = load_csv_rows("timing_negm.csv")
    if rows is None:
        return

    seconds = {}
    for row in rows:
        key = (int(row["nB"]), int(row["nA"]), row["solver"], int(row["threads"]))
        seconds[key] = float(row["seconds"])

    expected_serial = {(40, 60): 12.5, (80, 120): 13.1, (160, 240): 13.1}
    expected_16t = {(40, 60): 3.7, (80, 120): 6.1, (160, 240): 7.9}

    mismatches = []
    check_negm_ratio_targets(seconds, expected_serial, 1, mismatches)
    check_negm_ratio_targets(seconds, expected_16t, 16, mismatches)
    report_artifact(
        "timing_negm.csv",
        mismatches,
        "serial ratios 12.5/13.1/13.1 and 16-thread ratios 3.7/6.1/7.9 match the CSV",
    )


def check_percent(mismatches, name, fraction, digits, target) -> None:
    """Append a mismatch unless ``100 * fraction`` rounds to ``target`` at ``digits``."""
    pct = float(f"{100 * float(fraction):.{digits}f}")
    if pct != target:
        mismatches.append(f"{name} {pct} != {target}")


def check_refinement_rows(by_label, expected_refinement, mismatches) -> None:
    """Append a mismatch for each refinement label whose rel_median misses target."""
    for label, target in expected_refinement.items():
        row = by_label.get(label)
        if row is None:
            mismatches.append(f"refinement row '{label}' missing")
            continue
        check_percent(
            mismatches,
            f"refinement '{label}' rel_median",
            row["rel_median"],
            1,
            target,
        )


def check_bounds_row(bounds_rows, expected_bounds, mismatches) -> None:
    """Append a mismatch for the single bounds row's fields that miss target."""
    if not bounds_rows:
        mismatches.append("bounds row missing")
        return
    row = bounds_rows[0]
    for field, target in expected_bounds.items():
        check_percent(mismatches, f"bounds {field}", row[field], 2, target)


def check_refinement_bounds_csv() -> None:
    """Check refinement_bounds_abrs.csv against the paper's quoted percentages.

    content/paper/8_online_appendix.md quotes refinement medians 9.3/3.7/0.9
    percent (1 display decimal); the bounds row (@tab-refinement-bounds-abrs)
    quotes 0.05/0.39/4.27 percent (2 display decimals).
    """
    rows = load_csv_rows("refinement_bounds_abrs.csv")
    if rows is None:
        return

    by_label = {row["label"]: row for row in rows if row["exercise"] == "refinement"}
    bounds_rows = [row for row in rows if row["exercise"] == "bounds"]

    mismatches = []
    check_refinement_rows(
        by_label,
        {"coarse": 9.3, "medium": 3.7, "fine": 0.9},
        mismatches,
    )
    check_bounds_row(
        bounds_rows,
        {"rel_median": 0.05, "rel_p95": 0.39, "rel_max": 4.27},
        mismatches,
    )
    report_artifact(
        "refinement_bounds_abrs.csv",
        mismatches,
        "refinement 9.3/3.7/0.9 percent and bounds 0.05/0.39/4.27 percent "
        "match the CSV",
    )


def check_extrap_frequency_csv() -> None:
    """Check extrap_frequency_abrs.csv against the paper's quoted pooled fractions.

    content/paper/8_online_appendix.md quotes pooled outside fractions 1/1440,
    1/5760, 1/23040, displayed as 0.069, 0.017, 0.004 percent
    (@tab-extrap-frequency-abrs).
    """
    rows = load_csv_rows("extrap_frequency_abrs.csv")
    if rows is None:
        return

    pooled = {row["resolution"]: row for row in rows if row["income_state"] == "all"}

    expected = {
        "coarse": (1, 1440, 0.069),
        "medium": (1, 5760, 0.017),
        "fine": (1, 23040, 0.004),
    }

    mismatches = []
    for resolution, (n_outside, n_queries, target_pct) in expected.items():
        row = pooled.get(resolution)
        if row is None:
            mismatches.append(f"resolution '{resolution}' pooled row missing")
            continue
        actual_outside = int(row["n_outside"])
        actual_queries = int(row["n_queries"])
        if (actual_outside, actual_queries) != (n_outside, n_queries):
            mismatches.append(
                f"'{resolution}' counts {actual_outside}/{actual_queries} != "
                f"{n_outside}/{n_queries}",
            )
        check_percent(
            mismatches,
            f"'{resolution}' outside_fraction",
            row["outside_fraction"],
            3,
            target_pct,
        )
    report_artifact(
        "extrap_frequency_abrs.csv",
        mismatches,
        "pooled outside fractions 1/1440, 1/5760, 1/23040 (0.069/0.017/0.004 "
        "percent) match the CSV",
    )


def collect_solvetime_ratios(seconds, resolutions, mismatches) -> list[float]:
    """Return the ENGINE-vs-curvilinear ratio per resolution, flagging outliers."""
    ratios = []
    for res in resolutions:
        eng = seconds.get((res, "engine"))
        cur = seconds.get((res, "curvilinear"))
        if eng is None or cur is None:
            mismatches.append(f"{res} converged rows missing")
            continue
        ratio = cur / eng
        ratios.append(ratio)
        if not 2.5 <= ratio < 5.0:
            mismatches.append(
                f"{res} ratio {ratio:.2f} outside three-to-four-and-a-half"
            )
    return ratios


def check_solvetime_ratio_csv() -> None:
    """Check solvetime_abrs.csv against the paper's ENGINE-vs-curvilinear claim.

    The paper quotes the full-solve ratio as "three to four and a half" times,
    widening with the grid: each ratio must lie in [2.5, 5.0) and rise with resolution,
    the precision every run supports (run-to-run endpoint noise is about 0.3).
    """
    rows = load_csv_rows("solvetime_abrs.csv")
    if rows is None:
        return

    seconds = {}
    for row in rows:
        if row["status"].startswith("ok"):
            seconds[row["resolution"], row["method"]] = float(row["solve_seconds"])

    mismatches = []
    ratios = collect_solvetime_ratios(seconds, ("coarse", "medium", "fine"), mismatches)
    if ratios != sorted(ratios):
        mismatches.append(
            f"ratios {[f'{r:.2f}' for r in ratios]} do not widen with the grid",
        )

    report_artifact(
        "solvetime_abrs.csv",
        mismatches,
        "ENGINE-vs-curvilinear ratios "
        + "/".join(f"{r:.2f}" for r in ratios)
        + " three to four and a half, widening with the grid",
    )


def check_realign_share_csv() -> None:
    """Section 4.4: the Stage-0 realign calls account for more than four fifths
    of ENGINE's per-iteration advantage over the curvilinear method at every
    grid, the per-call gap (regrid_percall_abrs.csv, one realign call per
    iteration) over the per-iteration gap (solvetime_abrs.csv).
    """
    solve = load_csv_rows("solvetime_abrs.csv")
    percall = load_csv_rows("regrid_percall_abrs.csv")
    if solve is None or percall is None:
        return
    per_iter = {
        (r["resolution"], r["method"]): float(r["solve_seconds"]) / int(r["iters"])
        for r in solve
        if r["status"] == "ok"
    }
    shares = {}
    for row in percall:
        res = row["label"].lower()
        gap_iter = per_iter[res, "curvilinear"] - per_iter[res, "engine"]
        gap_call = float(row["curvilinear_percall_s"]) - float(row["engine_percall_s"])
        shares[res] = gap_call / gap_iter
    mismatches = [
        f"{res} share {s:.3f} <= 0.8" for res, s in shares.items() if s <= 0.8
    ]
    report_artifact(
        "regrid_percall_abrs.csv",
        mismatches,
        "realign share of the per-iteration gap "
        + "/".join(f"{s:.3f}" for s in shares.values())
        + " > four fifths at every grid",
    )


# Section 2's MPC readings at their printed precision:
# (quantity, rounding digits of value, expected value).
LABOR_MPC_PINS = (
    ("mpc_first_segment", 2, 0.48),
    ("mpc_trough", 3, 0.024),
    ("mpc_plateau", 2, 0.06),
)


def check_labor_mpc_csv() -> None:
    """Section 2: the age-0 MPC is about 0.48 at the borrowing constraint (the
    first knot segment from m = 0), about 0.024 near m = 4.4 and about 0.06 from
    m = 20; expected future labor reaches zero near m = 18
    (consumption_labor_mpc.csv).
    """
    rows = load_csv_rows("consumption_labor_mpc.csv")
    if rows is None:
        return
    by_q = {r["quantity"]: (float(r["m"]), float(r["value"])) for r in rows}
    mismatches = [
        f"{q} {by_q[q][1]:.4g} != {want}"
        for q, digits, want in LABOR_MPC_PINS
        if round(by_q[q][1], digits) != want
    ]
    if not 4.0 <= by_q["mpc_trough"][0] <= 5.0:
        mismatches.append(f"trough at m = {by_q['mpc_trough'][0]} outside [4, 5]")
    zero_m = by_q["future_labor_first_zero"][0]
    if round(zero_m) != 18:
        mismatches.append(f"future labor first 0 at m = {zero_m}, not about 18")
    report_artifact(
        "consumption_labor_mpc.csv",
        mismatches,
        "MPC about 0.48 at the constraint, 0.024 near m = 4.4, 0.06 from m = 20; "
        "future labor zero near m = 18",
    )


def validate_artifact_consistency() -> None:
    """Cross-check committed CSV artifacts against paper-quoted numbers.

    Read-only: no solves. Covers timing_negm.csv, refinement_bounds_abrs.csv,
    extrap_frequency_abrs.csv, solvetime_abrs.csv, regrid_percall_abrs.csv and
    consumption_labor_mpc.csv. The expected values are
    the manuscript's quoted numbers recorded here as constants, not parsed
    from content/paper/; when the paper's numbers change, update these
    constants in the same commit.
    """
    check_timing_negm_csv()
    check_refinement_bounds_csv()
    check_extrap_frequency_csv()
    check_solvetime_ratio_csv()
    check_realign_share_csv()
    check_labor_mpc_csv()


# =============================================================================
# Main
# =============================================================================
def run_validation(label, fn, *args):
    """Run one validator; on exception log as INCONCLUSIVE/ERROR, not a claim failure.

    Distinguishes between:
    - Validation failures: raised by log_fail() inside fn (already recorded).
    - Infrastructure/import crashes: unexpected exceptions caught here, recorded as
      errors (INCONCLUSIVE) so they do not inflate the failed-claims count.
    """
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - isolates one validator's crash; see docstring
        tb = traceback.format_exc()
        log_error(
            f"{label} - validator crashed (INCONCLUSIVE)",
            f"{type(e).__name__}: {e}\n{tb}",
        )
        return None


def print_validation_summary() -> int:
    """Print pass/fail/warn/error counts; exit non-zero on failures or errors.

    Exit codes:
        0 -> all validations passed (warnings allowed)
        1 -> at least one validation FAILED (a claim was actively refuted)
        2 -> at least one INFRASTRUCTURE ERROR (a validation could not run);
             returned only when there are no failures. Treating "inconclusive"
             as success would let a ``MemoryError`` or import failure mask a
             broken claim, so CI must fail on it.
    """
    logger.info(
        "SUMMARY: %d passed, %d failed, %d warnings, %d errors",
        len(RESULTS["passed"]),
        len(RESULTS["failed"]),
        len(RESULTS["warnings"]),
        len(RESULTS["errors"]),
    )
    if RESULTS["errors"]:
        for claim, details in RESULTS["errors"]:
            logger.error(
                "ERROR: %s: %s",
                claim,
                details.split("\n")[0] if details else "",
            )

    if RESULTS["failed"]:
        for claim, details in RESULTS["failed"]:
            logger.error("FAILED CLAIM: %s: %s", claim, details)
        return 1
    if RESULTS["errors"]:
        return 2
    return 0


def main():
    """Run all validations, after one JIT warmup so the timed sections run warm."""
    start_logging()
    warmup_jit()
    run_validation("Consumption-labor params", validate_consumption_labor_params)
    run_validation("Health model params", validate_health_model_params)
    run_validation("Speedup claims", validate_speedup_claims)
    agent = run_validation("Full model speedup", validate_full_model_speedup)
    run_validation("Interpolation convergence", validate_interpolation_convergence)
    run_validation("Complexity scaling", validate_complexity_scaling)
    run_validation("Regrid linear scaling", validate_regrid_linear_scaling)

    if agent is None:
        log_error(
            "Health-mesh checks - not run (INCONCLUSIVE)",
            "the full-model solve that supplies the mesh crashed",
        )
    else:
        run_validation("Health-mesh checks", validate_health_meshes, agent)
    run_validation("Health summary sweep IPM", validate_summary_sweep_ipm)
    run_validation("Artifact consistency", validate_artifact_consistency)

    return print_validation_summary()


if __name__ == "__main__":
    sys.exit(main())
