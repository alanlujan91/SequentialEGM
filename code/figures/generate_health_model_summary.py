"""Generate Health Model Summary Figure.

This script demonstrates that EGM^n + ENGINE is faster than HARK baseline
across both:
1. Increasing cycle counts (backward induction steps)
2. Increasing grid sizes

Both sweeps pass through the paper's timing point (``PAPER_TIMING_GRID`` at
``PAPER_TIMING_CYCLES``). The grid sweep scales the asset axis at the paper's
health density, plus one small grid, and the cycle sweep stops at 15: denser
health grids and 20 cycles put the endogenous mesh outside ENGINE's ``def-ipm``
precondition (the health coordinate folds back between the two lowest health
nodes), where ``Engine2DInterp`` raises.

Key comparison: EGM^n + ENGINE vs HARK baseline
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cli import start_logging
from engine.engine2d import warmup_jit
from figure_styles import (
    ENGINE_COLORS,
    FIGURE_DIR,
    apply_engine_style,
    format_axes_for_publication,
    save_engine_figure,
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
)
from HARKmodels.health_interp import HealthEngine2DInterp
from repo_paths import read_rows, write_rows
from scipy import stats
from timing import timed

logger = logging.getLogger(__name__)

CSV_PATH = Path(FIGURE_DIR) / "health_model_summary.csv"
TITLE_STYLE = {"fontsize": 11, "fontweight": "bold", "pad": 10}


def grid_label(grid, sep="x") -> str:
    """``"J<sep>K"`` label of the ``(aXtraCount, hLvlCount)`` grid."""
    return f"{grid[0]}{sep}{grid[1]}"


def timed_solve(AgentClass, params, n_timed=10):
    """Time ``AgentClass(**params).solve()`` after one untimed warm solve; report
    mean and 95% CI.
    """
    AgentClass(**params).solve()

    times = []
    for _ in range(n_timed):
        agent = AgentClass(**params)
        times.append(timed(agent.solve)[1])

    avg_time = float(np.mean(times))
    std_time = float(np.std(times, ddof=1))
    # t critical value, not 1.96: with n_timed = 10 the normal approximation
    # understates the 95% CI by ~15% (t_{0.975, 9} = 2.262).
    ci_95 = stats.t.ppf(0.975, len(times) - 1) * std_time / np.sqrt(len(times))
    return {"mean": avg_time, "ci_95": ci_95}


def time_configs(configs, base_params):
    """Time each ``(name, AgentClass, interpolator)`` config at ``base_params``."""
    return {
        name: timed_solve(AgentClass, {**base_params, "interpolator": interpolator})
        for name, AgentClass, interpolator in configs
    }


def run_benchmarks():
    """Benchmark solvers across cycles and grid sizes."""
    cycle_counts = list(SUMMARY_CYCLES)
    grid_sizes = list(SUMMARY_GRIDS)

    warmup_jit()

    # Key configs to compare
    configs = [
        ("HARK baseline", BasicHealthConsumerType, Curvilinear2DInterp),
        ("ENGINE baseline", BasicHealthConsumerType, HealthEngine2DInterp),
        ("EGM$^n$ + ENGINE", SequentialHealthConsumerType, HealthEngine2DInterp),
    ]

    base_params = init_basic_health.copy()

    # Results storage
    results = {
        "cycles": {},  # Varying cycles, fixed grid
        "grids": {},  # Varying grid, fixed cycles
    }

    # =========================================================================
    # Test 1, varying cycles on the paper's (HARK default) grid
    # =========================================================================

    base_params.update(paper_timing_params())

    for cycles in cycle_counts:
        base_params["cycles"] = cycles

        results["cycles"][cycles] = time_configs(configs, base_params)

    # =========================================================================
    # Test 2, varying grid sizes at the paper's fixed cycle count
    # =========================================================================

    base_params["cycles"] = PAPER_TIMING_CYCLES

    for grid in grid_sizes:
        base_params["aXtraCount"] = grid[0]
        base_params["hLvlCount"] = grid[1]
        # The paper's grid at the paper's cycle count was timed in Test 1.
        timings = (
            results["cycles"][PAPER_TIMING_CYCLES]
            if grid == PAPER_TIMING_GRID
            else time_configs(configs, base_params)
        )
        results["grids"][grid_label(grid)] = {
            "n_points": grid[0] * grid[1],
            **timings,
        }

    return results, cycle_counts, grid_sizes


def means_and_cis(bucket, keys):
    """Stack per-key (mean, ci_95) pairs into flat arrays.

    Returns (hark_mean, hark_ci, eng_mean, eng_ci, egmn_mean, egmn_ci).
    """

    def stack(metric, kind):
        return np.array([bucket[k][metric][kind] for k in keys])

    return (
        stack("HARK baseline", "mean"),
        stack("HARK baseline", "ci_95"),
        stack("ENGINE baseline", "mean"),
        stack("ENGINE baseline", "ci_95"),
        stack("EGM$^n$ + ENGINE", "mean"),
        stack("EGM$^n$ + ENGINE", "ci_95"),
    )


def err_from_ratio(num, num_err, den, den_err):
    """Error propagation for a/b with independent errors."""
    return (num / den) * np.sqrt((num_err / num) ** 2 + (den_err / den) ** 2)


HARK_LABEL = "Simultaneous + curvilinear"
ENGINE_LABEL = "Simultaneous + ENGINE"
EGMN_LABEL = r"Sequential (EGM$^n$) + ENGINE"


def plot_bands(ax, plot_fn, x, rows) -> None:
    """One line plus its +/- band per ``(y, err, fmt, label, color, linewidth,
    markersize, alpha, band_alpha)`` row. ``plot_fn`` names the axis's
    line-drawing method (``"plot"``, ``"loglog"``, ``"semilogx"``).
    """
    for y, err, fmt, label, color, lw, ms, alpha, band_alpha in rows:
        getattr(ax, plot_fn)(
            x,
            y,
            fmt,
            label=label,
            color=color,
            linewidth=lw,
            markersize=ms,
            alpha=alpha,
        )
        ax.fill_between(x, y - err, y + err, color=color, alpha=band_alpha)


def plot_solve_times(ax, plot_fn, x, series, title) -> None:
    """Solve time of the three configurations with their 95% bands."""
    h, hci, e, eci, egmn, egmnci = series
    hark, engine = ENGINE_COLORS["hark"], ENGINE_COLORS["engine"]
    plot_bands(
        ax,
        plot_fn,
        x,
        (
            (h, hci, "o-", HARK_LABEL, hark, 2, 8, None, 0.2),
            (e, eci, "s--", ENGINE_LABEL, engine, 1.5, 6, 0.7, 0.15),
            (egmn, egmnci, "s-", EGMN_LABEL, engine, 2, 8, None, 0.2),
        ),
    )
    ax.set_ylabel("Solve Time (seconds)")
    ax.set_title(title, **TITLE_STYLE)


def plot_speedups(ax, plot_fn, x, series, title):
    """Speedup of both ENGINE configurations over simultaneous + curvilinear,
    with bands and the y = 1 line. Returns the EGM$^n$ speedup and its error.
    """
    h, hci, e, eci, egmn, egmnci = series
    sp_egmn, sp_egmn_err = h / egmn, err_from_ratio(h, hci, egmn, egmnci)
    sp_engine, sp_engine_err = h / e, err_from_ratio(h, hci, e, eci)
    engine = ENGINE_COLORS["engine"]
    plot_bands(
        ax,
        plot_fn,
        x,
        (
            (sp_egmn, sp_egmn_err, "s-", EGMN_LABEL, engine, 2.5, 10, None, 0.2),
            (sp_engine, sp_engine_err, "s--", ENGINE_LABEL, engine, 1.5, 6, 0.7, 0.15),
        ),
    )
    ax.axhline(
        y=1,
        color=ENGINE_COLORS["neutral_dark"],
        linestyle="--",
        alpha=0.5,
        linewidth=1,
    )
    ax.set_ylabel("Speedup vs simultaneous + curvilinear")
    ax.set_title(title, **TITLE_STYLE)
    return sp_egmn, sp_egmn_err


def annotate_speedup_points(ax, x_values, sp_egmn, sp_egmn_err) -> None:
    """Label each EGM^n+ENGINE speedup point with its value and CI."""
    for i, x in enumerate(x_values):
        ax.annotate(
            f"{sp_egmn[i]:.2f}×±{sp_egmn_err[i]:.2f}",
            (x, sp_egmn[i]),
            textcoords="offset points",
            xytext=(9, 9),
            fontsize=7,
            color=ENGINE_COLORS["engine"],
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )


def finish_cycles_axis(ax, cycles) -> None:
    """Shared x-label/legend/x-tick/format-axis tail for the vs-cycles panels."""
    ax.set_xlabel("Number of Cycles")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_xticks(cycles)
    format_axes_for_publication(ax)


def finish_grid_axis(ax, n_points, grid_labels) -> None:
    """Shared x-label/legend/x-tick/format-axis tail for the vs-grid-size panels."""
    ax.set_xlabel("Grid Size (J × K points)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_xticks(n_points)
    ax.set_xticklabels([g.replace("x", "×") for g in grid_labels])
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    format_axes_for_publication(ax)


def print_speedup_line(label, bucket) -> None:
    hark_mean = bucket["HARK baseline"]["mean"]
    egmn_mean = bucket["EGM$^n$ + ENGINE"]["mean"]
    hark_ci = bucket["HARK baseline"]["ci_95"]
    egmn_ci = bucket["EGM$^n$ + ENGINE"]["ci_95"]
    speedup = hark_mean / egmn_mean
    err = err_from_ratio(
        np.asarray(hark_mean),
        np.asarray(hark_ci),
        np.asarray(egmn_mean),
        np.asarray(egmn_ci),
    )
    logger.info("%s: speedup %.2fx +/- %.2f", label, speedup, float(err))


def print_summary(results, cycle_counts, grid_labels) -> None:
    for c in cycle_counts:
        print_speedup_line(f"Cycles={c:2d}", results["cycles"][c])
    for g in grid_labels:
        print_speedup_line(f"Grid={g}", results["grids"][g])


def persist_results(results, cycle_counts) -> None:
    """Write the timing readings to CSV so replots never re-time the solves."""
    sweeps = [("cycles", c, "", results["cycles"][c]) for c in cycle_counts]
    sweeps += [
        ("grids", g, bucket["n_points"], bucket)
        for g, bucket in results["grids"].items()
    ]
    write_rows(
        CSV_PATH,
        ["sweep", "label", "n_points", "series", "mean_s", "ci95_s"],
        [
            [sweep, label, n_points, series, f"{v['mean']:.6e}", f"{v['ci_95']:.6e}"]
            for sweep, label, n_points, bucket in sweeps
            for series, v in bucket.items()
            if series != "n_points"
        ],
    )
    logger.info("wrote %s", CSV_PATH)


def load_results_from_csv():
    """Rebuild the results structure run_benchmarks() returns from the CSV."""
    results = {"cycles": {}, "grids": {}}
    cycle_counts = []
    for row in read_rows(CSV_PATH):
        entry = {"mean": float(row["mean_s"]), "ci_95": float(row["ci95_s"])}
        if row["sweep"] == "cycles":
            c = int(row["label"])
            if c not in results["cycles"]:
                results["cycles"][c] = {}
                cycle_counts.append(c)
            results["cycles"][c][row["series"]] = entry
        else:
            g = row["label"]
            results["grids"].setdefault(g, {"n_points": int(row["n_points"])})
            results["grids"][g][row["series"]] = entry
    return results, cycle_counts


def generate_health_model_summary(plot_only=False):
    """Time the health solvers over cycles and grids, then plot the summary figure.

    With ``plot_only`` (CLI flag ``--plot-only``), replot from the persisted
    CSV instead of re-timing the solves.
    """
    apply_engine_style()
    if plot_only:
        results, cycle_counts = load_results_from_csv()
    else:
        results, cycle_counts, _grid_sizes = run_benchmarks()
        persist_results(results, cycle_counts)

    cycles = np.array(cycle_counts)
    grid_labels = list(results["grids"].keys())
    n_points = np.array([results["grids"][g]["n_points"] for g in grid_labels])

    by_cycles = means_and_cis(results["cycles"], cycle_counts)
    by_grid = means_and_cis(results["grids"], grid_labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    plot_solve_times(
        ax,
        "plot",
        cycles,
        by_cycles,
        f"(a) Solve Time vs Cycles (grid={grid_label(PAPER_TIMING_GRID, '×')})",
    )
    finish_cycles_axis(ax, cycles)
    ax = axes[0, 1]
    speedup = plot_speedups(ax, "plot", cycles, by_cycles, "(b) Speedup vs Cycles")
    finish_cycles_axis(ax, cycles)
    annotate_speedup_points(ax, cycles, *speedup)
    ax = axes[1, 0]
    plot_solve_times(
        ax,
        "loglog",
        n_points,
        by_grid,
        f"(c) Solve Time vs Grid Size (cycles={PAPER_TIMING_CYCLES})",
    )
    finish_grid_axis(ax, n_points, grid_labels)
    ax = axes[1, 1]
    speedup = plot_speedups(
        ax,
        "semilogx",
        n_points,
        by_grid,
        "(d) Speedup vs Grid Size",
    )
    finish_grid_axis(ax, n_points, grid_labels)
    annotate_speedup_points(ax, n_points, *speedup)

    plt.tight_layout()
    figure_path = save_engine_figure(fig, "health_model_summary.png")

    print_summary(results, cycle_counts, grid_labels)

    return figure_path


if __name__ == "__main__":
    start_logging()
    generate_health_model_summary(plot_only="--plot-only" in sys.argv[1:])
