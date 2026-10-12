"""Health Investment Model Interpolation Benchmark.

This script benchmarks three different interpolation methods for the health investment
model from White (2015):

1. HARK's Curvilinear2DInterp (baseline implementation)
2. ENGINE (ENdogenous Grid INterpolation and Extrapolation)
3. Scipy's LinearNDInterpolator (Delaunay)

The model uses the Endogenous Grid Method (EGM) which creates irregular, curvilinear
grids - exactly the use case ENGINE was designed for.

The module uses ConsHealthSeparableModel which provides:
- 2 solvers: Simultaneous (baseline) and Sequential (EGM^n)
- 3 interpolation methods via the `interpolator` parameter
"""

from __future__ import annotations

import logging
import os
import sys
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from engine.utilities import Delaunay2DInterp
from figure_styles import FIGURE_DIR, apply_engine_style, save_engine_figure
from HARK.interpolation import Curvilinear2DInterp

# Import solvers and interpolators
from HARKmodels.ConsHealthSeparableModel import (
    BasicHealthConsumerType,
    SequentialHealthConsumerType,
    health_grid_params,
)
from HARKmodels.health_interp import HealthEngine2DInterp
from numerics import rel_diff
from repo_paths import write_rows

log = logging.getLogger(__name__)

# Moderate grid used by both accuracy checks below. hLvlMin=0.01 avoids the
# singularity at h=0 in the health model.
ACCURACY_CHECK_PARAMS = health_grid_params(30.0, 25, 30.0, 20)

# The archived readings the paper quotes, one CSV per check.
POLICY_DIFF_CSV = os.path.join(FIGURE_DIR, "health_interp_policy_diff.csv")
POLICY_DIFF_HEADER = ["method", "policy", "rel_mean", "rel_max", "n_valid"]
SOLVER_EQUIVALENCE_CSV = os.path.join(FIGURE_DIR, "health_solver_equivalence.csv")
SOLVER_EQUIVALENCE_HEADER = ["policy", "abs_mean", "abs_max"]


@cache
def solved(agent_type, interpolator):
    """``agent_type`` solved on ``ACCURACY_CHECK_PARAMS``, once per pairing."""
    agent = agent_type(interpolator=interpolator, **ACCURACY_CHECK_PARAMS)
    agent.solve()
    return agent


def query_mesh(n_m, n_h):
    """``(M, H)`` query mesh over ``[0.5, 25]^2`` with ``n_m x n_h`` points."""
    return np.meshgrid(
        np.linspace(0.5, 25.0, n_m),
        np.linspace(0.5, 25.0, n_h),
        indexing="ij",
    )


def interpolation_accuracy():
    """Policies of HARK's Curvilinear (the reference), ENGINE and Delaunay on the
    ``query_mesh(50, 40)`` points, each from its own backward solve, keyed by
    method name, with the query mesh under ``"grids"``.
    """
    M_test, H_test = query_mesh(50, 40)
    results = {
        name: solved(BasicHealthConsumerType, interp).solution[0](M_test, H_test)
        for name, interp in (
            ("HARK", Curvilinear2DInterp),
            ("ENGINE", HealthEngine2DInterp),
            ("Delaunay", Delaunay2DInterp),
        )
    }
    return {**results, "grids": (M_test, H_test)}


def policy_diff_rows(accuracy_results):
    """``POLICY_DIFF_HEADER`` rows: each method's relative policy difference
    from the curvilinear reference, the readings the paper's figure
    discussion quotes.
    """
    rows = []
    for method, name in (("engine", "ENGINE"), ("delaunay", "Delaunay")):
        for policy, idx in (("c", 1), ("n", 2)):
            mean, mx, n_valid = relative_error(
                accuracy_results["HARK"][idx],
                accuracy_results[name][idx],
            )
            rows.append([method, policy, f"{mean:.3e}", f"{mx:.3e}", str(n_valid)])
    return rows


def relative_error(ref, approx):
    """Mean and max of ``rel_diff(approx, ref)`` over its finite entries, and
    the count of those entries; any inf or NaN entry is logged.
    """
    rel = rel_diff(approx, ref)
    finite = np.isfinite(rel)
    if not finite.all():
        log.warning(
            "%d of %d relative errors are inf or NaN",
            (~finite).sum(),
            rel.size,
        )
    return np.mean(rel[finite]), np.max(rel[finite]), int(finite.sum())


def solver_equivalence_rows():
    """``SOLVER_EQUIVALENCE_HEADER`` rows: the sequential EGM^n solver's policy
    gap to the simultaneous solver, both under HARK's curvilinear interpolant.
    """
    agent_simult = solved(BasicHealthConsumerType, Curvilinear2DInterp)
    agent_seq = solved(SequentialHealthConsumerType, Curvilinear2DInterp)
    M_test, H_test = query_mesh(30, 25)

    _, c_simult, n_simult = agent_simult.solution[0](M_test, H_test)
    _, c_seq, n_seq = agent_seq.solution[0](M_test, H_test)

    return [
        [policy, f"{np.mean(np.abs(a - b)):.3e}", f"{np.max(np.abs(a - b)):.3e}"]
        for policy, a, b in (("c", c_simult, c_seq), ("n", n_simult, n_seq))
    ]


def draw_interpolation_errors(accuracy_results):
    """The consumption disagreement maps of the paper's error-map figure, for
    ``interpolation_accuracy`` output. Returns the figure.

    Both panels share one color scale (the comparison is the message), use an
    unsmoothed pcolormesh on the actual query grid, and mark each method's
    maximum disagreement so the numbers quoted in the text are locatable.
    """
    hark_results = accuracy_results["HARK"]
    engine_results = accuracy_results["ENGINE"]
    delaunay_results = accuracy_results["Delaunay"]
    M_test, H_test = accuracy_results["grids"]

    # Disagreement fields are clipped at 1e-9 (well below every plotted decade)
    # so exact agreement does not send log10 to -inf.
    c_hark = hark_results[1]
    log_err = {}
    for name, res in (
        ("ENGINE", engine_results),
        ("Delaunay triangulation", delaunay_results),
    ):
        err = rel_diff(res[1], c_hark)
        err[~np.isfinite(err)] = np.nan
        log_err[name] = np.log10(np.clip(err, 1e-9, None))

    vmin, vmax = -9.0, 0.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7, 3.1),
        sharey=True,
        constrained_layout=True,
    )
    meshes = []
    for ax, (name, field) in zip(axes, log_err.items(), strict=False):
        mesh = ax.pcolormesh(
            M_test,
            H_test,
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )
        meshes.append(mesh)
        ax.set_title(name)
        ax.set_xlabel("Market Resources")

        # Mark the worst disagreement so the text's quoted maximum is locatable.
        flat = np.nanargmax(field)
        i, j = np.unravel_index(flat, field.shape)
        ax.plot(
            M_test[i, j],
            H_test[i, j],
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
        )
        max_err = 10.0 ** field[i, j]
        mantissa, exponent = f"{max_err:.1e}".split("e")
        ax.annotate(
            f"max ${mantissa} \\times 10^{{{int(exponent)}}}$",
            xy=(M_test[i, j], H_test[i, j]),
            xytext=(8, 8),
            textcoords="offset points",
            color="black",
            fontsize=8,
        )
    axes[0].set_ylabel("Health Capital")

    cbar = fig.colorbar(meshes[-1], ax=axes, pad=0.02)
    tick_decades = np.arange(-8, 1, 2)
    cbar.set_ticks(tick_decades)
    cbar.set_ticklabels([f"$10^{{{int(d)}}}$" for d in tick_decades])
    cbar.set_label("Relative consumption disagreement\nwith curvilinear reference")
    return fig


def main(accuracy_only=False) -> None:
    """Write the solver-equivalence CSV, then the policy-difference CSV and the
    error-map figure; ``accuracy_only`` (CLI flag ``--accuracy-only``) skips the
    solver-equivalence check.
    """
    if not accuracy_only:
        write_rows(
            SOLVER_EQUIVALENCE_CSV,
            SOLVER_EQUIVALENCE_HEADER,
            solver_equivalence_rows(),
        )
    accuracy_results = interpolation_accuracy()
    write_rows(POLICY_DIFF_CSV, POLICY_DIFF_HEADER, policy_diff_rows(accuracy_results))
    apply_engine_style()
    save_engine_figure(
        draw_interpolation_errors(accuracy_results),
        "health_model_interpolation_errors.png",
    )


if __name__ == "__main__":
    main(accuracy_only="--accuracy-only" in sys.argv[1:])
