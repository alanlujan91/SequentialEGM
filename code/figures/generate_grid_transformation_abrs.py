"""Generate the two-asset grid transformation figure (Section 3).

Solves the two-asset household block to convergence with the compact EGM$^n$
``@het`` block (``ssj.benchmark.solve_compact``), then extracts the Stage 0
endogenous mesh at the fixed point via ``ssj.benchmark.stage0_mesh``, which runs
the production kernel's Phase 1 (Stage 1 kinked liquid EGM plus closed-form
``Phi_1`` inversion) and returns the mesh instead of regridding it. The figure
therefore plots the exact arithmetic the solver runs: the exogenous ``(l, a')``
grid beside the endogenous ``(b, a)`` mesh colored by consumption.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from cli import start_logging
from figure_styles import (
    add_mesh_legend,
    apply_engine_style,
    exogenous_panel,
    label_panel,
    plot_mesh_lines,
    save_engine_figure,
    scatter_with_colorbar,
)
from numerics import require_finite
from ssj.benchmark import (
    assert_mesh_fold_free,
    assert_mesh_ipm,
    calibration,
    solve_compact,
    stage0_mesh,
)

logger = logging.getLogger(__name__)


def get_two_asset_meshes(nB=20, nA=20):
    """Solve the two-asset block and extract the Stage 0 mesh transformation.

    Returns the exogenous ``(l, a')`` grid and the endogenous ``(b, a)`` mesh at
    the converged fixed point, per productivity state, re-derived from the
    compact block's Phase 1 so the figure shows the same arithmetic the solver
    and tests pin.
    """
    cal = calibration(nB, nA)
    out = solve_compact(cal)  # raises unless the iteration converged
    mesh = stage0_mesh(out, cal)
    b_endo, a_endo = mesh["b_endo"], mesh["a_endo"]

    require_finite(a_endo=a_endo, b_endo=b_endo)
    assert_mesh_fold_free(b_endo, a_endo)
    assert_mesh_ipm(b_endo, a_endo)

    return {
        "exog_l": mesh["l_grid"],
        "exog_ap": mesh["a_grid"],
        "endog_b": b_endo,
        "endog_a": a_endo,
        "consumption": mesh["c"],
        "n_e": cal["Pi"].shape[0],
    }


def draw_grid_transformation(grids):
    """The two-panel grid transformation figure for ``get_two_asset_meshes``
    output: the exogenous ``(l, a')`` grid beside the endogenous ``(b, a)``
    mesh at the middle productivity state. Returns the figure.
    """
    l_grid = grids["exog_l"]
    a_grid = grids["exog_ap"]

    # Middle productivity state; the caption footnote records this choice.
    ie = grids["n_e"] // 2
    b_endog = grids["endog_b"][ie]
    a_endog = grids["endog_a"][ie]
    consumption = grids["consumption"][ie]

    (l_mesh, ap_mesh) = np.meshgrid(l_grid, a_grid, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    exogenous_panel(
        axes[0],
        l_mesh,
        ap_mesh,
        "Exogenous Grid",
        r"Liquid Resources $\ell$",
        r"Illiquid Choice $a'$",
    )

    # Panel (b): endogenous mesh (b, a) colored by consumption.
    ax = axes[1]
    plot_mesh_lines(ax, b_endog, a_endog, linewidth=1.0, alpha=0.7)
    scatter_with_colorbar(
        ax,
        b_endog.flatten(),
        a_endog.flatten(),
        consumption.flatten(),
        "Consumption $c$",
    )
    label_panel(
        ax,
        "(b)",
        "Endogenous Mesh (converged)",
        r"Liquid Wealth $b$",
        r"Illiquid Balance $a$",
    )
    for panel in axes:
        panel.set_xscale("symlog", linthresh=1.0)
        panel.set_yscale("symlog", linthresh=1.0)
        panel.grid(False)

    # Two mesh-line families are otherwise indistinguishable without a key.
    add_mesh_legend(
        ax,
        r"Constant illiquid choice $a'$",
        r"Constant liquid resources $\ell$",
    )

    # Tighten x to data extent: the raw symlog autoscale rounds up to the
    # next decade, which leaves a large dead range past the rightmost point.
    b_min, b_max = float(b_endog.min()), float(b_endog.max())
    b_pad = 0.04 * (b_max - b_min)
    ax.set_xlim(b_min - b_pad, b_max + b_pad)
    # A linear-space y-pad would be invisible against the multi-decade
    # symlog range; margins() pads in the transformed (log) space instead.
    ax.margins(y=0.04)

    plt.tight_layout()
    return fig


def generate_grid_transformation_abrs():
    """Generate the two-asset grid transformation figure for Section 3."""
    apply_engine_style()

    logger.info("Solving the two-asset household block for mesh data...")
    fig = draw_grid_transformation(get_two_asset_meshes())
    figure_path = save_engine_figure(fig, "grid_transformation_abrs.png")
    logger.info("Saved: %s", figure_path)
    return figure_path


if __name__ == "__main__":
    start_logging()
    generate_grid_transformation_abrs()
