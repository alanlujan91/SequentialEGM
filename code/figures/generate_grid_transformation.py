"""Generate Figure 2, the health model's grid transformation.

This script creates a figure showing the transformation from regular exogenous
grids to warped endogenous grids in the health investment model, using actual
model solution data rather than synthetic warping.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from engine.engine2d import Engine2DInterp
from figure_styles import (
    add_mesh_legend,
    apply_engine_style,
    exogenous_panel,
    label_panel,
    plot_mesh_lines,
    save_engine_figure,
    scatter_with_colorbar,
)
from HARKmodels.ConsHealthSeparableModel import (
    SequentialHealthConsumerType,
    init_basic_health,
    invert_sequential_stages,
    make_solution_terminal_ConsBasicHealth,
    post_decision_marginals,
)
from numerics import require_finite

log = logging.getLogger(__name__)


def get_health_model_grids():
    """Solve health model and extract exogenous/endogenous grids.

    Returns the grid transformation from the EGM solution:
    - Exogenous: (a, H) - post-decision assets x post-investment health
    - Endogenous: (m, h) - market resources x beginning health
    """
    # Set up parameters for a clear grid visualization
    params = init_basic_health.copy()
    params["aXtraCount"] = 25  # Smaller grid for visibility
    params["hLvlCount"] = 20
    params["cycles"] = 1
    params["interpolator"] = Engine2DInterp

    # Construction builds the grids, ShockDstn and DieProbMax; no solve needed.
    agent = SequentialHealthConsumerType(**params)

    # Get grid parameters
    aLvlGrid = agent.aLvlGrid
    HLvlGrid = agent.HLvlGrid

    # Terminal continuation: agent.solution[0] would feed the backward step its
    # own output and plot a three-period model, where the caption has two.
    solution_next = make_solution_terminal_ConsBasicHealth()

    DiscFac = agent.DiscFac
    Rfree = agent.Rfree
    CRRA = agent.CRRA
    alpha = agent.HealthProdExp
    gamma = agent.HealthProdFac
    DieProbMax = agent.DieProbMax[0]
    ShockDstn = agent.ShockDstn[0]
    rho = CRRA

    # Create exogenous grid
    (aMat, HMat) = np.meshgrid(aLvlGrid, HLvlGrid, indexing="ij")

    # The solver's own stage functions, so the figure cannot drift from them.
    _, w_a, w_H = post_decision_marginals(
        solution_next,
        ShockDstn,
        aMat,
        HMat,
        (Rfree, rho, alpha, gamma),
        DieProbMax,
        DiscFac,
    )
    cLvl, _lLvl, nLvl, mLvl, hLvl = invert_sequential_stages(
        w_a,
        w_H,
        aMat,
        HMat,
        rho,
        alpha,
        gamma,
    )

    # Validate endogenous grid (finite). Negative health is expected: at the
    # lowest H nodes the inversion can assign more investment than the node
    # holds. Those preimages lie outside the state space (OA-A).
    require_finite(mLvl=mLvl, hLvl=hLvl)
    if np.any(hLvl < 0):
        log.info("Negative health capital: min=%.4f", hLvl.min())

    return {
        "exog_x": aMat,
        "exog_y": HMat,
        "endog_x": mLvl,
        "endog_y": hLvl,
        "consumption": cLvl,
        "health_investment": nLvl,
    }


def draw_health_grid_transformation(grids):
    """The two-panel grid transformation figure for ``get_health_model_grids``
    output: the exogenous ``(a, H)`` grid beside the endogenous ``(m, h)``
    mesh, colored by consumption. Returns the figure.
    """
    # Extract data
    a_exog = grids["exog_x"]
    H_exog = grids["exog_y"]
    m_endog = grids["endog_x"]
    h_endog = grids["endog_y"]
    consumption = grids["consumption"]

    n_a, n_H = a_exog.shape

    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    exogenous_panel(
        axes[0],
        a_exog,
        H_exog,
        "Exogenous Grid ($t=0$)",
        "Post-Decision Assets $a_t$",
        "Post-Investment Health $H_t$",
    )

    # Panel (b), the endogenous (m, h) grid at t=0, colored by consumption.
    ax = axes[1]

    # Axis limits of the plotted region.
    m_max = 50
    h_min = 0  # health is nonnegative

    # Curvilinear grid lines, clipped to the focus region: every 10th-ish column
    # (constant H) and 12th-ish row (constant a).
    plot_mesh_lines(
        ax,
        m_endog,
        h_endog,
        step=(max(1, n_a // 12), max(1, n_H // 10)),
        mask=(m_endog <= m_max) & (h_endog >= h_min),
        linewidth=1.0,
        alpha=0.7,
    )

    # Scatter points colored by consumption (only within limits)
    m_flat = m_endog.flatten()
    h_flat = h_endog.flatten()
    c_flat = consumption.flatten()
    mask_flat = (m_flat <= m_max) & (h_flat >= h_min)

    scatter_with_colorbar(
        ax,
        m_flat[mask_flat],
        h_flat[mask_flat],
        c_flat[mask_flat],
        "Consumption $c_t$",
    )

    # Pad the focus viewport by 4% so markers clear the spines instead of
    # sitting flush against the hard-coded (0, m_max) / (h_min, None) cutoffs.
    y_top = float(np.nanmax(h_flat[mask_flat])) if np.any(mask_flat) else h_endog.max()
    x_pad = 0.04 * m_max
    y_pad = 0.04 * (y_top - h_min)
    ax.set_xlim(-x_pad, m_max + x_pad)
    ax.set_ylim(h_min - y_pad, y_top + y_pad)
    label_panel(
        ax,
        "(b)",
        "Endogenous Grid ($t=0$)",
        "Market Resources $m_t$",
        "Beginning Health $h_t$",
    )
    ax.grid(False)

    # Two mesh-line families are otherwise indistinguishable without a key.
    add_mesh_legend(
        ax,
        "Constant post-investment health $H$",
        "Constant post-decision assets $a$",
    )

    plt.tight_layout()
    return fig


def generate_grid_transformation():
    """Generate Figure 2, the health model's grid transformation."""
    apply_engine_style()
    fig = draw_health_grid_transformation(get_health_model_grids())
    return save_engine_figure(fig, "grid_transformation.png")


if __name__ == "__main__":
    generate_grid_transformation()
