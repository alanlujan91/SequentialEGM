"""Generate the grid-homeomorphism figure.

This script visualizes the homeomorphism between a curvilinear grid in
physical space and a rectilinear grid in index space: the same grid seen
in two coordinate systems, with one tracked cell shared across panels.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from cli import start_logging
from figure_styles import (
    ENGINE_COLORS,
    apply_engine_style,
    get_shared_grid_data,
    label_panel,
    plot_mesh_lines,
    save_engine_figure,
)
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon

logger = logging.getLogger(__name__)

# One interior cell tracked across both panels so a reader can trace the
# same quad from physical space into index space.
# Chosen for visible warp (not a boundary cell).
HIGHLIGHT_J0, HIGHLIGHT_K0 = 1, 2
HIGHLIGHT_COLOR = ENGINE_COLORS["error"]

# One shared color for grid lines and nodes in both panels, so the panels
# read as one grid seen in two coordinate systems rather than two grids.
GRID_COLOR = ENGINE_COLORS["engine"]


def highlight_cell(ax, xx, yy, zorder=4) -> None:
    """Draw the tracked cell as a filled quad with a bold, opaque outline."""
    js = [HIGHLIGHT_J0, HIGHLIGHT_J0 + 1, HIGHLIGHT_J0 + 1, HIGHLIGHT_J0]
    ks = [HIGHLIGHT_K0, HIGHLIGHT_K0, HIGHLIGHT_K0 + 1, HIGHLIGHT_K0 + 1]
    verts = list(zip(xx[js, ks], yy[js, ks], strict=True))
    patch = Polygon(
        verts,
        closed=True,
        facecolor=to_rgba(HIGHLIGHT_COLOR, alpha=0.35),
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=2.2,
        zorder=zorder,
    )
    ax.add_patch(patch)


def draw_grid_panel(ax, xx, yy, title, xlabel, ylabel) -> None:
    """Grid lines and nodes in ``GRID_COLOR`` with the tracked cell, on equal
    axes; ``title`` is the ``(label, text)`` pair for ``label_panel``.
    """
    plot_mesh_lines(
        ax,
        xx,
        yy,
        colors=(GRID_COLOR, GRID_COLOR),
        linewidth=1.5,
        alpha=0.8,
    )
    highlight_cell(ax, xx, yy)
    ax.scatter(xx, yy, c=GRID_COLOR, s=35, edgecolor="black", linewidth=0.6, zorder=5)
    label_panel(ax, *title, xlabel, ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def draw_grid_homeomorphism():
    """Draw the grid-homeomorphism figure and return it."""
    # Grid parameters
    J, K = 6, 7
    warp_factor = 0.18  # Slightly stronger warping than mapping/partition figures (0.15) for visual contrast

    # Get grid data
    _, _, xx_endog, yy_endog = get_shared_grid_data(J, K, warp_factor)

    # Create figure with two panels: physical space and index space
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))

    # Panel (a): the curvilinear grid in physical space.
    draw_grid_panel(
        axes[0],
        xx_endog,
        yy_endog,
        ("(a)", "Curvilinear Grid"),
        "$x$ (physical)",
        "$y$ (physical)",
    )

    # Panel (b): one-based index axes, matching the paper's formal statements
    # (j = 1..J, k = 1..K; the homeomorphism domain is [1, J] x [1, K]).
    JJ, KK = np.meshgrid(np.arange(1, J + 1), np.arange(1, K + 1), indexing="ij")
    draw_grid_panel(
        axes[1],
        JJ,
        KK,
        ("(b)", "Rectilinear Index Space"),
        "Index $j$",
        "Index $k$",
    )

    # Add arrow between the panels
    fig.text(
        0.5,
        0.5,
        r"$\longleftrightarrow$",
        fontsize=20,
        ha="center",
        va="center",
        color=GRID_COLOR,
    )

    plt.tight_layout()
    return fig


def generate_grid_homeomorphism():
    """Generate the figure showing the homeomorphism between grid spaces."""
    apply_engine_style()
    return save_engine_figure(draw_grid_homeomorphism(), "grid_homeomorphism.png")


if __name__ == "__main__":
    start_logging()
    path = generate_grid_homeomorphism()
    logger.info("Saved grid-homeomorphism figure to %s", path)
