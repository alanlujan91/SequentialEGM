"""Generate Figure X3: Conceptual Comparison of State Space Partitioning.

This script creates a professional comparison showing how different interpolation
methods partition and approach the warped endogenous grid space.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from figure_styles import (
    ENGINE_COLORS,
    apply_engine_style,
    format_axes_for_publication,
    get_shared_grid_data,
    panel_title,
    plot_mesh_lines,
    save_engine_figure,
)
from matplotlib import patches
from scipy.spatial import Delaunay

_GRID_POINT_STYLE = {
    "s": 25,
    "c": ENGINE_COLORS["endog_grid"],
    "alpha": 0.8,
    "edgecolor": "black",
    "linewidth": 0.5,
}
_QUERY_POINT_STYLE = {
    "s": 200,
    "c": "gold",
    "marker": "*",
    "edgecolor": "black",
    "linewidth": 2,
    "zorder": 10,
}
# Fraction of the data y-range added as blank headroom above the plot so the
# frameless upper-left legend sits in whitespace instead of over grid lines.
_LEGEND_TOP_PAD = 0.22


def finalize_panel(ax, label, title) -> None:
    format_axes_for_publication(
        ax,
        xlabel="Market Resources $(m_t)$",
        ylabel="Beginning Health $(h_t)$",
    )
    panel_title(ax, label, title)
    ax.set_xticks([])
    ax.set_yticks([])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + _LEGEND_TOP_PAD * (ymax - ymin))
    ax.set_aspect("equal", "box")
    ax.legend(fontsize=9, loc="upper left", frameon=False)


def draw_cell_weights(ax, vertices, color, a_star, h_star, label) -> None:
    """Shade the cell holding the query and draw a dashed spoke to each vertex."""
    ax.add_patch(
        patches.Polygon(
            vertices,
            closed=True,
            edgecolor=color,
            facecolor=color,
            alpha=0.2,
            linewidth=2,
        ),
    )
    for i, vert in enumerate(vertices):
        ax.plot(
            [vert[0], a_star],
            [vert[1], h_star],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=label if i == 0 else "",
        )


def panel_white(ax, xx_endog, yy_endog, a_star, h_star) -> None:
    plot_mesh_lines(ax, xx_endog, yy_endog, lw=1.2, alpha=0.6)
    ax.scatter(xx_endog, yy_endog, **_GRID_POINT_STYLE)
    ax.scatter(a_star, h_star, label="Query Point $(m^*, h^*)$", **_QUERY_POINT_STYLE)

    j_idx, k_idx = 2, 2
    quad_vertices = [
        (xx_endog[j_idx, k_idx], yy_endog[j_idx, k_idx]),
        (xx_endog[j_idx + 1, k_idx], yy_endog[j_idx + 1, k_idx]),
        (xx_endog[j_idx + 1, k_idx + 1], yy_endog[j_idx + 1, k_idx + 1]),
        (xx_endog[j_idx, k_idx + 1], yy_endog[j_idx, k_idx + 1]),
    ]
    draw_cell_weights(
        ax,
        quad_vertices,
        ENGINE_COLORS["white_method"],
        a_star,
        h_star,
        "Bilinear Weights",
    )


def panel_delaunay(ax, points, query_point, a_star, h_star) -> None:
    tri = Delaunay(points)
    ax.triplot(
        points[:, 0],
        points[:, 1],
        tri.simplices,
        linewidth=0.8,
        color=ENGINE_COLORS["neutral_medium"],
        alpha=0.5,
    )
    ax.scatter(points[:, 0], points[:, 1], **_GRID_POINT_STYLE)
    ax.scatter(a_star, h_star, label="Query Point $(m^*, h^*)$", **_QUERY_POINT_STYLE)

    triangle_index = tri.find_simplex(query_point)
    if triangle_index == -1:
        return
    triangle_vertices = points[tri.simplices[triangle_index]]
    draw_cell_weights(
        ax,
        triangle_vertices,
        ENGINE_COLORS["delaunay"],
        a_star,
        h_star,
        "Barycentric Weights",
    )


def draw_engine_slice(ax, k_idx, xx_endog, yy_endog, k_active, a_star) -> None:
    is_active = k_idx in k_active
    line_color = ENGINE_COLORS["engine"] if is_active else "lightgray"
    line_width = 2.5 if is_active else 1.0
    line_alpha = 0.9 if is_active else 0.3
    ax.plot(
        xx_endog[:, k_idx],
        yy_endog[:, k_idx],
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        label="Active $k$-slices" if k_idx == k_active[0] else "",
    )
    if not is_active:
        return
    h_interp = np.interp(a_star, xx_endog[:, k_idx], yy_endog[:, k_idx])
    ax.plot(
        a_star,
        h_interp,
        "o",
        color=ENGINE_COLORS["engine"],
        markersize=6,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Intermediate Points $\\hat{h}_k$" if k_idx == k_active[0] else "",
    )


def panel_engine(
    ax,
    xx_endog,
    yy_endog,
    J_grid,
    K_grid,
    a_star,
    h_star,
) -> None:
    ax.scatter(xx_endog, yy_endog, **_GRID_POINT_STYLE)
    ax.scatter(a_star, h_star, label="Query Point $(m^*, h^*)$", **_QUERY_POINT_STYLE)

    k_active = np.argsort(np.abs(yy_endog[J_grid // 2, :] - h_star))[:3]
    for k_idx in range(K_grid):
        draw_engine_slice(ax, k_idx, xx_endog, yy_endog, k_active, a_star)

    ax.axvline(
        a_star,
        color=ENGINE_COLORS["error"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Fixed $m^*$",
    )
    for j in range(J_grid):
        ax.plot(
            xx_endog[j, :],
            yy_endog[j, :],
            color=ENGINE_COLORS["neutral_light"],
            linewidth=0.8,
            alpha=0.3,
        )


def generate_partition_comparison():
    """Generate Figure X3: Conceptual Comparison of State Space Partitioning."""
    apply_engine_style()

    J_grid, K_grid = 6, 7
    _, _, xx_endog, yy_endog = get_shared_grid_data(
        J=J_grid,
        K=K_grid,
        warp_factor=0.15,
    )
    points = np.vstack((xx_endog.ravel(), yy_endog.ravel())).T
    a_star, h_star = 0.5, 0.45
    query_point = np.array([a_star, h_star])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    panel_white(axs[0], xx_endog, yy_endog, a_star, h_star)
    finalize_panel(axs[0], "(a)", "Curvilinear (White 2015)")

    panel_delaunay(axs[1], points, query_point, a_star, h_star)
    finalize_panel(axs[1], "(b)", "Delaunay (Ludwig and Schön 2018)")

    panel_engine(
        axs[2],
        xx_endog,
        yy_endog,
        J_grid,
        K_grid,
        a_star,
        h_star,
    )
    finalize_panel(axs[2], "(c)", "ENGINE (Sequential 1D)")

    plt.tight_layout()
    return save_engine_figure(fig, "partition_comparison.png")


if __name__ == "__main__":
    generate_partition_comparison()
