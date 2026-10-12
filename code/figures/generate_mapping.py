"""Generate ENGINE interpolation visual figure.

This script creates a two-panel visualization of ENGINE's two-pass interpolation,
showing how the algorithm efficiently interpolates on curvilinear grids.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from figure_styles import (
    ENGINE_COLORS,
    apply_engine_style,
    get_shared_grid_data,
    panel_title,
    save_engine_figure,
)

# Marker style for the intermediate points of both ENGINE passes.
HIGHLIGHT_MARKERS = {
    "color": ENGINE_COLORS["highlight"],
    "linewidth": 2.5,
    "markersize": 9,
    "markeredgecolor": "black",
    "markeredgewidth": 1.2,
    "alpha": 0.9,
    "zorder": 10,
}


def pass1_intersections(xx_endog, yy_endog, K, x_query):
    """Return [(k, y_interp), ...] where each k-slice crosses ``x = x_query``."""
    intersections = []
    for k in range(K):
        x_slice = xx_endog[:, k]
        y_slice = yy_endog[:, k]
        if x_slice[0] <= x_query <= x_slice[-1]:
            intersections.append((k, np.interp(x_query, x_slice, y_slice)))
    return intersections


def draw_pass1_grid(ax, xx_endog, yy_endog, J, K) -> None:
    for k in range(K):
        ax.plot(
            xx_endog[:, k],
            yy_endog[:, k],
            color=ENGINE_COLORS["endog_lines_j"],
            linewidth=2.0,
            alpha=0.7,
            label="$k$-slices" if k == 0 else "",
        )
    for j in range(J):
        ax.plot(
            xx_endog[j, :],
            yy_endog[j, :],
            color=ENGINE_COLORS["neutral_light"],
            linewidth=1.0,
            alpha=0.5,
            label="$j$-lines" if j == 0 else "",
        )
    ax.scatter(
        xx_endog,
        yy_endog,
        c=ENGINE_COLORS["endog_grid"],
        s=25,
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )


def draw_pass1_intersections(ax, intersections, x_query) -> None:
    if not intersections:
        return
    ys = [y for _, y in intersections]
    ax.plot(
        [x_query] * len(ys),
        ys,
        "o-",
        **HIGHLIGHT_MARKERS,
        label=r"Intermediate $\hat{y}_k$",
    )


def schematic_f(x, y):
    """Smooth monotone surface used to illustrate Pass 2 function interpolation."""
    return -1.0 / (2.0 + x + y)


def draw_pass2_panel(ax, y_hat, f_hat, y_query, f_label) -> None:
    """Pass 2: the values ``f_hat`` (legend ``f_label``) at the intermediate
    ordinates ``y_hat``, one per crossed k-slice, and their linear interpolant
    at ``y = y_query``.
    """
    if len(y_hat) <= 1:
        return
    order = np.argsort(y_hat)
    y_sorted = y_hat[order]
    f_hat = f_hat[order]

    ax.plot(
        y_sorted,
        f_hat,
        "o-",
        **HIGHLIGHT_MARKERS,
        label=f_label,
    )
    ax.axvline(
        y_query,
        color=ENGINE_COLORS["white_method"],
        linestyle=":",
        linewidth=2.0,
        alpha=0.7,
        label="$y = y^*$",
    )

    f_interp = np.interp(y_query, y_sorted, f_hat)
    ax.scatter(
        y_query,
        f_interp,
        c="gold",
        s=200,
        marker="*",
        edgecolor="black",
        linewidth=1.5,
        zorder=15,
        label=r"Interpolated $f(x^*, y^*)$",
    )


def draw_engine_visual(
    xx_endog,
    yy_endog,
    query,
    f_on_slices,
    f_span,
    f_label=r"$f(x^*, \hat{y}_k)$",
):
    """Draw the two ENGINE passes for the query ``(x*, y*)`` on the mesh
    ``(xx_endog, yy_endog)`` and return the figure.

    ``f_on_slices(ks, y_hat)`` returns the Pass 1 values on the k-slices ``ks``
    that cross ``x = x*``, at their intermediate ordinates ``y_hat``; ``f_span``
    is the ``(low, high)`` value range that sets panel (b)'s vertical limits, and
    ``f_label`` is the legend entry of those values.
    """
    x_query, y_query = query
    J, K = xx_endog.shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a), Pass 1: interpolate along k-slices at x = x*
    ax = axes[0]
    draw_pass1_grid(ax, xx_endog, yy_endog, J, K)
    ax.axvline(
        x_query,
        color=ENGINE_COLORS["error"],
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label="$x = x^*$",
    )
    intersection_y = pass1_intersections(xx_endog, yy_endog, K, x_query)
    draw_pass1_intersections(ax, intersection_y, x_query)
    ax.axhline(
        y_query,
        color=ENGINE_COLORS["white_method"],
        linestyle=":",
        linewidth=2.0,
        alpha=0.7,
        label="$y = y^*$",
    )
    ax.scatter(
        x_query,
        y_query,
        c="gold",
        s=200,
        marker="*",
        edgecolor="black",
        linewidth=1.5,
        zorder=15,
        label=r"Query $(x^*, y^*)$",
    )
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    panel_title(ax, "(a)", "Pass 1: Find $\\hat{y}_k$ at $x = x^*$")
    # Top and bottom grid rows both run flat across the full x-range, so any
    # in-plot legend needs real headroom above y=1 to avoid sitting on top of
    # a grid node (framealpha used to mask this; frameon=False cannot).
    ax.set_xlim(xx_endog.min() - 0.05, xx_endog.max() + 0.1)
    ax.set_ylim(yy_endog.min() - 0.05, yy_endog.max() + 0.55)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel (b), Pass 2: interpolate f across the intermediate y-coordinates
    ax = axes[1]
    if intersection_y:
        ks, y_hat = (np.asarray(v) for v in zip(*intersection_y, strict=True))
        f_hat = np.asarray(f_on_slices(ks, y_hat))
        draw_pass2_panel(ax, y_hat, f_hat, y_query, f_label)
    ax.set_xlabel(r"$\hat{y}_k$ (intermediate coordinate)", fontsize=12)
    ax.set_ylabel("$f$", fontsize=12)
    panel_title(
        ax,
        "(b)",
        r"Pass 2: Interpolate $f$ across $\{\hat{y}_k\}$ at $y = y^*$",
    )
    ax.set_xlim(yy_endog.min() - 0.05, yy_endog.max() + 0.05)
    f_lo, f_hi = f_span
    f_range = f_hi - f_lo
    ax.set_ylim(f_lo - 0.1 * f_range, f_hi + 0.35 * f_range)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def schematic_mesh_and_query():
    """The paper figure's ``7 x 8`` mesh and its query ``(x*, y*)``, the center
    of the mesh's bounding box.
    """
    _, _, xx_endog, yy_endog = get_shared_grid_data(7, 8, warp_factor=0.15)
    x_query = (xx_endog.min() + xx_endog.max()) / 2
    y_query = (yy_endog.min() + yy_endog.max()) / 2
    return xx_endog, yy_endog, (x_query, y_query)


def generate_mapping():
    """Generate ENGINE two-pass interpolation visualization."""
    apply_engine_style()
    xx_endog, yy_endog, query = schematic_mesh_and_query()
    x_query, y_query = query
    inside_x = xx_endog.min() < x_query < xx_endog.max()
    if not (inside_x and yy_endog.min() < y_query < yy_endog.max()):
        msg = f"schematic query ({x_query}, {y_query}) lies outside the mesh"
        raise RuntimeError(msg)

    f_hat_all = schematic_f(x_query, yy_endog)
    fig = draw_engine_visual(
        xx_endog,
        yy_endog,
        query,
        lambda _ks, y_hat: schematic_f(x_query, y_hat),
        (f_hat_all.min(), f_hat_all.max()),
    )
    return save_engine_figure(fig, "engine_interpolation_visual.png")


if __name__ == "__main__":
    path = generate_mapping()
