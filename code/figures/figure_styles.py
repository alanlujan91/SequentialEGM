"""Shared styling configuration for ENGINE paper figures.

One publication style for every ENGINE paper figure, following journal
standards.

Design Principles:
- Clean, minimal aesthetics suitable for print and digital
- Colorblind-friendly palette with sufficient contrast
- Appropriate font sizes for journal reproduction (typically 2-column format)
- Consistent line weights and marker styles across all figures
- No decorative elements that distract from data
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from repo_paths import FIG_DIR

# Shared output directory for all paper figures, as a string for os.path callers.
FIGURE_DIR = os.fspath(FIG_DIR)

# Journal figure settings, sized for a two-column format.
ENGINE_STYLE = {
    # Figure dimensions and resolution
    "figure.figsize": (7, 5),  # Single column width typical in journals
    "figure.dpi": 150,
    "savefig.dpi": 300,  # High resolution for publication
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "savefig.format": "pdf",  # Vector format preferred
    # Typography - using standard LaTeX-compatible fonts
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    # Mathematical text
    "mathtext.fontset": "cm",  # Computer Modern for consistency
    "text.usetex": False,  # Set True if LaTeX available
    # Legend styling - clean and minimal
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.shadow": False,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    # Axes styling
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(
        color=[
            "#2166AC",  # Muted blue
            "#B2182B",  # Muted red
            "#1B7837",  # Muted green
            "#762A83",  # Muted purple
            "#666666",  # Gray
            "#D6604D",  # Muted coral
            "#4393C3",  # Light blue
            "#92C5DE",  # Pale blue
        ],
    ),
    # Grid styling
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "grid.color": "#CCCCCC",
    # Lines and markers
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "lines.markeredgewidth": 0.8,
    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}

# Muted, colorblind-friendly palette: ColorBrewer qualitative colors at
# reduced saturation.
ENGINE_COLORS = {
    # Primary method colors (muted, professional)
    "engine": "#2166AC",  # Muted blue - ENGINE (primary)
    "white_method": "#B2182B",  # Muted red - HARK baseline (sector-walking)
    "delaunay": "#1B7837",  # Muted green - Delaunay triangulation
    "scipy": "#762A83",  # Muted purple - SciPy baseline
    "hark": "#B2182B",  # Same as white_method (HARK baseline)
    # Grid visualization colors (muted)
    "exog_grid": "#2166AC",  # Muted blue for exogenous grid points
    "endog_grid": "#B2182B",  # Muted red for endogenous grid points
    "exog_lines": "#666666",  # Gray for exogenous grid lines
    "endog_lines_j": "0.45",  # Neutral gray, outside viridis hue range
    "endog_lines_k": "#E66100",  # Warm orange, outside viridis hue range
    # Panel/background colors (very light, subtle)
    "input_bg": "#F7F7F7",  # Very light gray
    "pass1_bg": "#E5F5E0",  # Very light green
    "pass2_bg": "#FEE0D2",  # Very light red
    "output_bg": "#F7F7F7",  # Very light gray
    "complexity_bg": "#F7F7F7",  # Very light gray
    # Performance/results colors
    "speedup": "#1B7837",  # Muted green for positive results
    "error": "#B2182B",  # Muted red for errors
    # Neutral colors
    "neutral_dark": "#333333",
    "neutral_medium": "#666666",
    "neutral_light": "#999999",
    "highlight": "#F4A582",  # Muted coral for emphasis (not gold)
}

# Helper functions.


def apply_engine_style() -> None:
    """Apply ENGINE paper styling to matplotlib.

    Call this at the start of each figure generation script to ensure
    consistent styling across all figures.
    """
    plt.style.use("default")  # Start from clean state
    plt.rcParams.update(ENGINE_STYLE)


def save_engine_figure(fig, filename, figure_dir=FIGURE_DIR, formats=("png", "pdf")):
    """Save figure with ENGINE paper standards, then close it.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save; closed once every format is written.
    filename : str
        Base filename (without extension).
    figure_dir : str
        Directory to save figures in (default ``FIGURE_DIR``).
    formats : tuple
        File formats to save. Default saves both PNG and PDF.

    Returns
    -------
    str
        Path to the primary saved file (PNG).

    """
    # Ensure directory exists
    os.makedirs(figure_dir, exist_ok=True)

    # Remove extension if provided
    base_name = os.path.splitext(filename)[0]

    primary_path = None
    for fmt in formats:
        filepath = os.path.join(figure_dir, f"{base_name}.{fmt}")
        fig.savefig(
            filepath,
            dpi=ENGINE_STYLE["savefig.dpi"] if fmt == "png" else None,
            bbox_inches=ENGINE_STYLE["savefig.bbox"],
            pad_inches=ENGINE_STYLE["savefig.pad_inches"],
            facecolor="white",
            edgecolor="none",
            format=fmt,
            # No creation stamp, so an unchanged figure regenerates byte-identical.
            metadata={"CreationDate": None} if fmt == "pdf" else None,
        )
        if fmt == "png":
            primary_path = filepath
    plt.close(fig)
    return primary_path or filepath


def format_axes_for_publication(
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    legend=True,
    grid=True,
    despine=True,
):
    """Format axes with publication-quality styling.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to format.
    title : str, optional
        Axes title.
    xlabel, ylabel : str, optional
        Axis labels.
    legend : bool
        Whether to show legend (if handles exist).
    grid : bool
        Whether to show grid.
    despine : bool
        Whether to remove top and right spines.

    Returns
    -------
    Axes
        The formatted axes.

    """
    if title:
        ax.set_title(
            title,
            fontsize=ENGINE_STYLE["axes.titlesize"],
            fontweight="normal",
            pad=8,
        )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=ENGINE_STYLE["axes.labelsize"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ENGINE_STYLE["axes.labelsize"])

    if grid:
        ax.grid(
            True,
            alpha=ENGINE_STYLE["grid.alpha"],
            linewidth=ENGINE_STYLE["grid.linewidth"],
        )
    else:
        ax.grid(False)

    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Only create a legend where none exists: clobbering an existing legend
    # would discard caller-chosen loc/frameon styling.
    if legend and ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")

    return ax


def get_shared_grid_data(J=6, K=7, warp_factor=0.15):
    """Generate consistent grid data for schematic figures.

    Parameters
    ----------
    J, K : int
        Grid dimensions.
    warp_factor : float
        Amount of grid warping (0 = rectangular).
        Default is 0.15 for most figures. Different figures may use
        different warping values for illustrative purposes to emphasize
        specific features (e.g., the grid-homeomorphism figure uses 0.18 for stronger curvature).

    Returns
    -------
    xx_exog, yy_exog : ndarray
        Exogenous (regular) grid coordinates.
    xx_endog, yy_endog : ndarray
        Endogenous (warped) grid coordinates.

    """
    x_exog_nodes = np.linspace(0, 1, J)
    y_exog_nodes = np.linspace(0, 1, K)
    xx_exog, yy_exog = np.meshgrid(x_exog_nodes, y_exog_nodes, indexing="ij")

    # Consistent warping transformation
    xx_endog = xx_exog + warp_factor * np.sin(np.pi * yy_exog) * np.cos(np.pi * xx_exog)
    yy_endog = (
        yy_exog + warp_factor * np.cos(np.pi * xx_exog) * np.sin(np.pi * yy_exog) * 0.8
    )

    return xx_exog, yy_exog, xx_endog, yy_endog


def panel_title(ax, label, title) -> None:
    """Title the panel ``"(a) title"``; the label is part of the header."""
    ax.set_title(f"{label} {title}", fontsize=11, fontweight="normal")


def label_panel(ax, label, title, xlabel, ylabel) -> None:
    """Axis labels at fontsize 11 plus the ``"(a) title"`` header of one panel."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    panel_title(ax, label, title)


def exogenous_panel(ax, x, y, title, xlabel, ylabel) -> None:
    """Panel (a) of a grid-transformation figure: the exogenous mesh, unruled."""
    plot_exogenous_mesh(ax, x, y)
    label_panel(ax, "(a)", title, xlabel, ylabel)
    ax.grid(False)


def plot_exogenous_mesh(ax, x, y) -> None:
    """Grey index lines and blue nodes of a regular exogenous grid."""
    grey = ENGINE_COLORS["neutral_medium"]
    plot_mesh_lines(ax, x, y, colors=(grey, grey), linewidth=0.8, alpha=0.6)
    ax.scatter(
        x,
        y,
        c=ENGINE_COLORS["exog_grid"],
        s=20,
        edgecolor="black",
        linewidth=0.4,
        zorder=5,
    )


def add_right_colorbar(ax, mappable, label) -> None:
    """Colorbar for ``mappable`` in a slim axis along ``ax``'s right edge."""
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
    plt.colorbar(mappable, cax=cax).set_label(label, fontsize=10)


def scatter_with_colorbar(ax, x, y, c, label, s=20):
    """Scatter (x, y) colored by ``c`` (viridis) with a right-side colorbar:
    the endogenous-mesh scatter of the grid-transformation figures.
    """
    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap="viridis",
        s=s,
        edgecolor="none",
        alpha=0.9,
        zorder=5,
    )
    add_right_colorbar(ax, sc, label)
    return sc


def add_mesh_legend(ax, label_j, label_k) -> None:
    """Frameless legend keying the two endogenous mesh-line families."""
    handles = [
        Line2D([0], [0], color=ENGINE_COLORS[key], linewidth=1.0, alpha=0.7, label=lab)
        for key, lab in (("endog_lines_j", label_j), ("endog_lines_k", label_k))
    ]
    ax.legend(
        handles=handles,
        loc="best",
        framealpha=0,
        edgecolor="none",
        handlelength=1.5,
        fontsize=9,
    )


def plot_mesh_lines(
    ax,
    x,
    y,
    z=None,
    colors=None,
    labels=(None, None),
    step=(1, 1),
    mask=None,
    **kw,
) -> None:
    """Both index-line families of a tensor-indexed mesh ``x[j, k], y[j, k]`` (with
    ``z`` on a 3D axis): the constant-``k`` columns, then the constant-``j`` rows,
    every ``step = (j_step, k_step)``-th line. Each family takes its color from
    ``colors`` (default ``endog_lines_j``, ``endog_lines_k``) and its ``labels``
    entry on its first line only. Points outside ``mask`` are dropped, and a line
    left empty is skipped.
    """
    colors = colors or (ENGINE_COLORS["endog_lines_j"], ENGINE_COLORS["endog_lines_k"])
    coords = (x, y) if z is None else (x, y, z)

    def draw(line, color, label) -> None:
        keep = slice(None) if mask is None else mask[line]
        if mask is not None and not np.any(keep):
            return
        ax.plot(*(c[line][keep] for c in coords), color=color, label=label, **kw)

    for k in range(0, x.shape[1], step[1]):
        draw((slice(None), k), colors[0], labels[0] if k == 0 else None)
    for j in range(0, x.shape[0], step[0]):
        draw((j, slice(None)), colors[1], labels[1] if j == 0 else None)
