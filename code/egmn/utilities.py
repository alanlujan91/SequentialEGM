from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from HARK.interpolation import LinearInterp, LinearInterpOnInterp1D
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})

figures_path = "../../content/figures/"


def plot_hark_bilinear(function, meta_data={}):
    x_list = np.array(function.x_list)
    f_vals = np.array(function.f_values)

    default_data = {
        "xlabel": "X-axis label",
        "ylabel": "Y-axis label",
        "title": "Title of the plot",
    }
    default_data.update(meta_data)

    plt.plot(x_list, f_vals)
    plt.xlabel(default_data["xlabel"])
    plt.ylabel(default_data["ylabel"])
    plt.title(default_data["title"])


def plot_warped_bilinear_flat(function, min_x=None, max_x=None, n=100):
    x_grid = function.grids[0]
    y_grid = function.grids[1]
    values = function.values

    if min_x is not None and max_x is not None:
        x_vals = np.linspace(min_x, max_x, n)
        x_grid, y_grid = np.meshgrid(x_vals, y_grid[0], indexing="ij")
        values = function(x_grid, y_grid)

    for i in range(len(y_grid[0])):
        plt.plot(x_grid[:, i], values[:, i])


def plot_3d_func(func, xlims, ylims, n=100, meta={}, savename=None):
    xgrid = np.linspace(xlims[0], xlims[1], n)
    ygrid = np.linspace(ylims[0], ylims[1], n)

    xMat, yMat = np.meshgrid(xgrid, ygrid, indexing="ij")

    zMat = func(xMat, yMat)

    meta_data = {"title": "surface", "xlabel": "x", "ylabel": "y", "zlabel": "function"}

    meta_data.update(meta)

    ax = plt.axes(projection="3d")
    ax.plot_surface(xMat, yMat, zMat, cmap="viridis")
    ax.set_title(meta_data["title"])
    ax.set_xlabel(meta_data["xlabel"])
    ax.set_ylabel(meta_data["ylabel"])
    ax.set_zlabel(meta_data["zlabel"])

    if savename is not None:
        plt.savefig(figures_path + savename + ".svg")
        plt.savefig(figures_path + savename + ".pdf")

    plt.show()


def plot_retired(
    min,
    max,
    cFunc_retired,
    vPFunc_retired,
    vFunc_retired,
    n=100,
):
    plt.figure(figsize=(15, 6))

    mgrid = np.linspace(min, max, n)

    plt.subplot(1, 3, 1)
    plt.title("consumption")
    for cFunc in cFunc_retired:
        plt.plot(mgrid, cFunc(mgrid))

    plt.subplot(1, 3, 2)
    plt.title("inverse vPfunc")
    for vPfunc in vPFunc_retired:
        plt.plot(mgrid, vPfunc.cFunc(mgrid))

    plt.subplot(1, 3, 3)
    plt.title("inverse vFunc")
    for vFunc in vFunc_retired:
        plt.plot(mgrid, vFunc.vFuncNvrs(mgrid))

    plt.show()


def scatter_hist(x, y, color, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    hist = ax.scatter(x, y, c=color, cmap="viridis", alpha=0.6)

    # now determine nice limits by hand:
    binwidth = 1.5
    xymax = max(np.max(x), np.max(y))
    xymin = min(np.min(x), np.min(y))
    top = (int(xymax / binwidth) + 1) * binwidth
    bottom = (int(xymin / binwidth) + 1) * binwidth

    bins = np.arange(bottom, top + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation="horizontal")

    return hist


def plot_scatter_hist(x, y, color, title, xlabel, ylabel, filename):
    # Create a Figure, which doesn't have to be square.
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    # Create the main axes, leaving 25% of the figure space at the top and on the
    # right to position marginals.
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    # The main axes' aspect can be fixed.
    ax.set(aspect=1)
    # Create marginal axes, which have 25% of the size of the main axes.  Note that
    # the inset axes are positioned *outside* (on the right and the top) of the
    # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
    # less than 0 would likewise specify positions on the left and the bottom of
    # the main axes.
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    # remove non-finite values
    idx = np.logical_and.reduce([np.isfinite(x), np.isfinite(y), np.isfinite(color)])
    idx = np.logical_and.reduce([idx, x > 0, y > 0, x < 12, y < 15])

    x = x[idx]
    y = y[idx]
    color = color[idx]

    # Draw the scatter plot and marginals.
    hist = scatter_hist(x, y, color, ax, ax_histx, ax_histy)
    cbar = fig.colorbar(hist)
    cbar.ax.set_ylabel("Pension Deposits $d$")

    fig.suptitle(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()
    fig.savefig(figures_path + filename + ".svg")
    fig.savefig(figures_path + filename + ".pdf")


def interp_on_interp(values, grids):
    temp = []
    x, y = grids
    grid = y[0]
    for i in range(grid.size):
        temp.append(LinearInterp(x[:, i], values[:, i]))

    return LinearInterpOnInterp1D(temp, grid)
