import matplotlib.pyplot as plt
import numpy as np


def plot_hark_bilinear(function):
    x_list = function.x_list
    f_vals = function.f_values

    plt.plot(x_list, f_vals)


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


def plot_3d_func(func, min, max, n=100):
    # get_ipython().run_line_magic("matplotlib", "widget")
    xgrid = np.linspace(min, max, n)
    ygrid = xgrid

    xMat, yMat = np.meshgrid(xgrid, ygrid, indexing="ij")

    zMat = func(xMat, yMat)

    ax = plt.axes(projection="3d")
    ax.plot_surface(xMat, yMat, zMat, cmap="viridis")
    ax.set_title("surface")
    ax.set_xlabel("m")
    ax.set_ylabel("n")
    ax.set_zlabel("f")
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


def plot_3d_func(func, lims_x, lims_y, n=100, label_x="x", label_y="y", label_z="z"):
    # get_ipython().run_line_magic("matplotlib", "widget")
    xmin, xmax = lims_x
    ymin, ymax = lims_y
    xgrid = np.linspace(xmin, xmax, n)
    ygrid = np.linspace(ymin, ymax, n)

    xMat, yMat = np.meshgrid(xgrid, ygrid, indexing="ij")

    zMat = func(xMat, yMat)

    ax = plt.axes(projection="3d")
    ax.plot_surface(xMat, yMat, zMat, cmap="viridis")
    ax.set_title("surface")
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)
    plt.show()


def plot_retired(min, max, n=100):
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


def plot_3d_func(func, min, max, n=100):
    xgrid = np.linspace(min, max, n)
    ygrid = xgrid

    xmat, ymat = np.meshgrid(xgrid, ygrid, indexing="ij")

    zmat = func(xmat, ymat)

    ax = plt.axes(projection="3d")
    ax.plot_surface(xmat, ymat, zmat, cmap="viridis")
    ax.set_title("surface")
    ax.set_xlabel("m")
    ax.set_ylabel("n")
    ax.set_zlabel("f")
    plt.show()


def scatter_hist(x, y, color, ax, ax_histx, ax_histy, fig):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    plot = ax.scatter(x, y, c=color, cmap="jet")
    fig.colorbar(plot)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(x), np.max(y))
    xymin = min(np.min(x), np.min(y))
    top = (int(xymax / binwidth) + 1) * binwidth
    bottom = (int(xymin / binwidth) + 1) * binwidth

    bins = np.arange(bottom, top + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation="horizontal")


def plot_3d_func(func, min, max, n=100):
    get_ipython().run_line_magic("matplotlib", "widget")
    xgrid = np.linspace(min, max, n)
    ygrid = xgrid

    xmat, ymat = np.meshgrid(xgrid, ygrid, indexing="ij")

    zmat = func(xmat, ymat)

    ax = plt.axes(projection="3d")
    ax.plot_surface(xmat, ymat, zmat, cmap="viridis")
    ax.set_title("surface")
    ax.set_xlabel("m")
    ax.set_ylabel("n")
    ax.set_zlabel("f")
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
    idx = np.logical_and.reduce([idx, x > 0, y > 0])

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
    fig.savefig(figures_path + filename)
