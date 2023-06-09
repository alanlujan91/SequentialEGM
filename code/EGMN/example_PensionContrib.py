# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
import numpy as np
from ConsPensionContribModel import PensionContribConsumerType
from IPython import get_ipython

figures_path = "../../Figures/"

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent = PensionContribConsumerType(cycles=19)


# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
def plot_bilinear(function):
    x_list = function.x_list
    f_vals = function.f_values

    plt.plot(x_list, f_vals)


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


# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent.solve()

T = 0

# %% [markdown]
# ## Post Decision Stage

# %%
plot_3d_func(agent.solution[T].post_decision_stage.v_func.vFuncNvrs, 0, 5)

# %%
plot_3d_func(agent.solution[T].post_decision_stage.dvda_func.cFunc, 0, 5)

# %%
plot_3d_func(agent.solution[T].post_decision_stage.dvdb_func.cFunc, 0, 5)

# %% [markdown]
# ## Consumption Stage

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].consumption_stage.c_func, 0, 5)

# %%
plot_3d_func(agent.solution[T].consumption_stage.v_func.vFuncNvrs, 0, 5)

# %%
plot_3d_func(agent.solution[T].consumption_stage.dvdl_func.cFunc, 0, 5)

# %%
plot_3d_func(agent.solution[T].consumption_stage.dvdb_func.cFunc, 0, 5)

# %% [markdown]
# ## Deposit Stage

# %%
plot_3d_func(agent.solution[T].deposit_stage.d_func, 0, 5)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].deposit_stage.c_func, 0, 5)

# %%
plot_3d_func(agent.solution[T].deposit_stage.v_func.vFuncNvrs, 0, 5)

# %%
plot_3d_func(agent.solution[T].deposit_stage.dvdm_func.cFunc, 0, 5)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].deposit_stage.dvdn_func.cFunc, 0, 5)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %time
plot_3d_func(agent.solution[T].deposit_stage.gaussian_interp, 0, 5)


# %% [markdown]
# ## Grids


# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
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


# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

x = agent.solution[T].deposit_stage.gaussian_interp.grids[0]
y = agent.solution[T].deposit_stage.gaussian_interp.grids[1]
color = agent.solution[T].deposit_stage.gaussian_interp.values

idx = np.logical_or(x < 0, y < 0)

x = x[~idx]
y = y[~idx]
color = color[~idx]

# use the previously defined function
hist = scatter_hist(x, y, color, ax, ax_histx, ax_histy)
cbar = fig.colorbar(hist)
cbar.ax.set_ylabel("Pension Deposits $d$")

fig.suptitle("Pension Deposit on Endogenous Grid", fontsize=16)
ax.set_xlabel("Market Resources $m$")
ax.set_ylabel("Retirement Savings $n$")
plt.show()
fig.savefig(figures_path + "EndogenousGrid.svg")

# %%
grids = agent.solution[T].consumption_stage.grids_before_cleanup

# %%
fig, ax = plt.subplots()
plot = ax.scatter(
    grids["mMat"],
    grids["nMat"],
    c=grids["dMat"],
    cmap="viridis",
    vmin=-1,
    vmax=5,
    plotnonfinite=True,
    alpha=0.6,
)
cbar = fig.colorbar(plot)
cbar.ax.set_ylabel("Pension Deposits $d$")

plt.xlim([-1, 10])
plt.ylim([-1, 10])

# %%
fig, ax = plt.subplots()
scatter = ax.scatter(
    grids["lMat"],
    grids["b2Mat"],
    c=np.maximum(grids["dMat"], 0),
    cmap="viridis",
    vmin=-2,
    vmax=15,
    plotnonfinite=True,
    alpha=0.6,
)
cbar = fig.colorbar(scatter)
cbar.ax.set_ylabel("Pension Deposits $d$")

plt.title("Pension Deposits on Exogenous Post-Decision Grid")
plt.xlabel(r"Liquid Wealth $\ell$")
plt.ylabel("Retirement Balance $b$")
fig.savefig(figures_path + "ExogenousGrid.svg")

# %%
grids = agent.solution[T].consumption_stage.grids_before_cleanup

# %%
from HARK.interpolation._sklearn import GeneralizedRegressionUnstructuredInterp

# %%
gauss_interp = GeneralizedRegressionUnstructuredInterp(
    grids["dMat"],
    [grids["mMat"], grids["nMat"]],
    model="gaussian-process",
    std=True,
    model_kwargs={"normalize_y": True},
)

# %%
get_ipython().run_line_magic("matplotlib", "widget")
plot_3d_func(gauss_interp, 0, 5)

# %%
