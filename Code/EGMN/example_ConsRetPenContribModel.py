# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: egmn-dev
#     language: python
#     name: python3
# ---

# %% pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
import numpy as np
from ConsRetPenContribModel import PensionContribConsumerType

# %% pycharm={"name": "#%%\n"}
agent = PensionContribConsumerType()
agent.solve()


# %% pycharm={"name": "#%%\n"}
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


# %% pycharm={"name": "#%%\n"}
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


# %% pycharm={"name": "#%%\n"}
solution = agent.solution
retired_solution = [solution[t].retired_solution for t in range(agent.T_age)]
worker_solution = [solution[t].worker_solution for t in range(agent.T_age)]
working_solution = [solution[t].working_solution for t in range(agent.T_age)]
retiring_solution = [solution[t].retiring_solution for t in range(agent.T_age)]

# Retired

cFunc_retired = [retired_solution[t].cFunc for t in range(agent.T_age)]
vPFunc_retired = [retired_solution[t].vPfunc for t in range(agent.T_age)]
vFunc_retired = [retired_solution[t].vFunc for t in range(agent.T_age)]

# Worker

cFunc_worker = [worker_solution[t].deposit_stage.cFunc for t in range(agent.T_age)]
dFunc_worker = [worker_solution[t].deposit_stage.dFunc for t in range(agent.T_age)]
dvdmFunc_worker = [
    worker_solution[t].deposit_stage.dvdmFunc for t in range(agent.T_age)
]
dvdnFunc_worker = [
    worker_solution[t].deposit_stage.dvdnFunc for t in range(agent.T_age)
]
vFunc_worker = [worker_solution[t].deposit_stage.vFunc for t in range(agent.T_age)]
prbWrkFunc_worker = [
    worker_solution[t].probabilities.prob_working for t in range(agent.T_age)
]
prbRetFunc_worker = [
    worker_solution[t].probabilities.prob_retiring for t in range(agent.T_age)
]

# Working

# End of Period

dvdaEndOfPrdFunc_working = [
    working_solution[t].post_decision_stage.dvdaFunc for t in range(agent.T_age)
]
dvdbEndOfPrdFunc_working = [
    working_solution[t].post_decision_stage.dvdbFunc for t in range(agent.T_age)
]
vEndOfPrdFunc_working = [
    working_solution[t].post_decision_stage.vFunc for t in range(agent.T_age)
]

# Inner Loop

cInnerFunc_working = [
    working_solution[t].consumption_stage.cFunc for t in range(agent.T_age)
]
dvdlInnerFunc_working = [
    working_solution[t].consumption_stage.dvdlFunc for t in range(agent.T_age)
]
dvdbInnerFunc_working = [
    working_solution[t].consumption_stage.dvdbFunc for t in range(agent.T_age)
]
vInnerFunc_working = [
    working_solution[t].consumption_stage.vFunc for t in range(agent.T_age)
]

# Outer Loop

cFunc_working = [working_solution[t].deposit_stage.cFunc for t in range(agent.T_age)]
dFunc_working = [working_solution[t].deposit_stage.dFunc for t in range(agent.T_age)]
dvdmFunc_working = [
    working_solution[t].deposit_stage.dvdmFunc for t in range(agent.T_age)
]
dvdnFunc_working = [
    working_solution[t].deposit_stage.dvdnFunc for t in range(agent.T_age)
]
vFunc_working = [working_solution[t].deposit_stage.vFunc for t in range(agent.T_age)]

# Retiring

cFunc_retiring = [retiring_solution[t].cFunc for t in range(agent.T_age)]
vPfunc_retiring = [retiring_solution[t].vPfunc for t in range(agent.T_age)]
vPfunc_retiring = [retiring_solution[t].vPfunc for t in range(agent.T_age)]
vFunc_retiring = [retiring_solution[t].vFunc for t in range(agent.T_age)]

# %% pycharm={"name": "#%%\n"}
plot_retired(0, 10)


# %% pycharm={"name": "#%%\n"}
t = 0


# %% [markdown] pycharm={"name": "#%% md\n"}
# # Working
#

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Post Decision Stage
#

# %% pycharm={"name": "#%%\n"}
# wa
plot_3d_func(dvdaEndOfPrdFunc_working[t].cFunc, 0.0, 10)


# %% pycharm={"name": "#%%\n"}
# wb
plot_3d_func(dvdbEndOfPrdFunc_working[t].cFunc, 0.0, 10)


# %% pycharm={"name": "#%%\n"}
# w
plot_3d_func(vEndOfPrdFunc_working[t].vFuncNvrs, 0.0, 10)


# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Consumption Stage
#

# %% pycharm={"name": "#%%\n"}
plot_3d_func(cInnerFunc_working[t], 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(dvdlInnerFunc_working[t].cFunc, 0.1, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(dvdbInnerFunc_working[t].cFunc, 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(vInnerFunc_working[t].vFuncNvrs, 0.0, 10)


# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Deposit Stage
#

# %% pycharm={"name": "#%%\n"}
plot_3d_func(cFunc_working[t], 0.1, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(dFunc_working[t], 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(dvdmFunc_working[t].cFunc, 0.1, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(dvdnFunc_working[t].cFunc, 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(vFunc_working[t].vFuncNvrs, 0.1, 10)


# %% [markdown] pycharm={"name": "#%% md\n"}
# # Retiring
#

# %% pycharm={"name": "#%%\n"}
plot_3d_func(cFunc_retiring[t], 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(vPfunc_retiring[t].cFunc, 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(vFunc_retiring[t].vFuncNvrs, 0.0, 10)


# %% [markdown] pycharm={"name": "#%% md\n"}
# # Worker
#

# %% pycharm={"name": "#%%\n"}
plot_3d_func(cFunc_worker[t], 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(dFunc_worker[t], 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(dvdmFunc_worker[t].cFunc, 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(dvdnFunc_worker[t].cFunc, 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(vFunc_worker[t].vFuncNvrs, 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(prbWrkFunc_worker[t], 0.0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(prbRetFunc_worker[t], 0.0, 10)


# %% [markdown] pycharm={"name": "#%% md\n"}
#

# %% pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[0].working_solution.deposit_stage.linear_interp, 0, 10)


# %% pycharm={"name": "#%%\n"}
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


# %% pycharm={"name": "#%%\n"}
# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

x = agent.solution[0].working_solution.deposit_stage.linear_interp.points[:, 0]
y = agent.solution[0].working_solution.deposit_stage.linear_interp.points[:, 1]
color = agent.solution[0].working_solution.deposit_stage.linear_interp.values

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

# use the previously defined function
scatter_hist(x, y, color, ax, ax_histx, ax_histy, fig)
