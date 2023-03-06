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

# %%
import matplotlib.pyplot as plt
import numpy as np

# %% pycharm={"name": "#%%\n"}
from ConsLaborPortfolioModel import LaborPortfolioConsumerType
from HARK.utilities import plot_funcs

# %% pycharm={"name": "#%%\n"}
agent = LaborPortfolioConsumerType()
agent.cycles = 10


# %%
def plot_3d_func(func, lims_x, lims_y, n=100):
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
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("b")
    ax.set_zlabel("f")
    plt.show()


# %% pycharm={"name": "#%%\n"}
consumption_stage = agent.solution_terminal.consumption_stage


# %%
consumption_stage


# %% pycharm={"name": "#%%\n"}
plot_funcs(consumption_stage.c_func, 0, 5)


# %% pycharm={"name": "#%%\n"}
agent.solve()


# %% pycharm={"name": "#%%\n"}
plot_funcs(agent.solution[0].portfolio_stage.share_func, 0, 10)


# %% pycharm={"name": "#%%\n"}
plot_funcs(agent.solution[0].consumption_stage.c_func, 0, 10)


# %% pycharm={"name": "#%%\n"}
plot_3d_func(
    agent.solution[0].labor_stage.labor_func,
    [min(agent.TranShkGrid), max(agent.TranShkGrid)],
    [0, 10],
)


# %% pycharm={"name": "#%%\n"}
grids = agent.solution[0].labor_stage.grids


# %%
ax = plt.axes(projection="3d")
ax.plot_surface(grids["bNrm"], grids["tShk"], grids["leisure"], cmap="viridis")
ax.set_title("surface")
ax.set_xlabel("b")
ax.set_ylabel(r"$\theta$")
ax.set_zlabel("f")


# %%
plot_funcs(agent.solution[0].labor_stage.labor_func.xInterpolators, 0, 10)

# %%
xInterps = agent.solution[0].labor_stage.labor_func.xInterpolators
y = agent.TranShkGrid[0]

for i in range(len(xInterps)):
    x_list = np.array(xInterps[i].x_list).ravel()
    c_list = np.array(xInterps[i].y_list).ravel()
    y_list = np.array([y[i]] * len(x_list)).ravel()
    print(len(x_list), len(y_list), len(c_list))
    plt.scatter(x_list, y_list, c=c_list, cmap="viridis")

# %%
