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
#     display_name: egmn-dev
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import numpy as np

# %% pycharm={"name": "#%%\n"}
from ConsLaborPortfolioModel import LaborPortfolioConsumerType
from HARK.interpolation import WarpedInterpOnInterp2D
from HARK.utilities import plot_funcs

# %% pycharm={"name": "#%%\n"}
agent = LaborPortfolioConsumerType()
agent.cycles = 1


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
    ax.set_xlabel("b")
    ax.set_ylabel(r"$\theta$")
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
plot_3d_func(agent.solution[0].labor_stage.labor_func, [0, 2.5], [0, 10])


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
grids["tShk"].shape

# %%
plt.scatter(grids["tShk"], grids["bNrm"], c=grids["leisure"])

# %%
labor_unconstrained_func = WarpedInterpOnInterp2D(
    grids["leisure"], [grids["bNrm"], grids["tShk"]]
)

# %%
labor_unconstrained_func([0, 1], [0, 1])

# %%
grids["bNrm"].shape, grids["tShk"].shape, grids["leisure"].shape

# %%
