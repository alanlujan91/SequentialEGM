# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: egmn-dev
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt

# %%
from ConsLaborSeparableModel import LaborSeparableConsumerType
from HARK.utilities import plot_funcs
from utilities import plot_3d_func

figures_path = "../../content/figures/"

# %%
agent = LaborSeparableConsumerType(aXtraNestFac=-1, aXtraCount=25, cycles=10)

# %%
grids = agent.solution_terminal.terminal_grids

# %%
plt.scatter(grids["bnrm"], grids["tshk"], c=grids["labor"], norm="symlog")
plt.colorbar()
plt.ylim([0.85, 1.2])
plt.savefig(figures_path + "LaborSeparableWarpedGrid.svg")
plt.savefig(figures_path + "LaborSeparableWarpedGrid.pdf")

# %%
plt.scatter(grids["mnrm"], grids["tshk"], c=grids["labor"], norm="symlog")
plt.colorbar()
plt.ylim([0.85, 1.2])
plt.savefig(figures_path + "LaborSeparableRectangularGrid.svg")
plt.savefig(figures_path + "LaborSeparableRectangularGrid.pdf")

# %%
plot_funcs(agent.solution_terminal.labor_leisure.c_func.xInterpolators, 0, 5)

# %%
plot_3d_func(
    agent.solution_terminal.labor_leisure.labor_func,
    [0, 5],
    [0.85, 1.2],
    meta={
        "title": "Labor function",
        "xlabel": "Bank Balances",
        "ylabel": "Transitory shock",
        "zlabel": "Labor",
    },
)

# %%
plot_3d_func(
    agent.solution_terminal.labor_leisure.v_func,
    [0, 5],
    [0.85, 1.2],
    meta={
        "title": "Value function",
        "zlabel": "Value",
        "xlabel": "Bank Balances",
        "ylabel": "Transitory Income",
    },
)

# %%
agent.solve()

# %%
plot_3d_func(
    agent.solution[0].labor_leisure.labor_func,
    [0, 15],
    [0.85, 1.2],
    meta={
        "title": "Labor function",
        "xlabel": "Bank Balances",
        "ylabel": "Transitory shock",
        "zlabel": "Labor",
    },
)

# %%
