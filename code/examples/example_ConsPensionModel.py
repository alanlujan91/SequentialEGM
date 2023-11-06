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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
from ConsPensionModel import PensionConsumerType
from utilities import plot_3d_func, plot_scatter_hist

figures_path = "../../content/figures/"

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent = PensionConsumerType(cycles=19)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent.solve()

T = 0

# %% [markdown]
# ## Post Decision Stage

# %%
plot_3d_func(agent.solution[T].post_decision_stage.v_func.vFuncNvrs, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].post_decision_stage.dvda_func.cFunc, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].post_decision_stage.dvdb_func.cFunc, [0, 5], [0, 5])

# %% [markdown]
# ## Consumption Stage

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].consumption_stage.c_func, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].consumption_stage.v_func.vFuncNvrs, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].consumption_stage.dvdl_func.cFunc, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].consumption_stage.dvdb_func.cFunc, [0, 5], [0, 5])

# %% [markdown]
# ## Deposit Stage

# %%
plot_3d_func(agent.solution[T].deposit_stage.d_func, [0, 5], [0, 5])

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].deposit_stage.c_func, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].deposit_stage.v_func.vFuncNvrs, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].deposit_stage.dvdm_func.cFunc, [0, 5], [0, 5])

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].deposit_stage.dvdn_func.cFunc, [0, 5], [0, 5])

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %time
plot_3d_func(agent.solution[T].deposit_stage.gaussian_interp, [0, 5], [0, 5])


# %% [markdown]
# ## Grids


# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

# %%
grids = agent.solution[T].consumption_stage.grids_before_cleanup

# %%
plot_scatter_hist(
    grids["lMat"],
    grids["blMat"],
    grids["dMat"],
    "Pension Deposit on Exogenous Grid",
    r"Market Resources $\ell$",
    "Retirement balance $b$",
    "PensionFullExogenousGrid",
)

# %%
plot_scatter_hist(
    grids["mMat"],
    grids["nMat"],
    grids["dMat"],
    "Pension Deposit on Endogenous Grid",
    "Market Resources $m$",
    "Retirement balance $n$",
    "PensionFullEndogenousGrid",
)

# %%
grids = agent.solution[T].consumption_stage.grids_before_cleanup

# %%
plot_scatter_hist(
    agent.solution[T].deposit_stage.gaussian_interp.grids[0],
    agent.solution[T].deposit_stage.gaussian_interp.grids[1],
    agent.solution[T].deposit_stage.gaussian_interp.values,
    "Pension Deposit on Endogenous Grid",
    "Market Resources $m$",
    "Retirement balance $n$",
    "2ndStagePensionFullEndogenousGrid",
)

# %%
