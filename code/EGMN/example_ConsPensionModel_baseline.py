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
import matplotlib.pyplot as plt
from ConsPensionModel import PensionConsumerType, init_pension_contrib
from HARK.interpolation._sklearn import GeneralizedRegressionUnstructuredInterp
from utilities import plot_3d_func, plot_scatter_hist

figures_path = "../../content/figures/"

# %%
baseline_params = init_pension_contrib.copy()
baseline_params["mCount"] = 50
baseline_params["mMax"] = 10
baseline_params["mNestFac"] = -1

baseline_params["nCount"] = 50

baseline_params["nMax"] = 12
baseline_params["nNestFac"] = -1

baseline_params["lCount"] = 50
baseline_params["lMax"] = 9
baseline_params["lNestFac"] = -1

baseline_params["blCount"] = 50
baseline_params["blMax"] = 13
baseline_params["blNestFac"] = -1

baseline_params["aCount"] = 50
baseline_params["aMax"] = 8
baseline_params["aNestFac"] = -1

baseline_params["bCount"] = 50
baseline_params["bMax"] = 14
baseline_params["bNestFac"] = -1

baseline_params["cycles"] = 1

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent = PensionConsumerType(**baseline_params)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent.solve()

T = 0

# %% [markdown]
# ## Post Decision Stage
#

# %%
plot_3d_func(agent.solution[T].post_decision_stage.v_func.vFuncNvrs, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].post_decision_stage.dvda_func.cFunc, [0, 5], [0, 5])

# %%
plot_3d_func(agent.solution[T].post_decision_stage.dvdb_func.cFunc, [0, 5], [0, 5])

# %% [markdown]
# ## Consumption Stage
#

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
#

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
#


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
    "PensionExogenousGrid",
)

# %%
plot_scatter_hist(
    grids["mMat"],
    grids["nMat"],
    grids["dMat"],
    "Pension Deposit on Endogenous Grid",
    "Market Resources $m$",
    "Retirement balance $n$",
    "PensionEndogenousGrid",
)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_scatter_hist(
    agent.solution[T].deposit_stage.gaussian_interp.grids[0],
    agent.solution[T].deposit_stage.gaussian_interp.grids[1],
    agent.solution[T].deposit_stage.gaussian_interp.values,
    "Pension Deposit on Endogenous Grid",
    "Market Resources $m$",
    "Retirement balance $n$",
    "2ndStagePensionEndogenousGrid",
)

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
    s=5,
)
cbar = fig.colorbar(plot)
cbar.ax.set_ylabel("Pension Deposits $d$")

plt.xlim([-1, 10])
plt.ylim([-1, 10])

# %%
fig, ax = plt.subplots()
scatter = ax.scatter(
    grids["lMat"],
    grids["blMat"],
    # c=np.maximum(grids["dMat"], 0),
    # cmap="viridis",
    vmin=-2,
    vmax=15,
    plotnonfinite=True,
    alpha=0.6,
    s=5,
)
# cbar = fig.colorbar(scatter)
# cbar.ax.set_ylabel("Pension Deposits $d$")

plt.title("Pension Deposits on Exogenous Post-Decision Grid")
plt.xlabel(r"Liquid Wealth $\ell$")
plt.ylabel("Retirement Balance $b$")
fig.savefig(figures_path + "SparsePensionExogenousGrid.svg")
fig.savefig(figures_path + "SparsePensionExogenousGrid.pdf")

# %%
grids = agent.solution[T].consumption_stage.grids_before_cleanup

# %%

# %%
gauss_interp = GeneralizedRegressionUnstructuredInterp(
    grids["dMat"],
    [grids["mMat"], grids["nMat"]],
    model="gaussian-process",
    std=True,
    model_kwargs={"normalize_y": True},
)

# %%
# get_ipython().run_line_magic("matplotlib", "widget")
plot_3d_func(gauss_interp, [0, 5], [0, 5])

# %%
