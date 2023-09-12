# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: egmn-dev
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # G2EGM
#

# %% [markdown] pycharm={"name": "#%% md\n"}
# This notebook produces the timing and accuracy results for the comparison of **NEGM** and **G$^2$EGM** in [A Guide to Solve Non-Convex Consumption-Saving Models](https://doi.org/10.1007/s10614-020-10045-x).
#

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Setup
#

# %% pycharm={"name": "#%%\n"}
# %load_ext autoreload
# %autoreload 2

import numpy as np
from figs import decision_functions, retirement, segments

np.seterr(all="ignore")  # ignoring all warnings

# %% pycharm={"name": "#%%\n"}
# load the G2EGMModel module
from G2EGMModel import G2EGMModelClass

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Choose number of threads in numba
#
#
#
# nb.set_num_threads(10)

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Settings
#

# %% pycharm={"name": "#%%\n"}
T = 20
Neta = 7
var_eta = 0.1
do_print = False

# %% [markdown]
#

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Timing function
#


# %% pycharm={"name": "#%%\n"}
def timing(model, rep=1, do_print=True):  # set to 5 in the paper
    name = model.name

    time_best = np.inf
    for i in range(rep):
        model.solve()
        model.calculate_euler()

        tot_time = np.sum(model.par.time_work)
        if do_print:
            print(f"{i}: {tot_time:.2f} secs, euler: {np.nanmean(model.sim.euler):.3f}")

        if tot_time < time_best:
            time_best = tot_time
            model_best = model.copy("best")

    model_best.name = name
    return model_best


# %% [markdown] pycharm={"name": "#%% md\n"}
# # G2EGM
#

# %% pycharm={"name": "#%%\n"}
model_G2EGM_shocks = G2EGMModelClass(
    name="G2EGM_shocks",
    par={
        "solmethod": "G2EGM",
        "T": T,
        "do_print": do_print,
        "Neta": Neta,
        "var_eta": var_eta,
    },
)
model_G2EGM_shocks.precompile_numba()
model_G2EGM_shocks = timing(model_G2EGM_shocks)

# %% pycharm={"name": "#%%\n"}
retirement(model_G2EGM_shocks)

# %% pycharm={"name": "#%%\n"}
decision_functions(model_G2EGM_shocks, 0)

# %% pycharm={"name": "#%%\n"}
segments(model_G2EGM_shocks, 0)

# %% pycharm={"name": "#%%\n"}
