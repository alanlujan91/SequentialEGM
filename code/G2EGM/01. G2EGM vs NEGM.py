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
#     display_name: Python 3.9.13 ('consav-dev')
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # G2EGM vs. NEGM
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

np.seterr(all="ignore")  # ignoring all warnings

# %% pycharm={"name": "#%%\n"}
# load the G2EGMModel module
from G2EGMModel import G2EGMModelClass

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Choose number of threads in numba
#
#
#
# nb.set_num_threads(1)


# %% [markdown] pycharm={"name": "#%% md\n"}
# # Settings
#

# %% pycharm={"name": "#%%\n"}
T = 20
Neta = 16
var_eta = 0.1**2
do_print = False


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
# # NEGM
#

# %% pycharm={"name": "#%%\n"}
model_NEGM = G2EGMModelClass(
    name="NEGM", par={"solmethod": "NEGM", "T": T, "do_print": do_print}
)
model_NEGM.precompile_numba()
model_NEGM = timing(model_NEGM)


# %% pycharm={"name": "#%%\n"}
model_NEGM_shocks = G2EGMModelClass(
    name="NEGM_shocks",
    par={
        "solmethod": "NEGM",
        "T": T,
        "do_print": do_print,
        "Neta": Neta,
        "var_eta": var_eta,
    },
)
model_NEGM_shocks.precompile_numba()
model_NEGM_shocks = timing(model_NEGM_shocks)


# %% [markdown] pycharm={"name": "#%% md\n"}
# # G2EGM
#

# %% pycharm={"name": "#%%\n"}
model_G2EGM = G2EGMModelClass(
    name="G2EGM", par={"solmethod": "G2EGM", "T": T, "do_print": do_print}
)
model_G2EGM.precompile_numba()
model_G2EGM = timing(model_G2EGM)


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


# %% [markdown] pycharm={"name": "#%% md\n"}
# # Table
#

# %% pycharm={"name": "#%%\n"}
# a. models
models = [model_G2EGM, model_NEGM, model_G2EGM_shocks, model_NEGM_shocks]
postfix = "_G2EGM_vs_NEGM"

# b. euler erros
lines = []
txt = "All (average)"
for i, model in enumerate(models):
    txt += f" & {np.nanmean(model.sim.euler):.3f}"
txt += "\\\\ \n"
lines.append(txt)

txt = "\\,\\,5th percentile"
for i, model in enumerate(models):
    txt += f" & {np.nanpercentile(model.sim.euler, 5):.3f}"
txt += "\\\\ \n"
lines.append(txt)

txt = "\\,\\,95th percentile"
for i, model in enumerate(models):
    txt += f" & {np.nanpercentile(model.sim.euler, 95):.3f}"
txt += "\\\\ \n"
lines.append(txt)

with open(f"tabs_euler_errors{postfix}.tex", "w") as txtfile:
    txtfile.writelines(lines)

# c. timings
lines = []
txt = "Total"
for model in models:
    txt += f" & {np.sum(model.par.time_work) / 60:.2f}"
txt += "\\\\ \n"
lines.append(txt)

txt = "Post-decision functions"
for model in models:
    txt += f" & {np.sum(model.par.time_w) / 60:.2f}"
txt += "\\\\ \n"
lines.append(txt)

txt = "EGM-step"
for model in models:
    txt += f" & {np.sum(model.par.time_egm) / 60:.2f}"
txt += "\\\\ \n"
lines.append(txt)

txt = "VFI-step"
for model in models:
    tot_time = np.sum(model.par.time_vfi)
    if tot_time == 0:
        txt += " & "
    else:
        txt += f" & {tot_time / 60:.2f}"
txt += "\\\\ \n"
lines.append(txt)

with open(f"tabs_timings{postfix}.tex", "w") as txtfile:
    txtfile.writelines(lines)
