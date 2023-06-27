# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: egmn-dev
#     language: python
#     name: python3
# ---

# %%
from ConsLaborSeparableModel import LaborSeparableConsumerType
from utilities import plot_warped_bilinear_flat

agent = LaborSeparableConsumerType(Disutility=True)
from HARK.utilities import plot_funcs

# %%
plot_funcs(agent.solution_terminal.consumption_saving.vp_func, 0, 10)

# %%
plot_warped_bilinear_flat(agent.solution_terminal.labor_leisure.leisure_func, 0, 0.25)

# %%
agent.solution_terminal.labor_leisure.c_func.values.max()

# %%
agent.WageRte

# %%
