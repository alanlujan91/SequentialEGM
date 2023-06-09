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

# %%
from ConsLaborSeparableModel import LaborSeparableConsumerType

agent = LaborSeparableConsumerType()

# %%
from HARK.utilities import plot_funcs

plot_funcs(agent.solution_terminal.cFunc, 0, 5)
