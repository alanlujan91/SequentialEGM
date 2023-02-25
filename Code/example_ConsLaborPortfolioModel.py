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

# %% pycharm={"name": "#%%\n"}
from HARK.ConsumptionSaving.ConsLaborPortfolioModel import LaborPortfolioConsumerType
from HARK.utilities import plot_funcs

# %% pycharm={"name": "#%%\n"}
agent = LaborPortfolioConsumerType()
agent.cycles = 0

# %% pycharm={"name": "#%%\n"}
labor_stage = agent.solution_terminal.labor_stage

# %% pycharm={"name": "#%%\n"}
plot_funcs(labor_stage.laborFunc.xInterpolators, -1, 10)

# %% pycharm={"name": "#%%\n"}
agent.solve()

# %% pycharm={"name": "#%%\n"}
plot_funcs(agent.solution[0].portfolio_stage.shareFunc, 0, 10)

# %% pycharm={"name": "#%%\n"}
plot_funcs(agent.solution[0].consumption_stage.cFunc, 0, 10)

# %% pycharm={"name": "#%%\n"}
plot_funcs(agent.solution[0].labor_stage.laborFunc.xInterpolators, 0, 10)

# %% pycharm={"name": "#%%\n"}
