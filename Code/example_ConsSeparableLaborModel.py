# %%

from SeparableConsLaborModel import SeparableLaborConsumerType

agent = SeparableLaborConsumerType()

# %%

from HARK.utilities import plot_funcs

plot_funcs(agent.solution_terminal.cFunc, 0, 5)
# %%
