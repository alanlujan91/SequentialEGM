---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: egmn-dev
    language: python
    name: python3
---

```python
import sys

import matplotlib.pyplot as plt

sys.path.append("../")
```

```python
from egmn.ConsLaborSeparableModel import LaborSeparableConsumerType
from egmn.utilities import plot_3d_func
from HARK.utilities import plot_funcs

figures_path = "../../content/figures/"
```

```python
agent = LaborSeparableConsumerType(aXtraNestFac=-1, aXtraCount=25, cycles=10)
```

```python
grids = agent.solution_terminal.terminal_grids
```

```python
plt.scatter(grids["bnrm"], grids["tshk"], c=grids["labor"], norm="symlog")
plt.colorbar()
plt.ylim([0.85, 1.2])
plt.savefig(figures_path + "LaborSeparableWarpedGrid.svg")
plt.savefig(figures_path + "LaborSeparableWarpedGrid.pdf")
```

```python
plt.scatter(grids["mnrm"], grids["tshk"], c=grids["labor"], norm="symlog")
plt.colorbar()
plt.ylim([0.85, 1.2])
plt.savefig(figures_path + "LaborSeparableRectangularGrid.svg")
plt.savefig(figures_path + "LaborSeparableRectangularGrid.pdf")
```

```python
plot_funcs(agent.solution_terminal.labor_leisure.c_func.xInterpolators, 0, 5)
```

```python
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
```

```python
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
```

```python
agent.solve()
```

```python
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
```

```python

```
