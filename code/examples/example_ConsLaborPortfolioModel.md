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

sys.path.append("../")
```

```python
import matplotlib.pyplot as plt
import numpy as np
from egmn.ConsLaborSeparableModel import LaborPortfolioConsumerType
from HARK.utilities import plot_funcs
```

```python pycharm={"name": "#%%\n"}
agent = LaborPortfolioConsumerType()
agent.cycles = 10
```

```python
def plot_3d_func(func, lims_x, lims_y, n=100, label_x="x", label_y="y", label_z="z"):
    # get_ipython().run_line_magic("matplotlib", "widget")
    xmin, xmax = lims_x
    ymin, ymax = lims_y
    xgrid = np.linspace(xmin, xmax, n)
    ygrid = np.linspace(ymin, ymax, n)

    xMat, yMat = np.meshgrid(xgrid, ygrid, indexing="ij")

    zMat = func(xMat, yMat)

    ax = plt.axes(projection="3d")
    ax.plot_surface(xMat, yMat, zMat, cmap="viridis")
    ax.set_title("surface")
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)
    plt.show()
```

```python pycharm={"name": "#%%\n"}
agent.solve()
```

```python
share_func = agent.solution[0].portfolio_stage.share_func
c_func = agent.solution[0].consumption_stage.c_func
labor_func = agent.solution[0].labor_stage.labor_func
leisure_func = agent.solution[0].labor_stage.leisure_func
```

```python
plot_funcs(share_func, 0, 100)
```

```python
plot_funcs(c_func, 0, 100)
```

```python pycharm={"name": "#%%\n"}
plot_3d_func(
    labor_func,
    (0, 10),
    [min(agent.TranShkGrid), max(agent.TranShkGrid)],
    label_x="m",
    label_y=r"$\theta$",
    label_z="labor",
)
```

```python
plot_3d_func(
    leisure_func,
    (0, 10),
    [min(agent.TranShkGrid), max(agent.TranShkGrid)],
    label_x="m",
    label_y=r"$\theta$",
    label_z="leisure",
)
```

```python pycharm={"name": "#%%\n"}
plot_funcs(agent.solution[0].consumption_stage.c_func, 0, 10)
```

```python

```
