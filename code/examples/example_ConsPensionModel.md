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
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import sys

sys.path.append("../")
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
from egmn.ConsPensionModel import PensionConsumerType
from egmn.utilities import plot_3d_func, plot_scatter_hist

figures_path = "../../content/figures/"
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent = PensionConsumerType(cycles=19)
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent.solve()

T = 0
```

## Post Decision Stage


```python
plot_3d_func(agent.solution[T].post_decision_stage.v_func.vFuncNvrs, [0, 5], [0, 5])
```

```python
plot_3d_func(agent.solution[T].post_decision_stage.dvda_func.cFunc, [0, 5], [0, 5])
```

```python
plot_3d_func(agent.solution[T].post_decision_stage.dvdb_func.cFunc, [0, 5], [0, 5])
```

## Consumption Stage


```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].consumption_stage.c_func, [0, 5], [0, 5])
```

```python
plot_3d_func(agent.solution[T].consumption_stage.v_func.vFuncNvrs, [0, 5], [0, 5])
```

```python
plot_3d_func(agent.solution[T].consumption_stage.dvdl_func.cFunc, [0, 5], [0, 5])
```

```python
plot_3d_func(agent.solution[T].consumption_stage.dvdb_func.cFunc, [0, 5], [0, 5])
```

## Deposit Stage


```python
plot_3d_func(agent.solution[T].deposit_stage.d_func, [0, 5], [0, 5])
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].deposit_stage.c_func, [0, 5], [0, 5])
```

```python
plot_3d_func(agent.solution[T].deposit_stage.v_func.vFuncNvrs, [0, 5], [0, 5])
```

```python
plot_3d_func(agent.solution[T].deposit_stage.dvdm_func.cFunc, [0, 5], [0, 5])
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_3d_func(agent.solution[T].deposit_stage.dvdn_func.cFunc, [0, 5], [0, 5])
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
%time
plot_3d_func(agent.solution[T].deposit_stage.gaussian_interp, [0, 5], [0, 5])
```


## Grids



```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}

```

```python
grids = agent.solution[T].consumption_stage.grids_before_cleanup
```

```python
plot_scatter_hist(
    grids["lMat"],
    grids["blMat"],
    grids["dMat"],
    "Pension Deposit on Exogenous Grid",
    r"Market Resources $\ell$",
    "Retirement balance $b$",
    "PensionFullExogenousGrid",
)
```

```python
plot_scatter_hist(
    grids["mMat"],
    grids["nMat"],
    grids["dMat"],
    "Pension Deposit on Endogenous Grid",
    "Market Resources $m$",
    "Retirement balance $n$",
    "PensionFullEndogenousGrid",
)
```

```python
grids = agent.solution[T].consumption_stage.grids_before_cleanup
```

```python
plot_scatter_hist(
    agent.solution[T].deposit_stage.gaussian_interp.grids[0],
    agent.solution[T].deposit_stage.gaussian_interp.grids[1],
    agent.solution[T].deposit_stage.gaussian_interp.values,
    "Pension Deposit on Endogenous Grid",
    "Market Resources $m$",
    "Retirement balance $n$",
    "2ndStagePensionFullEndogenousGrid",
)
```

```python

```
