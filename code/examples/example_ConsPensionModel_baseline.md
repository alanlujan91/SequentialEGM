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

```python
import sys

sys.path.append("../")
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
from egmn.ConsPensionModel import PensionConsumerType, init_pension_contrib
from egmn.utilities import plot_3d_func, plot_scatter_hist
from HARK.interpolation._sklearn import GeneralizedRegressionUnstructuredInterp

figures_path = "../../content/figures/"
```

```python
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
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
agent = PensionConsumerType(**baseline_params)
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
    "PensionExogenousGrid",
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
    "PensionEndogenousGrid",
)
```

```python jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
plot_scatter_hist(
    agent.solution[T].deposit_stage.gaussian_interp.grids[0],
    agent.solution[T].deposit_stage.gaussian_interp.grids[1],
    agent.solution[T].deposit_stage.gaussian_interp.values,
    "Pension Deposit on Endogenous Grid",
    "Market Resources $m$",
    "Retirement balance $n$",
    "2ndStagePensionEndogenousGrid",
)
```

```python
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
```

```python
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
```

```python
grids = agent.solution[T].consumption_stage.grids_before_cleanup
```

```python

```

```python
gauss_interp = GeneralizedRegressionUnstructuredInterp(
    grids["dMat"],
    [grids["mMat"], grids["nMat"]],
    model="gaussian-process",
    std=True,
    model_kwargs={"normalize_y": True},
)
```

```python
# get_ipython().run_line_magic("matplotlib", "widget")
plot_3d_func(gauss_interp, [0, 5], [0, 5])
```

```python

```
