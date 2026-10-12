---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The Health Investment Model with Sequential EGM

Part A of the [Online Appendix](#appendix-online) tests interpolation methods on a consumption-saving problem with health investment, the model of Ludwig and Schoen (2018) extended with the wage and depreciation risk that White (2015) adds. This notebook first solves it as HARK does. EGM$^n$ then splits each period into stages, each one a closed-form inversion, whose last stage produces the curvilinear mesh that ENGINE interpolates. A final part compares the two solutions and reruns two of the archived comparisons.

:::{seealso}
Part A of the [Online Appendix](#appendix-online) derives every step shown here and reports the warp sweep. [Section 4](#multinterp) defines ENGINE and its two mesh conditions. The model is in `code/HARKmodels/ConsHealthSeparableModel.py`, the matched ENGINE search margin in `code/HARKmodels/health_interp.py`, and the archived comparisons in `code/figures/health_model_interpolation_benchmark.py`.
:::

```{code-cell} ipython3
import sys

sys.path.append("../")
sys.path.append("../figures")

import matplotlib.pyplot as plt
import numpy as np
from cli import format_table
from engine.engine2d import Engine2DInterp, ipm_violations
from figure_styles import apply_engine_style
from generate_grid_transformation import (
    draw_health_grid_transformation,
    get_health_model_grids,
)
from HARK.ConsumptionSaving.ConsHealthModel import eval_marg_health_prod
from HARK.interpolation import Curvilinear2DInterp
from HARK.rewards import CRRAutilityP
from health_model_interpolation_benchmark import (
    POLICY_DIFF_CSV,
    SOLVER_EQUIVALENCE_CSV,
    draw_interpolation_errors,
    interpolation_accuracy,
    policy_diff_rows,
    solved,
    solver_equivalence_rows,
)
from HARKmodels.ConsHealthSeparableModel import (
    BasicHealthConsumerType,
    SequentialHealthConsumerType,
    invert_sequential_stages,
    paper_timing_params,
    post_decision_marginals,
    post_decision_mesh,
)
from HARKmodels.health_interp import HEALTH_CURVATURE_BOUND, HealthEngine2DInterp
from repo_paths import FIG_DIR, read_rows

apply_engine_style()
```

## The Household Problem

Each period the household observes market resources $\mRat_t$ and health capital $\hRat_t$, and chooses consumption $\cRat_t$ and health investment $\nRat_t$:

$$
\begin{aligned}
\vFunc_t(\mRat_t, \hRat_t) &= \max_{\cRat_t \geq 0, \, \nRat_t \geq 0} \util(\cRat_t) + \DiscFac \Surv_t \Ex_t \left[ \vFunc_{t+1}(\mRat_{t+1}, \hRat_{t+1}) \right] \\
\HRat_t &= \hRat_t + \gFunc(\nRat_t), \qquad \aRat_t = \mRat_t - \cRat_t - \nRat_t, \qquad \Surv_t = 1 - \DiePrb_t / (1 + \HRat_t), \\
\hRat_{t+1} &= (1 - \delta_{t+1}) \HRat_t, \qquad \mRat_{t+1} = \Rfree \aRat_t + \omega_{t+1} \HRat_t,
\end{aligned}
$$

with $\util(\cRat) = \cRat^{1-\CRRA}/(1-\CRRA)$ and $\gFunc(\nRat) = (\HealthProdFac / \HealthProdExp) \nRat^{\HealthProdExp}$. Investment builds post-investment health $\HRat_t$, which lowers the probability of death and raises next period's labor income $\omega_{t+1} \HRat_t$ before it depreciates at the random rate $\delta_{t+1}$. The online appendix's timing comparison sweeps the number of backward-induction cycles and the grid size around one base point, a $44 \times 50$ grid solved over ten cycles. The calibration below is that base point.

```{code-cell} ipython3
params = paper_timing_params()
hark = BasicHealthConsumerType(**params)
wages = hark.ShockDstn[0].atoms[0]
print(
    format_table(
        ["Parameter", "Code name", "Value"],
        [
            ["Discount factor", "DiscFac", hark.DiscFac],
            ["Relative risk aversion", "CRRA", hark.CRRA],
            ["Interest factor", "Rfree", hark.Rfree[0]],
            ["Health production elasticity", "HealthProdExp", hark.HealthProdExp],
            ["Health production scale", "HealthProdFac", hark.HealthProdFac],
            ["Maximum death probability", "DieProbMax", hark.DieProbMax[0]],
            ["Mean depreciation rate", "DeprRteMean", hark.DeprRteMean[0]],
            ["Smallest wage draw", "", f"{wages.min():.4f}"],
            ["Asset grid nodes", "aXtraCount", hark.aXtraCount],
            ["Extra nodes near zero assets", "aXtraExtra", len(hark.aXtraExtra)],
            ["Health grid nodes", "HLvlGrid", hark.HLvlGrid.size],
            ["Health grid range", "HLvlGrid", f"[{hark.HLvlGrid[0]}, {hark.HLvlGrid[-1]}]"],
            ["Backward-induction cycles", "cycles", params["cycles"]],
        ],
    )
)
```

HARK's asset grid adds the extra nodes near zero to the regular ones. In the timing table, a grid labelled $44 \times 50$ counts only the regular asset nodes and is the grid of this calibration.

:::{warning}
Both solvers require $\CRRA < 1$ and $0 < \HealthProdExp < 1$. With $\CRRA > 1$ utility is negative, so the survival weight $\Surv_t$ would reward mortality, and the ratio of marginal values that the investment inversion needs turns negative. Each solver checks this contract and raises an error, as `BasicHealthConsumerType(**params, CRRA=2.0).solve()` shows.
:::

## HARK's Baseline Solution

White (2015) folds the transitions and the risk into the end-of-period value $\wFunc_t(\aRat_t, \HRat_t) = \DiscFac \Surv_t \Ex_t[\vFunc_{t+1}]$ and inverts both first-order conditions at once on a rectangular grid of $(\aRat_t, \HRat_t)$:

$$
\cRat_t = \left(\wFunc^{\aRat}_t\right)^{-1/\CRRA}, \qquad
\gFunc'(\nRat_t) = \frac{\wFunc^{\aRat}_t}{\wFunc^{\HRat}_t}, \qquad
\mRat_t = \aRat_t + \cRat_t + \nRat_t, \qquad
\hRat_t = \HRat_t - \gFunc(\nRat_t).
$$

The inverted states $(\mRat_t, \hRat_t)$ form a curvilinear mesh, which White interpolates with the sector-walking method HARK implements as `Curvilinear2DInterp`. `BasicHealthConsumerType` runs that solver with that interpolant.

:::{note}
The model code follows White's HARK implementation with two documented departures. Labor income accrues on post-investment health before depreciation, $\yRat_{t+1} = \omega_{t+1} \HRat_t$, as the online appendix states the law of motion. The health grid starts at $\HRat = 0.01$, because at $\HRat = 0$ and zero assets the investment inversion divides one infinite marginal value by another.
:::

```{code-cell} ipython3
hark.solve()
x_hark, y_hark = hark.solution[0].x_values, hark.solution[0].y_values
print(f"periods solved: {len(hark.solution)}")
print(f"endogenous mesh at t = 0: {x_hark.shape[0]} x {x_hark.shape[1]} nodes")
```

Every period's solution is one interpolant with three outputs: the inverse value $\util^{-1}(\vFunc_t)$, consumption and health investment. Its nodes at $t = 0$:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_hark.ravel(), y_hark.ravel(), ".", color="0.2", ms=2)
ax.set_xlim(0.0, 100.0)
ax.set_ylim(0.0, 50.0)
ax.set_xlabel(r"Market Resources $m_t$")
ax.set_ylabel(r"Health Capital $h_t$")
plt.show()
```

Each near-vertical string of dots is one asset node traced across the health grid. At low post-investment health the inversion assigns more investment, which maps the same asset node to more market resources and less beginning health. Every string therefore bends to the right as health falls. Nodes crowd near $\mRat_t = 0$, where the asset grid is densest.

Below are cross-sections of the two policies at health capital $\hRat_t = 20$.

```{code-cell} ipython3
m_line = np.linspace(0.0, 20.0, 400)
h_line = np.full_like(m_line, 20.0)
_, c_hark, n_hark = hark.solution[0](m_line, h_line)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(m_line, c_hark, label=r"Consumption $c_t$")
ax.plot(m_line, n_hark, label=r"Health Investment $n_t$")
ax.set_xlabel(r"Market Resources $m_t$")
ax.legend()
plt.show()

kink_row = 1 + hark.constrained_N  # the zero-asset row of the mesh
m_kink = np.interp(20.0, y_hark[kink_row], x_hark[kink_row])
print(f"at h = 20, the zero-asset mesh line lies at m = {m_kink:.2f}")
```

Both policies rise with market resources. The kink at the printed value of $\mRat_t$ marks where the household starts to save. Below it the liquidity constraint binds. Every additional unit of resources then goes to consumption or investment. Investment stays positive at every $\mRat_t > 0$, since the marginal product $\gFunc'(\nRat)$ grows without bound as $\nRat \to 0$.

## The Stage Decomposition

EGM$^n$ splits the period into three stages, each solved by one inversion. Following the online appendix, $v^0_t \equiv \vFunc_t$ denotes the value at the first decision stage and $v^1_t$ the value once investment is chosen.

1. The post-decision stage values the end-of-period states, $\wFunc_t(\aRat_t, \HRat_t)$, together with its marginals $\wFunc^{\aRat}_t$ and $\wFunc^{\HRat}_t$.
2. The consumption stage solves $v^1_t(\lRat_t, \HRat_t) = \max_{\cRat_t} \util(\cRat_t) + \wFunc_t(\lRat_t - \cRat_t, \HRat_t)$. For each fixed $\HRat_t$ this is a standard consumption-saving problem: $\cRat_t = (\wFunc^{\aRat}_t)^{-1/\CRRA}$ and liquid wealth $\lRat_t = \aRat_t + \cRat_t$.
3. The health investment stage solves $v^0_t(\mRat_t, \hRat_t) = \max_{\nRat_t} v^1_t(\mRat_t - \nRat_t, \hRat_t + \gFunc(\nRat_t))$. Its first-order condition $\gFunc'(\nRat_t) = (v^1_t)^{\lRat} / (v^1_t)^{\HRat}$ inverts in closed form. The transitions then return $\mRat_t = \lRat_t + \nRat_t$ and $\hRat_t = \HRat_t - \gFunc(\nRat_t)$.

:::{important}
The investment stage is an instance of the invertible-transition-marginal form of [Section 5](#conditions). For $0 < \HealthProdExp < 1$ the marginal product $\gFunc'(\nRat) = \HealthProdFac \nRat^{\HealthProdExp - 1}$ is strictly decreasing, which lets the condition give $\nRat_t = \left( (v^1_t)^{\lRat} / (\HealthProdFac \, (v^1_t)^{\HRat}) \right)^{1/(\HealthProdExp - 1)}$ directly. No stage searches for a root.
:::

`SequentialHealthConsumerType` keeps each stage as its own interpolant. Run here with ENGINE:

```{code-cell} ipython3
egmn = SequentialHealthConsumerType(interpolator=HealthEngine2DInterp, **params)
egmn.solve()
sol = egmn.solution[0]
for name in ("post_decision", "consumption", "health_investment"):
    print(f"{name:18s} {type(getattr(sol, name).interp).__name__}")
```

The first two stages are defined on the rectangular $(\aRat, \HRat)$ grid. Only the last one, which maps to the decision-time states $(\mRat, \hRat)$, needs curvilinear interpolation.

:::{note}
ENGINE's bracket search looks a bounded number of cells away from the previous bracket. The online appendix measures that displacement on the health meshes at 10 to 16, far above ENGINE's default margin of 2, so `HealthEngine2DInterp` sets the margin to the value printed below. Both settings give the same values, because ENGINE verifies every windowed bracket and searches the whole row on a miss. Matching the margin spares those extra searches.
:::

The cell below solves the model again at the default margin. It compares every period's mesh and the policies at a grid of off-mesh points with the matched-margin solution.

```{code-cell} ipython3
print(f"search margin used for the health meshes: {HEALTH_CURVATURE_BOUND}")
default = SequentialHealthConsumerType(interpolator=Engine2DInterp, **params)
default.solve()
same_mesh = all(
    np.array_equal(a.health_investment.interp.fx, b.health_investment.interp.fx)
    and np.array_equal(a.health_investment.interp.fy, b.health_investment.interp.fy)
    for a, b in zip(egmn.solution, default.solution, strict=True)
)
m_q, h_q = np.meshgrid(np.linspace(0.5, 25.0, 50), np.linspace(0.5, 25.0, 40), indexing="ij")
same_policy = all(
    np.array_equal(x, y) for x, y in zip(sol(m_q, h_q), default.solution[0](m_q, h_q), strict=True)
)
print(f"default margin 2: identical meshes in all periods: {same_mesh}")
print(f"default margin 2: identical policies at t = 0:      {same_policy}")
```

### Every Stage in Closed Form

The solver's stage functions are ordinary module functions, which this notebook calls directly. Rebuilding the post-decision marginals at $t = 0$ from next period's solution and applying the two inversions reproduces the mesh ENGINE interpolates. Both first-order conditions hold at every node.

```{code-cell} ipython3
constrained, a_mat, H_mat = post_decision_mesh(egmn.aLvlGrid, egmn.HLvlGrid, egmn.ShockDstn[0])
rho, eps, gam = egmn.CRRA, egmn.HealthProdExp, egmn.HealthProdFac
w, w_a, w_H = post_decision_marginals(
    egmn.solution[1],
    egmn.ShockDstn[0],
    a_mat,
    H_mat,
    (egmn.Rfree[0], rho, eps, gam),
    egmn.DieProbMax[0],
    egmn.DiscFac,
)
c, ell, n, m, h = invert_sequential_stages(w_a, w_H, a_mat, H_mat, rho, eps, gam)

mesh = sol.health_investment.interp
interior = slice(1 + egmn.constrained_N, None)  # below: the m = 0 row and the constrained rows
print(f"max |u'(c) / w_a - 1|        = {np.max(np.abs(CRRAutilityP(c, rho) / w_a - 1)):.1e}")
print(f"max |g'(n) w_H / w_a - 1|    = {np.max(np.abs(eval_marg_health_prod(n, eps, gam) * w_H / w_a - 1)):.1e}")
print(f"max |m - ENGINE mesh m|      = {np.max(np.abs(mesh.fx[interior] - m)):.1e}")
print(f"max |h - ENGINE mesh h|      = {np.max(np.abs(mesh.fy[interior] - h)):.1e}")
```

The investment stage works on the consumption stage's own nodes $(\lRat, \HRat)$, where the envelope conditions give $(v^1_t)^{\lRat} = \util'(\cRat_t) = \wFunc^{\aRat}_t$ and $(v^1_t)^{\HRat} = \wFunc^{\HRat}_t$. No interpolation therefore separates the two inversions. Because every wage draw is positive (the calibration table prints the smallest), a household can end the period with no assets. Beneath the inverted nodes, the solver adds `constrained_N` liquidity-constrained rows at $\aRat = 0$ and a row at $\mRat = 0$.

### The Endogenous Mesh

The paper's grid-transformation figure, redrawn below on its $25 \times 20$ grid for a two-period model, shows the map from the regular post-decision grid to the curvilinear decision-time mesh:

```{code-cell} ipython3
fig = draw_health_grid_transformation(get_health_model_grids())
plt.show()
```

Lines of constant post-investment health and constant assets keep their order through the warp, the index-monotone property (IPM) ENGINE requires. The check below runs on every period's mesh of the ten-cycle solution.

```{code-cell} ipython3
rows = []
for t, sol_t in enumerate(egmn.solution):
    x, y = sol_t.health_investment.interp.fx, sol_t.health_investment.interp.fy
    rows.append([t, sum(ipm_violations(x, y))])
print(format_table(["t", "IPM breaks"], rows))
```

Every mesh satisfies IPM.

## Comparing the Two Solutions

### The Same Inversion

Given the same interpolant, the two solvers perform the same inversions on the same grid. The archive tests this on its small accuracy grid, where `solver_equivalence_rows` solves the model both ways under HARK's curvilinear interpolant and measures the gap between the two sets of policies at off-mesh points; `health_solver_equivalence.csv` stores the result. Rerunning the check reproduces the file.

```{code-cell} ipython3
def compare_to_csv(path, rows, n_keys=1):
    """Print freshly computed rows beside the committed CSV at ``path``; the
    first ``n_keys`` columns label each row and must match the file's.
    """
    published = read_rows(path)
    header = list(published[0])
    table = []
    for row, pub in zip(rows, published, strict=True):
        key = [pub[col] for col in header[:n_keys]]
        if row[:n_keys] != key:
            raise ValueError(f"row {row[:n_keys]} does not match published row {key}")
        table += [
            [" ".join([*key, col]), value, pub[col]]
            for col, value in zip(header[n_keys:], row[n_keys:], strict=True)
        ]
    print(format_table(["Reading", "This run", "Published"], table))


compare_to_csv(SOLVER_EQUIVALENCE_CSV, solver_equivalence_rows())
```

The largest absolute gap in consumption is below $10^{-12}$, rounding error.

:::{note}
Agreement between the two solvers holds to rounding, not bit for bit. Where the sequential solver computes $\wFunc^{\aRat}$ as `DiscFac * Surv * dvda`, the simultaneous solver keeps HARK's grouping `DiscFac * (LivPrb * dvdm)`. Because floating-point multiplication is not associative, the two products differ in their last bits.
:::

Because the solvers agree, any difference between two solutions computed at the same grid comes from the interpolant alone.

### Three Interpolants

The archive's second check holds the simultaneous solver fixed and runs each interpolant inside its own backward solve: White's curvilinear method, ENGINE and Delaunay triangulation, which Ludwig and Schoen (2018) introduced on the version of this model whose only risk is survival. On a uniform $50 \times 40$ grid of off-mesh points spanning $\mRat, \hRat \in [0.5, 25]$, it measures each method's relative policy difference from the curvilinear solution, stored in `health_interp_policy_diff.csv`. Rerun here:

```{code-cell} ipython3
accuracy = interpolation_accuracy()
compare_to_csv(POLICY_DIFF_CSV, policy_diff_rows(accuracy), n_keys=2)
```

The paper's error map draws the consumption difference over the query plane:

```{code-cell} ipython3
fig = draw_interpolation_errors(accuracy)
reference = solved(BasicHealthConsumerType, Curvilinear2DInterp).solution[0]
kink_row = 1 + egmn.constrained_N  # the zero-asset row of the mesh
for ax in fig.axes[:2]:
    limits = ax.get_xlim(), ax.get_ylim()
    ax.plot(
        reference.x_values[kink_row],
        reference.y_values[kink_row],
        color="black",
        lw=1.0,
        ls="--",
        label="Zero-Asset Mesh Line",
    )
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
fig.axes[0].legend(loc="lower right", fontsize=8)
plt.show()
```

ENGINE stays close to the curvilinear solution over most of the plane. Its largest differences lie in a few spots at low market resources, all on the dashed mesh line of zero assets, where the liquidity constraint stops binding and the policies kink. The triangulation's differences are larger and cover the whole plane, with its maximum at the lowest market resources.

:::{seealso}
Part A's warp sweep scores all three methods against the model's own first-order conditions across nine calibrations, with no reference solution. `code/figures/health_warp_sweep.py` regenerates it.
:::

### Solve Times

Timings depend on the machine, so this page reads the published measurements from `health_model_summary.csv` instead of timing the solves during the build. Each cell is the mean of ten timed solves.

```{code-cell} ipython3
summary = {}
for r in read_rows(FIG_DIR / "health_model_summary.csv"):
    summary.setdefault((r["sweep"], r["label"]), {})[r["series"]] = float(r["mean_s"])

rows = []
for (sweep, label), s in summary.items():
    base = s["HARK baseline"]
    point = f"{label} cycles, 44x50 grid" if sweep == "cycles" else f"{label} grid, 10 cycles"
    egmn_s = s["EGM$^n$ + ENGINE"]
    rows.append([point, f"{base:.3f}", f"{egmn_s:.3f} ({base / egmn_s:.1f}x)"])
print(format_table(["Configuration", "Simultaneous + curvilinear", "EGM^n + ENGINE"], rows))
```

Times are in seconds. The ratio in parentheses is the curvilinear baseline's time over that of EGM$^n$ with ENGINE, which solves the model 2.8 to 4.7 times faster across both sweeps.

:::{note}
To regenerate the timings on your own machine, run `PYTHONPATH=code uv run python code/figures/generate_health_model_summary.py` from the repository root; `--plot-only` redraws the figure from the committed file.
:::
