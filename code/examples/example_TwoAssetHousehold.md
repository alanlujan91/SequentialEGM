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

# The Two-Asset Household with Sequential EGM

This notebook solves the two-asset liquid-illiquid household block of [Section 3](#multdim-abrs) at the published calibration of Auclert et al. (2021). It walks through the closed-form illiquid inversion, solves the household to its stationary policies, draws the endogenous mesh the inversion produces, and reruns one row of the paper's comparison with the nested method (NEGM).

:::{seealso}
[Section 3](#multdim-abrs) derives every step shown here. Part B of the [Online Appendix](#appendix-online) reports the full-solve interpolation diagnostic. The solver itself is `code/ssj/hh_egm.py`; `code/ssj/benchmark.py` holds the calibration, the solvers and the accuracy measures this notebook calls.
:::

```{code-cell} ipython3
import sys

sys.path.append("../")
sys.path.append("../figures")

import matplotlib.pyplot as plt
import numpy as np
from cli import format_table
from figure_styles import apply_engine_style
from generate_grid_transformation_abrs import (
    draw_grid_transformation,
    get_two_asset_meshes,
)
from matplotlib.colors import TwoSlopeNorm
from repo_paths import FIG_DIR, read_rows
from ssj.bench_accuracy_negm import accuracy_row
from ssj.benchmark import (
    assert_mesh_fold_free,
    assert_mesh_ipm,
    budget_violation,
    calibration,
    l_grid_for,
    solve_compact,
    solve_negm,
    stage0_mesh,
)
from ssj.hh_egm import psi_and_derivs, stage0_node

apply_engine_style()
```

## The Household Problem

A household enters the period with liquid wealth $b$, an illiquid balance $a$ and productivity $e$, which follows a three-state Markov chain. Out of market resources $m = R_b b + R_a a + y(e)$ it chooses consumption $c$ and new balances $b' \geq 0$, $a' \geq 0$ subject to

$$
c + b' + a' + \Psi(a', a) = m, \qquad
\Psi(a', a) = \frac{\chi_1}{2} \left( \frac{a' - R_a a}{R_a a + \chi_0} \right)^{2} (R_a a + \chi_0).
$$

The illiquid asset pays the higher return, but moving funds into or out of it incurs the convex cost $\Psi$. Every two-asset exhibit in the paper uses the calibration below. The table computes relative risk aversion as the inverse of the elasticity of intertemporal substitution, which the code stores.

```{code-cell} ipython3
cal = calibration(40, 60)
print(format_table(
    ["Parameter", "Value"],
    [
        ["Discount factor (beta)", cal["beta"]],
        ["Relative risk aversion (rho)", 1 / cal["eis"]],
        ["Liquid return (r_b)", cal["rb"]],
        ["Illiquid return (r_a)", cal["ra"]],
        ["Cost offset (chi_0)", cal["chi0"]],
        ["Cost scale (chi_1)", cal["chi1"]],
        ["Productivity persistence", cal["rho_z"]],
        ["Productivity dispersion", cal["sigma_z"]],
        ["Largest liquid node", cal["bmax"]],
        ["Largest illiquid node", cal["amax"]],
    ],
))
```

## The Illiquid Stage in Closed Form

The period splits into an illiquid choice (Stage 0), a consumption choice (Stage 1) and the expectation over next period's productivity. Stage 1 is the standard consumption EGM at each fixed $a'$. It leaves liquid resources $\ell = m - a' - \Psi(a', a)$ as the state of Stage 0, whose first-order condition is

$$
1 + \Psi_1(a', a) = \frac{v^1_{a'}(\ell, a', e)}{v^1_{\ell}(\ell, a', e)}.
$$

Nested methods search for the $a'$ that solves this equation at each state. EGM$^n$ fixes $a'$ on a grid and solves for the incoming balance $a$ instead. With the quadratic cost, $\Psi_1 = \chi_1 u$ for the adjustment rate $u = (a' - R_a a)/(R_a a + \chi_0)$, so the wedge $\tau = v^1_{a'}/v^1_{\ell} - 1$ inverts in two lines:

$$
u^* = \frac{\tau}{\chi_1}, \qquad a = \frac{a' - u^* \chi_0}{R_a (1 + u^*)}.
$$

:::{important}
The closed-form inversion is the whole difference between EGM$^n$ and NEGM. Where the nested method runs a bracketing search over $a'$ at every state, EGM$^n$ spends a few arithmetic operations per grid node, with no iteration. The inversion exists because $\Psi_1$ is strictly decreasing in the incoming balance $a$, the invertible-transition-marginal structure of [Section 5](#conditions).
:::

At every node of the $(\ell, a')$ grid, the solver applies `stage0_node`. One call at illustrative inputs shows that the incoming balance it returns satisfies the first-order condition.

```{code-cell} ipython3
Ra, Rb = 1 + cal["ra"], 1 + cal["rb"]
chi0, chi1, chi2 = cal["chi0"], cal["chi1"], cal["chi2"]

v1_ap, c, a_prime, ell = 0.9, 1.2, 3.0, 5.0  # Stage 1 outputs at one node
tau = v1_ap * c ** (1 / cal["eis"]) - 1  # v1_ap / u'(c) - 1
a_in, b_in = stage0_node(
    v1_ap, c, a_prime, ell, cal["z"][1], 1 / cal["eis"], Ra, Rb, chi0, chi1, chi2
)
_, psi1, _ = psi_and_derivs(a_prime, a_in, Ra, chi0, chi1, chi2)
print(f"incoming balances: a = {a_in:.6f}, b = {b_in:.6f}")
print(f"Psi_1(a', a) = {psi1:.15f}")
print(f"wedge tau    = {tau:.15f}")
```

Up to rounding error, the marginal cost $\Psi_1$ equals the wedge $\tau$. The incoming liquid balance $b$ then follows from the two accounting identities, with no root-finding or iteration at any step.

## Solving the Household

`solve_compact` iterates the EGM$^n$ backward step from the initial marginal values of the sequence-space Jacobian toolkit until the marginal values change by less than $10^{-9}$, the tolerance of every published solve. It raises an error if the iteration does not converge.

```{code-cell} ipython3
egmn = solve_compact(cal)
print(f"EGM^n converged in {egmn['iter']} iterations")
print(f"largest budget residual: {budget_violation(egmn, cal).max():.1e}")
adjustment = np.abs(egmn["a"] - (1 + cal["ra"]) * cal["a"][None, None, :])
print(f"median illiquid adjustment |a' - R_a a|: {np.median(adjustment):.3f}")
```

The delivered policies satisfy the within-period budget to rounding error at every node, because the liquid choice is the budget residual.

:::{tip}
`calibration` returns a plain dictionary, so any parameter can be changed before solving. Set `cal["chi1"] = 1000.0` and rerun the cells from here down. The solve still converges, in more iterations. As the household approaches the limit $\chi_1 \to \infty$, in which $a' = R_a a$ and the model collapses to one asset, the median illiquid adjustment printed above falls by about an order of magnitude. Node by node the approach is not monotone; at $\chi_1 = 20$ the median adjustment is slightly larger than at the published 6.5. `get_two_asset_meshes` builds its own calibration, so the grid-transformation figure below keeps the paper's.
:::

### The Liquid-Resources Grid

Stage 0 receives the Stage 1 policies on a regular grid of liquid resources $\ell$, which must cover every $\ell$ the inversion can request. A household rich in illiquid wealth can withdraw to finance liquidity, so liquid wealth alone does not bound that range. Since the marginal value of illiquid wealth is non-negative, the first-order condition caps the withdrawal and with it $\ell$, at

$$
\overline{\ell} = R_b \overline{b} + \overline{y} + \frac{R_a \overline{a} + \chi_0}{2 \chi_1},
$$

well past the largest liquid node. The solver builds its $\ell$-grid to this bound:

```{code-cell} ipython3
l_grid = l_grid_for(cal)
ell_bar = Rb * cal["b"][-1] + cal["z"].max() + (Ra * cal["a"][-1] + chi0) / (2 * chi1)
print(f"largest liquid node:      {cal['b'][-1]:.2f}")
print(f"closed-form bound:        {ell_bar:.2f}")
print(f"largest l-grid node:      {l_grid[-1]:.2f}")
print(f"l-grid nodes past b-grid: {l_grid.size - cal['b'].size}")
```


## The Endogenous Mesh

The inversion maps the exogenous $(\ell, a')$ grid into a curvilinear mesh of incoming states $(b, a)$. ENGINE interpolates on that mesh only if it is fold-free (no cell turns inside out) and index-monotone (each coordinate increases along its own grid index). The paper verifies both on every reported mesh. Here the same checks run on the converged $40 \times 60$ solution:

```{code-cell} ipython3
mesh = stage0_mesh(egmn, cal)
assert_mesh_fold_free(mesh["b_endo"], mesh["a_endo"])
assert_mesh_ipm(mesh["b_endo"], mesh["a_endo"])
print("Fold-free and index-monotone in all", mesh["a_endo"].shape[0], "productivity states")
```

Each check raises an error on the first violation it finds.

:::{note}
ENGINE's two conditions concern only the ordering of the mesh nodes. Nodes may land far from the exogenous grid, as the figure below shows; ENGINE's two-pass interpolation stays valid so long as no cell folds and both index orders survive.
:::

Below, the paper's grid-transformation figure is redrawn on its own $20 \times 20$ grid, where individual mesh lines stay visible.

```{code-cell} ipython3
fig = draw_grid_transformation(get_two_asset_meshes())
plt.show()
```

Where the optimal adjustment is largest, the inversion bends the mesh most. Lines of constant $\ell$ and constant $a'$ never cross. Nodes with negative $b$ belong to withdrawal-financed choices outside the state space; the regrid to the regular $(b, a)$ grid never queries them.

### The Adjustment Wedge

Every backward step produces the Stage 1 marginal values on the whole $(\ell, a')$ grid, and with them the wedge $\tau$: the shadow price, in units of liquid wealth, of moving one unit into the illiquid account. It follows from the inverted mesh through $\tau = \chi_1 u$:

```{code-cell} ipython3
ie = cal["Pi"].shape[0] // 2  # median productivity state
a_endo = mesh["a_endo"][ie]
u = (cal["a"][None, :] - Ra * a_endo) / (Ra * a_endo + chi0)
L, AP = np.meshgrid(l_grid, cal["a"], indexing="ij")

tau_grid = chi1 * u
fig, ax = plt.subplots(figsize=(7, 5))
pc = ax.pcolormesh(L, AP, tau_grid, cmap="RdBu_r", norm=TwoSlopeNorm(0.0), shading="auto")
ticks = [-1.0, -0.5, 0.0, *np.arange(10.0, tau_grid.max(), 10.0)]
fig.colorbar(pc, ax=ax, label=r"Wedge $\tau$", ticks=ticks)
ax.set_xscale("symlog", linthresh=1.0)
ax.set_yscale("symlog", linthresh=1.0)
ax.set_xlabel(r"Liquid Resources $\ell$")
ax.set_ylabel(r"Illiquid Choice $a'$")
plt.show()
```

:::{hint}
A positive wedge means the household pays to deposit; a negative one means it pays to withdraw.
:::

Households with ample liquid resources and a small illiquid position deposit at the largest rates (lower right). Where liquid resources are scarce relative to the illiquid choice, the marginal value of liquidity swamps that of the illiquid account. There the wedge approaches its lower bound of $-1$, which it attains at $\ell = 0$.

## Policy Functions

Consumption, the liquid choice and the net illiquid deposit $a' - R_a a$ on the regular $(b, a)$ grid, at the median productivity state and the four illiquid grid nodes nearest 0, 10, 100 and 1000:

```{code-cell} ipython3
ia_show = [int(np.argmin(np.abs(cal["a"] - a0))) for a0 in (0.0, 10.0, 100.0, 1000.0)]
policies = {
    "Consumption $c$": egmn["c"][ie],
    "Liquid choice $b'$": egmn["b"][ie],
    "Net deposit $a' - R_a a$": egmn["a"][ie] - Ra * cal["a"][None, :],
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (title, policy) in zip(axes, policies.items(), strict=True):
    for ia in ia_show:
        ax.plot(cal["b"], policy[:, ia], label=f"$a = {cal['a'][ia]:.0f}$")
    ax.set_xlabel("Liquid wealth $b$")
    ax.set_title(title)
axes[0].legend()
plt.tight_layout()
plt.show()
```

Holding $a$ fixed, more liquid wealth raises consumption and the liquid choice. The two larger illiquid balances fund part of their spending by withdrawing from the illiquid account, most heavily when liquid wealth is low. At the lowest $b$, the largest balance withdraws enough to carry some of it into liquid savings, so its $b'$ starts above the others.

## Comparison with the Nested Method

NEGM shares every stage with EGM$^n$ except Stage 0, where it solves the first-order condition at each regular-grid node by a safeguarded bracketing iteration. `accuracy_row` writes each row of the archive's `accuracy_negm.csv`. Run here on both solutions at the $(40, 60)$ timing grid, its output is shown beside the published row.

```{code-cell} ipython3
negm = solve_negm(cal)
grid = (len(cal["b"]), len(cal["a"]))
row = accuracy_row("timing", grid, cal, egmn, negm)
published = next(
    r
    for r in read_rows(FIG_DIR / "accuracy_negm.csv")
    if (r["family"], int(r["nB"]), int(r["nA"])) == ("timing", *grid)
)
print(format_table(
    ["Column", "This run", "Published"],
    [[k, v, published[k]] for k, v in zip(published, row, strict=True)][3:],
))
```

The two solvers converge to policies whose liquid and illiquid balances differ by less than half a percent at the median (`dist_b_median`, `dist_a_median`). At the regular grid nodes, EGM$^n$'s Euler residual is more than an order of magnitude below NEGM's (`euler_median_*`), because EGM$^n$ delivers consumption there as the output of an Euler inversion. Off the grid, where both policies are interpolated, the two medians (in $\log_{10}$) are close.

### Solve Times

Timings depend on the machine, so this page reads the published measurements from `timing_negm.csv` instead of timing the solves during the build. Every cell is the best of three warm solves to tolerance $10^{-9}$ on the benchmark machine the archive README describes.

```{code-cell} ipython3
times = {
    (int(r["nB"]), int(r["nA"]), r["solver"], int(r["threads"])): float(r["seconds"])
    for r in read_rows(FIG_DIR / "timing_negm.csv")
}
grids = sorted({(nb, na) for nb, na, _, _ in times})
rows = []
for nb, na in grids:
    cells = [f"({nb}, {na})"]
    for threads in (1, 16):
        negm_s, egmn_s = times[nb, na, "negm", threads], times[nb, na, "egmn", threads]
        cells += [f"{negm_s:.2f}", f"{egmn_s:.2f} ({negm_s / egmn_s:.1f}x)"]
    rows.append(cells)
print(format_table(
    ["Grid", "NEGM 1 thread", "EGM^n 1 thread", "NEGM 16 threads", "EGM^n 16 threads"],
    rows,
))
```

The ratio in parentheses is NEGM's time over EGM$^n$'s at the same thread count.

:::{note}
To regenerate the table on your own machine, run `PYTHONPATH=code uv run python -m ssj.bench_timing` from the repository root. The full benchmark includes a one-thread NEGM solve at $(160, 240)$ that takes about three minutes. It also refuses to run on a machine with fewer than sixteen threads.
:::
