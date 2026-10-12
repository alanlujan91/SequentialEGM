# EGM$^n$ and ENGINE Examples

This directory contains executable MyST Markdown notebooks for the Sequential Endogenous Grid Method (EGM$^n$) and ENGINE interpolation. Each example is an executable `.md` file with `{code-cell}` blocks that MyST runs at build time. They follow the paper's order.

## Example Notebooks

### `example_ConsumptionLabor`

The consumption-labor model of Section 2, solved in two stages, each a closed-form inversion. A risky portfolio-share stage is then added to the same model while the labor and consumption stages are reused unchanged. The share stage has no closed-form inversion, so it is the worked instance of the fallback the paper describes in Section 5: the first-order condition is evaluated on a grid of shares and its first sign change is interpolated.

### `example_TwoAssetHousehold`

Solves the two-asset liquid-illiquid household block of Section 3 with EGM$^n$: the closed-form illiquid inversion, the liquid-resources grid bound, the endogenous mesh and its fold-free and index-monotone checks, the adjustment wedge, and the policies. It then solves the same model with NEGM and reruns one row of the paper's accuracy comparison beside the published values.

### `example_HealthInvestment`

The health investment model of Online Appendix A: HARK's simultaneous solution with curvilinear interpolation, the EGM$^n$ stage decomposition with ENGINE, the mesh checks on every period, and the comparison between the two, rerun beside the archive's committed tables.

### `example_ENGINEInterpolation`

The ENGINE interpolation algorithm of Section 4: its two mesh conditions, both passes worked by hand and matched against the implementation, the pass order, the regrid onto a regular grid, and accuracy and timing beside HARK's curvilinear interpolant and Delaunay triangulation.

## Running the Examples

Each example is an executable MyST Markdown notebook. Build and execute them with MyST:

```bash
myst build --execute
```

To work with one interactively, convert it to a Jupyter notebook with Jupytext:

```bash
jupytext --to notebook example_ENGINEInterpolation.md
jupyter notebook example_ENGINEInterpolation.ipynb
```
