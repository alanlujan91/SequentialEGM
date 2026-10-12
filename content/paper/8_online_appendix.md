---
exports:
  - id: online-appendix-pdf
    format: pdf+tex
    template: https://github.com/econ-ark/elsarticle-myst/archive/1d2a43bb5e5ee778dccf6eb772dd3b2180799a3d.zip
    options:
      document_class: elsarticle
      elsarticle_layout: 3p
    output: ../public/SequentialEGMn_online_appendix.pdf
# The PDF letters the sections A and B through \appendix; the site leaves them unnumbered.
numbering:
  headings: false
---

(appendix-online)=

# Online Appendix for "The Sequential Endogenous Grid Method"

Part A specifies the health-investment interpolation testbed and its warp sweep; Part B reports the full-solve interpolation diagnostic's protocol and readings. Section references point to the main paper.

```{raw} latex
\appendix
```

(appendix-health)=

## The Health Investment Model as an Interpolation Testbed

The two-asset household block of Section 3 is the paper's primary multidimensional application. For two reasons, this appendix retains a second two-dimensional model, the consumption-saving problem with health investment of {cite:t}`Ludwig2018` extended with the wage and depreciation risk of {cite:t}`White2015`. First, the model is home ground for both alternatives: {cite:t}`Ludwig2018` introduced Delaunay interpolation on the version whose only risk is survival, and {cite:t}`White2015` introduced curvilinear interpolation on the stochastic version adapted here. Second, the health-investment stage is a concrete instance of the invertible-transition-marginal form of the paper's Section 5.

### A Health Investment Model

Each period $t \in \{0, \ldots, T\}$, the household observes market resources $\mRat_t$ and health capital $\hRat_t$ and chooses consumption $\cRat_t$ and health investment $\nRat_t$. Consumption yields CRRA utility; health investment produces health capital through a concave production function; and health capital both reduces mortality risk and raises labor income.

The recursive problem is

\begin{equation}
    \begin{split}
        \vFunc_t(\mRat_t, \hRat_t) &= \max_{\cRat_t \geq 0, \, \nRat_t \geq 0} \util(\cRat_t) + \DiscFac \Surv_t \Ex_t \left[ \vFunc_{t+1}(\mRat_{t+1}, \hRat_{t+1}) \right] \\
        & \text{subject to} \\
        \HRat_t &= \hRat_t + \gFunc(\nRat_t) \\
        \aRat_t &= \mRat_t - \cRat_t - \nRat_t \\
        \Surv_t &= 1 - \DiePrb_t / (1 + \HRat_t) \\
        \yRat_{t+1} &= \omega_{t+1} \HRat_t \\
        \hRat_{t+1} &= (1 - \delta_{t+1}) \HRat_t \\
        \mRat_{t+1} &= \Rfree \aRat_t + \yRat_{t+1},
    \end{split}
\end{equation}

where the utility function and health production function are

\begin{equation}
    \util(\cRat) = \frac{\cRat^{1-\CRRA}}{1-\CRRA}, \qquad \gFunc(\nRat) = \frac{\HealthProdFac}{\HealthProdExp} \nRat^{\HealthProdExp}.
\end{equation}

Here $\HRat_t$ is post-investment health capital, $\aRat_t$ end-of-period assets, $\Surv_t$ the survival probability (the maximum death probability $\DiePrb_t$ applies at $\HRat_t = 0$), $\delta_{t+1}$ the stochastic health depreciation rate, $\omega_{t+1}$ the stochastic wage rate, and $\yRat_{t+1}$ labor income. {cite:t}`Ludwig2018` take this survival function from {cite:t}`Hall2007`. With scale $\HealthProdFac > 0$ and elasticity $0 < \HealthProdExp < 1$, health investment has diminishing returns. The terminal condition $\vFunc_{T+1} \equiv 0$ makes the final-period household consume all remaining resources. Higher health investment raises $\Surv_t$ and thereby the weight on future value.

This model requires $\CRRA < 1$, so utility is non-negative for all $\cRat > 0$. With a positive probability of zero wage, ending a nonterminal period with no liquid assets would risk zero consumption, so the liquidity constraint never binds before the terminal period. Strict concavity then makes the first-order conditions necessary and sufficient at interior solutions. The implementation also handles positive wage floors, like the benchmark calibration's unemployment-income floor, through an explicit liquidity-constrained segment at $\aRat = 0$. As $\HealthProdExp \to 1$, $\gFunc'$ approaches a constant, the marginal inversion becomes ill-conditioned, and at the linear limit the first-order condition no longer identifies health investment. This boundary case therefore falls outside the invertible-transition-marginal class.[^model-restrictions]

[^model-restrictions]: With $\CRRA > 1$, utility is negative and the product $\Surv_t \cdot \Ex[\vFunc_{t+1}]$ would reward mortality. The restriction mirrors the calibrations of {cite:t}`Ludwig2018` and {cite:t}`White2015` and is specific to this health formulation, not to EGM$^n$ generally. Models with mortality risk and $\CRRA > 1$ can use a bequest motive or utility differences (see {cite:t}`DeNardi2004`).

### Sequential EGM Solution

The problem splits into three stages: health investment, consumption, and a post-decision stage that handles expectations. I retain the stage superscripts of Section 2, so $\vFunc_t \equiv v^0_t$ is the first decision stage, and write $\wFunc_t$ in place of $v^2_t$ for the post-decision value, following {cite:t}`Powell2011`. Because value functions have multiple arguments, partials appear as superscripts ($\wFunc_t^{\aRat} \equiv \partial \wFunc_t / \partial \aRat$).

The post-decision stage values the end-of-period states $(\aRat_t, \HRat_t)$ before next period's shocks are realized:

\begin{equation}
    \wFunc_t(\aRat_t, \HRat_t) = \DiscFac \left(1 - \frac{\DiePrb_t}{1 + \HRat_t}\right) \Ex_t \left[ \vFunc_{t+1}(\mRat_{t+1}, \hRat_{t+1}) \right].
\end{equation}

Building $\wFunc_t$ once lets the earlier stages reuse it without repeated expectation calculations.[^expectation-stage]

[^expectation-stage]: The marginal value functions $\wFunc_t^{\aRat}(\aRat_t, \HRat_t) \equiv \partial \wFunc_t / \partial \aRat$ and $\wFunc_t^{\HRat}(\aRat_t, \HRat_t) \equiv \partial \wFunc_t / \partial \HRat$ are computed once on an exogenous grid and interpolated as needed by earlier stages. The health marginal $\wFunc_t^{\HRat}$ includes both the survival-probability benefit of health (through $\partial \Surv_t/\partial \HRat_t = \DiePrb_t/(1+\HRat_t)^2$) and the continuation-value benefit through next period's health.

After the health investment decision, the household has liquid wealth $\lRat_t$ and post-investment health $\HRat_t$. The consumption problem is

\begin{equation}
    v^{1}_t(\lRat_t, \HRat_t) = \max_{\cRat_t \geq 0} \util(\cRat_t) + \wFunc_t(\aRat_t, \HRat_t), \qquad
    \aRat_t = \lRat_t - \cRat_t.
\end{equation}

Post-investment health $\HRat_t$ passes through this stage because it enters the continuation value and the consumption decision leaves it unchanged. For each fixed $\HRat_t$, this is a standard one-dimensional consumption-saving problem that EGM solves.[^consumption-structure] On an exogenous grid $(\aMat, \HMat)$ of post-decision states, the first-order condition $\util'(\cRat_t) = \wFunc_t^{\aRat}$ inverts to $\cEndFunc_t = (\wFunc_t^{\aRat})^{-1/\CRRA}$, which defines the endogenous liquid-wealth grid $\lEndFunc_t = \aMat + \cEndFunc_t$. On that grid the envelope conditions give the investment stage the marginal values $(v^{1}_t)^{\lRat} = \util'(\cRat_t)$ and $(v^{1}_t)^{\HRat} = \wFunc_t^{\HRat}$.

[^consumption-structure]: Strict concavity of $\util$ and $\wFunc_t$ ensures joint strict concavity of $v^1_t(\lRat_t, \HRat_t)$, so the recovered optimum is unique. Grid validity is separate: concavity alone does not deliver fold-freeness or index monotonicity, so the warp sweep below tests both directly on every solved mesh.

At the beginning of the period, the household has market resources $\mRat_t$ and health capital $\hRat_t$. The health investment problem is

\begin{equation}
    v^{0}_t(\mRat_t, \hRat_t) = \max_{\nRat_t \geq 0} v^{1}_t(\lRat_t, \HRat_t), \qquad
    \lRat_t = \mRat_t - \nRat_t, \quad
    \HRat_t = \hRat_t + \gFunc(\nRat_t).
\end{equation}

With no direct utility payoff at this stage, investment moves liquid wealth through the budget and post-investment health through the production function. Because $\gFunc'$ has an explicit inverse, the EGM step can run before the transition equations recover the incoming states. The Inada condition $\lim_{\nRat \to 0}\gFunc'(\nRat) = \infty$ handles the constraint $\nRat_t \geq 0$, keeping solutions interior whenever the marginal value of health is positive and resources permit.

The first-order condition of the health investment problem rearranges to

\begin{equation}
    \gFunc'(\nRat_t) = \frac{(v^{1}_t)^{\lRat}(\lRat_t, \HRat_t)}{(v^{1}_t)^{\HRat}(\lRat_t, \HRat_t)}.
\end{equation}

Since $\gFunc'(\nRat) = \HealthProdFac \nRat^{\HealthProdExp - 1}$ is strictly decreasing for $0 < \HealthProdExp < 1$, optimal health investment on an exogenous grid $(\lMat, \HMat)$ of liquid wealth and post-investment health follows:

\begin{equation}
    \nEndFunc_t(\lMat, \HMat) = \left( \frac{(v^{1}_t)^{\lRat}(\lMat, \HMat)}{\HealthProdFac (v^{1}_t)^{\HRat}(\lMat, \HMat)} \right)^{1/(\HealthProdExp - 1)}.
\end{equation}

From the transition equations, the endogenous pre-decision states then follow:

\begin{equation}
    \begin{split}
        \mEndFunc_t(\lMat, \HMat) &= \lMat + \nEndFunc_t(\lMat, \HMat), \\
        \hEndFunc_t(\lMat, \HMat) &= \HMat - \gFunc(\nEndFunc_t(\lMat, \HMat)).
    \end{split}
\end{equation}

By the envelope theorem, the beginning-of-period marginal values are

\begin{equation}
    \begin{split}
        (v^{0}_t)^{\mRat}(\mRat_t, \hRat_t) &= (v^{1}_t)^{\lRat}(\lRat_t, \HRat_t), \\
        (v^{0}_t)^{\hRat}(\mRat_t, \hRat_t) &= (v^{1}_t)^{\HRat}(\lRat_t, \HRat_t).
    \end{split}
\end{equation}

Starting from a regular rectilinear grid of $(\aMat, \HMat)$, the two EGM steps together produce the curvilinear endogenous grid of $(\mEndFunc_t, \hEndFunc_t)$ shown in @fig:health_grids.[^health-grid-params]

```{figure} ../figures/grid_transformation.*
:name: fig:health_grids
:align: center

EGM grid transformation for the health investment model at $t=0$ (the first period of economic life). Panel (a) shows the regular exogenous grid of post-decision assets $(\aMat)$ and post-investment health $(\HMat)$. Panel (b) shows the curvilinear endogenous grid of market resources $(\mEndFunc_t)$ and beginning-of-period health $(\hEndFunc_t)$, with points colored by optimal consumption and the plot windowed to $\mEndFunc_t \leq 50$ for legibility. The endogenous grid preserves the index ordering of the exogenous grid despite substantial warping near the constraint boundary.
```

[^health-grid-params]: The figure uses $\CRRA = 0.5$, $\DiscFac = 0.95$, $\Rfree = 1.03$, health production exponent $\HealthProdExp = 0.35$, health depreciation rate $\delta \sim U[0, 0.1]$, wage rate $\omega \sim \text{lognormal}$ with mean $0.1$ and standard deviation $0.1$, and a two-period model ($T = 1$, so the household lives periods $t \in \{0, 1\}$).

Index preservation across warping enables the interpolation strategy of Section 4. The mapping $(\lRat, \HRat) \mapsto (\mRat, \hRat)$ above is the recovery map whose sampled meshes the warp sweep below measures.

### Interpolation Comparison

This appendix adds Delaunay triangulation to the comparison of Section 4, on a model where all three methods complete. @fig:performance times the two structured methods across both time horizon (backward-induction cycles) and grid resolution.[^benchmark-impl] Apart from one small $20 \times 15$ grid, the grid sweep holds fifty health nodes and scales the asset axis from $44$ to $176$ points; the cycle sweep stops at fifteen cycles. Denser health grids or twenty cycles make the endogenous health coordinate non-monotone between the two lowest health nodes, which violates IPM, a condition ENGINE requires. On this model the comparison isolates the interpolation method cleanly, because the sequential and simultaneous solvers produce numerically identical policies (the maximum absolute difference in consumption is $3 \times 10^{-13}$). EGM$^n$ with ENGINE solves $2.8$ to $4.7$ times faster than the simultaneous solver with the curvilinear method across both sweeps. A third configuration that swaps only the interpolant on the simultaneous solver reproduces most of that gap, so the interpolation method, not the decomposition, accounts for the bulk of the speedup.

```{figure} ../figures/health_model_summary.*
:name: fig:performance
:align: center

Solve time and speedup on the health investment model, by backward-induction cycles at a $44 \times 50$ grid (a, b) and by grid size at ten cycles (c, d). Annotations give the speedup of EGM$^n$ with ENGINE over the simultaneous solver with the curvilinear method of {cite:t}`White2015`. Bands are 95 percent confidence intervals of the mean over ten timed runs.
```

[^benchmark-impl]: The two codebases are compiled alike (Section 4's implementation footnote) and run on the same hardware and software as Section 3's benchmarks, both recorded in the replication archive's README.

The converged policies separate the structured methods from the triangulation just as the timings do. Each interpolation method runs inside its own backward solve (holding the simultaneous solver fixed), and interpolation differences propagate through the solution. On a uniform $50 \times 40$ grid of off-mesh query points spanning $\mRat, \hRat \in [0.5, 25]$, with the curvilinear solution as the reference, ENGINE differs by a mean of $5.8 \times 10^{-5}$ in consumption (maximum $7.2 \times 10^{-3}$) and $1.8 \times 10^{-3}$ in health investment (maximum $4.8 \times 10^{-2}$); Delaunay triangulation differs by means of $3.5 \times 10^{-4}$ and $4.9 \times 10^{-3}$ (maxima $1.5 \times 10^{-1}$ and $3.5 \times 10^{-1}$).

@fig:health-interp-errors maps the consumption disagreement over the state space. ENGINE's differences from the curvilinear solution are small and nearly uniform, between $10^{-8}$ and $10^{-4}$ over most of the domain. However, the worst case, $7.2 \times 10^{-3}$, occurs in the few visible spots at the lowest market resources where the mesh cells are most warped, with fainter disagreement at the lowest health capital. The triangulation's differences are larger and spatially diffuse, following its simplex boundaries.

```{figure} ../figures/health_model_interpolation_errors.*
:name: fig:health-interp-errors
:align: center

Consumption disagreement with the curvilinear reference over the $(\mRat, \hRat)$ query plane, on a shared logarithmic color scale. (Left) ENGINE. (Right) Delaunay triangulation. Circles mark each panel's maximum.
```

### Accuracy against Measured Warp

The comparison above fixes the calibration and varies the query point. How do the methods behave as the mesh itself becomes more warped? I vary the two model parameters that bend the endogenous mesh hardest. Across the sweep, the health production elasticity takes values in $\{0.20, 0.35, 0.50\}$, controlling the curvature of the investment inversion, and the maximum death probability $\DiePrb_t$, held constant across ages in every sweep calibration, takes values in $\{0.12, 0.50, 0.82\}$, scaling the survival term in $\wFunc_t^{\HRat}$. I solve each of the nine calibrations for five periods on a smaller grid under the protocol above (each interpolant inside its own backward solve, with the simultaneous solver held fixed). After every solve I test IPM slice by slice and measure the warp by the empirical constant $\hat{\alpha}$ of Section 4, the maximum bracket displacement between adjacent health slices.

Accuracy is scored against the model's own optimality conditions rather than against any reference solution. At three thousand off-mesh query points I rebuild the post-decision marginals $\wFunc^{\aRat}$ and $\wFunc^{\HRat}$ by quadrature through the true continuation solution and evaluate both first-order conditions. The consumption Euler residual uses implied consumption capped at the $\aRat \geq 0$ budget bound, so constrained queries are scored rather than discarded. The investment margin $\gFunc'(\nRat)\, \wFunc^{\HRat} = \util'(\cRat)$ holds at interior investment whether or not the liquidity constraint binds, since the Inada condition on $\gFunc$ keeps investment interior. Queries where an interpolated policy violates feasibility, whether through the budget or through a negative control, are repaired minimally for the quadrature and counted, none of them dropped.

Every warp-sweep mesh the two structured methods produce satisfies IPM. Yet across those meshes $\hat{\alpha}$ ranges from $10$ to $16$, five to eight times ENGINE's default search margin of two. A mismatch in grid spacing produces these values: twenty health nodes span $[0.01, 50]$ whereas the asset axis is finely log-spaced, so a query's bracket can cross many asset cells between adjacent health slices. Without the containment fallback of Section 4, the default-margin search misses brackets on this sweep without raising an error, producing budget-infeasible interpolated policies and inflated extreme residuals. Setting the margin to the measured constant removes every violation on the sweep's declared finite query set at no measurable time cost. All sweep solves use that matched margin. The archived implementation also carries the fallback, which runs a full row search when a windowed bracket misses the query.

Every fold on the structured meshes is a sliver cell at the lowest post-decision health node. I start the health grid at $\HRat_t = 0.01$. The HARK implementation of this model starts it at zero, where the investment inversion at zero assets divides one infinite marginal value by another. At $\HRat_t = 0.01$, however, the investment the inversion assigns, even at zero assets, produces more health than the node holds, so every inverted state on that row has negative health and lies outside the state space. Whether a cell on that row folds depends on the calibration, so I count the folds directly. The final period, inverted on the analytic terminal solution with no interpolation, produces the same slivers. Evaluation points reach the cells only at $(\HealthProdExp,\DiePrb_t) = (0.35, 0.82)$ and $(0.50, 0.82)$. There, up to $0.45$ percent of continuation points fall in folded cells, and at $(0.50, 0.82)$ so do $0.37$ percent of final queries for the curvilinear method and $0.40$ percent for ENGINE. Although ENGINE's fold-free requirement fails in those cells, the reported residuals include every query inside one.

The triangulation's own meshes violate IPM at five of the nine calibrations, the three with the lowest production elasticity ($0.20$) and the other two with the highest mortality ($0.82$). At the remaining four the triangulation's $\hat{\alpha}$ is $12$ or $13$, against $10$ to $12$ for the structured meshes. In each of the five, between $82$ and $120$ of the mesh's $820$ nodes are undefined. The continuation queries of the highest health node fall above the convex hull of the next period's mesh, where the boundary triangles stretch to outlying mesh nodes, and linear extrapolation from those triangles returns values outside the admissible range, such as negative consumption. The first-order conditions are then undefined for that health column, whose nodes drop out of the next triangulation. As the hull shrinks, the undefined nodes spread one period back at a time. By contrast, the two structured methods extrapolate along the mesh's own index lines; every node of their meshes is defined at all nine calibrations.

```{figure} ../figures/health_warp_sweep.*
:name: fig:health-warp-sweep
:align: center

Consumption Euler residuals at off-mesh queries (median, 95th percentile, and maximum) against the measured bracket-displacement constant $\hat{\alpha}$ of each solve's own endogenous mesh, across nine calibrations varying the production elasticity and the maximum death probability. Markers are horizontally jittered for legibility, and the three panels use different vertical scales; the triangulation's five IPM-violating meshes have no defined $\hat{\alpha}$ and are absent from the panels.
```

@fig:health-warp-sweep reports the consumption residuals against this measured warp. Median accuracy, between $3 \times 10^{-4}$ and $3.4 \times 10^{-3}$ across the sweep, moves by an order of magnitude with the calibration but by less than a factor of two across methods at any single calibration. The interpolant therefore matters far less for the median than the calibration's warp does at this grid resolution.

The triangulation's maximum residual sits between $0.25$ and $0.66$ at every calibration, including the five IPM-violating meshes the figure cannot place, repeating the simplex-boundary artifacts in @fig:health-interp-errors. ENGINE has both a smaller 95th-percentile and a smaller maximum consumption residual than the curvilinear method in exactly three calibrations: $(\HealthProdExp,\DiePrb_t)=(0.35,0.82),(0.50,0.50),(0.50,0.82)$. At the last of these, its two consumption statistics, $0.019$ and $0.20$, are below the curvilinear method's $0.033$ and $0.41$, and its maximum investment residual falls from $0.60$ to $0.15$. The constant $\hat{\alpha}$ alone does not rank the tails: among the three calibrations where ENGINE's mesh has $\hat{\alpha}=16$, ENGINE's maximum consumption residual is higher at $(0.20,0.82)$, lower at $(0.35,0.82)$, and roughly half as large only at $(0.50,0.82)$. Across the full sweep, ENGINE's 95th percentile is smaller at seven of nine calibrations.

Every solve in the sweep completes. I compare timings only on the larger grids of @fig:performance.

(appendix-interp-loop)=

## The Full-Solve Interpolation Diagnostic

This appendix records the protocol behind Section 4's full-solve-time comparison of ENGINE with the sector-walking curvilinear method of {cite:t}`White2015`. [Online Appendix A](#appendix-online) adds the Delaunay triangulation of {cite:t}`Ludwig2018`.

I solve the Section 3 household block with each interpolator as the Stage-0 realign step, holding every other component fixed. Productivity follows the same three-state Rouwenhorst chain in every solve, with persistence $0.966$ and unconditional standard deviation $0.92$ for log productivity. Each iteration starts from the same marginal-value initialization and stops when the larger normalized sup-norm change in the beginning-of-period marginal values $v^{0}_{\aRat}$ and $v^{0}_{\bRat}$ (the arrays `Va` and `Vb` of the replication code) falls below $10^{-9}$, subject to a cap of $8{,}000$ backward steps; the normalization for either marginal is one plus the sup norm of its previous iterate. At coarse ($24 \times 20$), medium ($48 \times 40$), and fine ($96 \times 80$) grids, both methods use linear interpolation and sixteen Numba threads. Solve time is the best of three repetitions after a separate warm-up, so compilation is excluded. The residual compares consumption with the minimum of the Euler-implied and binding-constraint levels at off-grid queries, so no constraint-bound query is excluded. The queries are the axis midpoints below 85 percent of each grid maximum, crossed over both states and every income state.

Both solves use the same regular-grid Euler inversion. On the endogenous mesh, Stage 1 selects $\bRat'$, inverts to $\cRat=(\wFunc^{\bRat'})^{-1/\CRRA}$, and obtains $\lRat=\bRat'+\cRat$. On the regular pre-decision grid the solver again inverts the Euler equation on the interpolated continuation marginal, rather than interpolating the kinked consumption policy. With $\aRat'$ fixed, the Stage 1 policy supplies the evaluation point. One application of the inversion matches NEGM and the curvilinear interpolant off-grid, whereas a second, diagnostic application lowers on-grid Euler residuals to $10^{-6}$ to $10^{-7}$ without changing off-grid accuracy.

The grid axes are the model's own log-spaced asset grids, which concentrate points near the borrowing constraints. The diagnostic therefore exercises the mesh conditions on strongly non-uniform spacing by construction rather than on uniform grids, which are probably the most favorable to them. Each method runs with its native boundary behavior, so neither receives an out-of-hull correction. To measure how often that behavior is used, I classify every regular Stage-0 target at the converged fixed point as inside or outside the polygon bounded by the verified fold-free bilinear mesh. @tab-extrap-frequency-abrs reports the pooled count over all three income states at each grid. The replication CSV also reports each income state separately. Boundary points count as interpolation, so the reported outside share measures boundary-slope extrapolation rather than contact with the mesh boundary.

```{include} ../figures/tab_extrap_frequency_abrs.md
```

Only one target lies outside the mesh at every resolution. Because the number of targets grows with the grid, the outside share falls from $0.069$ to $0.004$ percent. This is a fixed-point result only; the iteration path was not recorded.

ENGINE's median $\log_{10}$ robust Euler error falls from $-1.79$ to $-2.47$ to $-3.14$, about fivefold per grid doubling. The 95th percentile remains four to six times the median, and the fine-grid worst-query error is below $1$ percent. Both structured interpolants reach these levels on this metric, so linear interpolation at these grid densities probably sets them on this model, in which each of the two stacked first-order conditions uses marginal values interpolated at off-grid midpoints. Inside the 85-percent interior query set, both passes bracket rather than extrapolate. In the complementary high-wealth band, the worst residual improves monotonically from $-1.08$ to $-1.69$ to $-2.32$ in $\log_{10}$ units. The band median, however, is not monotone, moving from $-2.24$ to $-1.74$ before falling to $-2.75$. At every resolution the band maximum is no larger than the interior maximum to within a fraction of a percent, with the two methods within one percent of each other.

With a $192 \times 160$ solution as the benchmark, median consumption-policy error falls from $9.3$ to $3.7$ to $0.9$ percent and the worst query from $18$ to $3.3$ percent. The median robust Euler residual is about six to twelve times smaller than the median policy error because the policy is locally optimal given its grid's continuation value. Extending both state-grid maxima by half barely moves the typical medium-grid policy (a median of $0.05$ percent), but the $4.3$ percent maximum rules out treating grid placement as irrelevant at every query. @tab-refinement-bounds-abrs reports both checks.

```{include} ../figures/tab_refinement_bounds_abrs.md
```

Section 4's pass-order reversal comparison uses this same protocol, with each pass order solved to convergence; the replication archive records its results at every grid.

The component table (@tab-components-abrs) assigns about one percent of serial time to expectations, a little over one third to the two stage inversions, five percent to ENGINE, and the remainder to the regular-grid Euler inversion and envelope updates. Components are timed on one thread at the Section 3 grids with the full-solve diagnostic's code, whose ENGINE configuration reproduces the paper's EGM$^n$ solver exactly. Summed, they track Section 3's timings within two percent at every grid.

```{include} ../figures/tab_components_abrs.md
```

The {cite:t}`Auclert2021` block also has a native hybrid NEGM/G2EGM solver, at the version pinned in the replication archive. At matched grids and tolerance (@tab-ssj-reference), EGM$^n$ uses fewer iterations ($395$ to $424$ versus $464$ to $478$) and has off-grid robust Euler medians within fifty percent of the native solver's, smaller at the smallest grid and larger at the two finer grids. The native solver's distributed array code runs $3.0$ to $4.6$ times faster than one-thread EGM$^n$, while sixteen-thread EGM$^n$ is about five percent faster on the coarse grid and faster by factors of $1.4$ and $1.7$ on the two finer grids. Its EGM$^n$ times, from a separate run of the configuration in @tab-timing-negm, lie within three percent of that table's. The native benchmark does not isolate the method. For that comparison, Section 3 uses NEGM with matched threading and shared components.

```{include} ../figures/tab_ssj_reference.md
```
