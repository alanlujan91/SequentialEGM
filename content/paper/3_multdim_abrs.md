
(multdim-abrs)=

# A Two-Asset Liquid-Illiquid Household

The consumption-labor problem in [Section 2](#method) carries at most one post-decision state variable per stage. The canonical stress test for a sequential EGM step is the two-asset household block at the center of the heterogeneous-agent New Keynesian (HANK) literature {cite:p}`Kaplan2018,Auclert2021`: a household holding liquid wealth alongside an illiquid account, moving funds between them at a convex adjustment cost. Two continuous state variables persist across the EGM stages, so both coordinates of the endogenous grid warp. The same structure, a second persistent stock alongside liquid wealth, appears in durable goods, human capital, and health investment problems; [Online Appendix A](#appendix-online) develops the health investment model of {cite:t}`Ludwig2018` and {cite:t}`White2015` as a second worked example. I solve the exact household block of {cite:t}`Auclert2021` at its published calibration, so that the benchmark runs on a model the macro audience already uses at scale. I treat only the household block and leave the New Keynesian aggregate machinery that closes the HANK economy to the literature cited above. Solver speed compounds in that setting: a general-equilibrium iteration can solve the household block thousands of times, each pass consuming the policies on the regular state grid, where the non-stochastic histogram method of {cite:t}`Young2010` carries the wealth distribution forward.

Nested methods conventionally handle the illiquid choice with a numerical first-order-condition search embedded in the backward step, exactly the numerical solve EGM$^n$ is designed to remove; the decomposition below replaces it with a second closed-form inversion. Timing the two solvers measures what that substitution saves, net of the regrid and the regular-grid Euler inversion it adds.

## The Household Problem

A household enters the period with state $(\bRat, \aRat, e)$, where $\bRat$ is incoming liquid wealth, $\aRat$ is the incoming illiquid balance, and $e$ is idiosyncratic labor productivity following a discrete Markov chain with transition matrix $\Pi(e' \mid e)$. The letters follow the two-asset convention of the heterogeneous-agent literature. Following the {cite:t}`Auclert2021` timing convention, $\aRat$ and $\bRat$ are post-decision balances: the choices this period, written $\aRat'$ and $\bRat'$, become next period's state. Both balances pay out their gross returns, $\Risky = 1 + r_a$ on illiquid and $\Rliq = 1 + r_b$ on liquid, labor income $y(e)$ is received, and market resources are $\mRat = \Rliq \bRat + \Risky \aRat + y(e)$. The household chooses consumption $\cRat \geq 0$ together with new balances $\bRat' \geq 0$ and $\aRat' \geq 0$ subject to the single within-period budget

\begin{equation}
    \cRat + \bRat' + \aRat' + \Psi(\aRat', \aRat) = \mRat.
\end{equation}

The illiquid asset earns the higher return ($r_a > r_b$), but moving funds into or out of it costs $\Psi$. Adjustment costs take the quadratic form of {cite:t}`Auclert2021`,

\begin{equation}
    \Psi(\aRat', \aRat) = \frac{\chi_1}{2} \left( \frac{\aRat' - \Risky \aRat}{\Risky \aRat + \chi_0} \right)^{2} \left( \Risky \aRat + \chi_0 \right),
\end{equation}

with offset $\chi_0 > 0$ and scale $\chi_1 > 0$. The cost vanishes when the household reinvests the illiquid interest ($\aRat' = \Risky \aRat$), is strictly convex in $\aRat'$, and scales with the post-return illiquid stock $\Risky \aRat + \chi_0$, so that even small illiquid positions are costly to deplete. The recursive problem is

\begin{equation}
    \vFunc(\bRat, \aRat, e) = \max_{\cRat, \bRat', \aRat'} \util(\cRat) + \DiscFac \, \Ex\left[ \vFunc(\bRat', \aRat', e') \,\big|\, e \right], \qquad
    \util(\cRat) =
    \begin{cases}
        \cRat^{1-\CRRA}/(1-\CRRA), & \CRRA \neq 1, \\
        \log \cRat, & \CRRA = 1,
    \end{cases}
\end{equation}

subject to the budget and the non-negativity constraints. Unlike the value-based curvilinear EGM of {cite:t}`White2015`, the inversions below need only positive risk aversion, $\CRRA > 0$, because each inverts a strictly positive marginal value rather than a utility level.

In one familiar limit, $\chi_1 \to \infty$, the illiquid balance moves only at the gross return ($\aRat' \to \Risky \aRat$) and the problem reduces to a one-asset consumption-saving model in $(\bRat, e)$. A fixed cost paid on any adjustment ($\aRat' \neq \Risky \aRat$) would generate a region of inaction and a non-convexity; that case leaves the smooth-EGM class and belongs to the G2EGM upper-envelope methods, the natural extension noted in the conclusion.

## Stage Decomposition and Backward Solution

The joint problem splits into three stages: an illiquid choice, a consumption choice, and a post-decision expectation. Under the stage convention of [Section 2](#method), $\vFunc \equiv v^0$ is the first decision stage, $v^1$ is the value entering consumption, and $\wFunc$ is the post-decision value, which carries its own letter here, following {cite:t}`Powell2011`, because both inversions read off it. With two persistent states, derivatives are partial: they appear as subscripts on stage values ($v^{1}_{\lRat} \equiv \partial v^{1}/\partial \lRat$) and, because $\wFunc$ carries no stage label, as superscripts on $\wFunc$ ($\wFunc^{\bRat'} \equiv \partial \wFunc/\partial \bRat'$). Productivity $e$ passes through every stage unchanged and resolves only in the expectation; every stage value therefore carries $e$ as a state. The expectation again appears only in the post-decision stage, so the equivalence argument of [Section 2](#method) applies unchanged. Where the consumption-labor stages shed a state variable apiece, here the state count stays at three throughout. Backward induction visits the stages in reverse, and both decision stages turn out to be EGM inversions, linked by envelope conditions.

The post-decision stage integrates over next period's productivity:

\begin{equation}
    \wFunc(\bRat', \aRat', e) = \DiscFac \sum_{e'} \Pi(e' \mid e) \, \vFunc(\bRat', \aRat', e').
\end{equation}

### Stage 1: The Consumption EGM with the Constrained Segment

Given the liquid resources $\lRat$ left after the illiquid choice, the consumption stage chooses $\cRat$:

\begin{equation}
    v^1(\lRat, \aRat', e) = \max_{\cRat \geq 0} \util(\cRat) + \wFunc(\bRat', \aRat', e), \qquad \bRat' = \lRat - \cRat \geq 0.
\end{equation}

The illiquid choice $\aRat'$ is already made when consumption is chosen, so at each fixed $\aRat'$ this stage is the consumption EGM of [Section 2](#method), run along the $\bRat'$ axis and resting on the same separable utility ([Section 5](#conditions)). The stage reads the post-decision marginals $\wFunc^{\bRat'}, \wFunc^{\aRat'}$ on the exogenous $(\bRat', \aRat')$ grid, inverts the Euler equation,

\begin{equation}
    \cEndFunc(\bRat', \aRat', e) = \left( \wFunc^{\bRat'}(\bRat', \aRat', e) \right)^{-1/\CRRA}, \qquad \lEndFunc = \bRat' + \cEndFunc,
\end{equation}

and returns $v^{1}_{\lRat}, v^{1}_{\aRat'}$ on a regular grid of liquid resources $\lRat$. The liquid borrowing constraint $\bRat' \geq 0$ is handled in the manner of {cite:t}`Carroll2006`, by prepending the single point $(\lRat = 0, \cRat = 0, \bRat' = 0)$ to the endogenous mesh. With $\bRat'$ starting at zero, the first bracket lies on the line $\cRat = \lRat$, and linear interpolation reproduces the constrained segment exactly. Through the envelope the illiquid marginal passes unchanged, $v^{1}_{\aRat'} = \wFunc^{\aRat'}$ evaluated at the Stage 1 optimum.

### Stage 0: The Illiquid EGM with Closed-Form Inversion

The illiquid stage, given $(\bRat, \aRat, e)$, chooses $\aRat'$ and converts market resources into liquid resources:

\begin{equation}
    v^0(\bRat, \aRat, e) = \max_{\aRat' \geq 0} v^1(\lRat, \aRat', e), \qquad \lRat = \mRat - \aRat' - \Psi(\aRat', \aRat).
\end{equation}

EGM$^n$ departs from NEGM at this stage, where the first-order condition for $\aRat'$ balances the marginal value of liquid against the marginal value of illiquid net of the adjustment-cost wedge:

\begin{equation}
    1 + \Psi_1(\aRat', \aRat) = \frac{v^{1}_{\aRat'}(\lRat, \aRat', e)}{v^{1}_{\lRat}(\lRat, \aRat', e)}, \qquad \Psi_1 \equiv \frac{\partial \Psi}{\partial \aRat'}.
\end{equation}

The right-hand side is read directly from Stage 1's regular-grid envelopes. A nested method searches over $\aRat'$, but EGM$^n$ holds the gridded choice $\aRat'$ fixed and recovers the incoming stock $\aRat$. Here the enabling structure is the invertible transition marginal of [Section 5](#conditions): the stage has no utility term of its own, and $\Psi_1$ is strictly decreasing in the unknown $\aRat$ because $\Psi_{12}<0$. For the quadratic cost, $\Psi_1(\aRat', \aRat) = \chi_1 u$, where $u \equiv (\aRat' - \Risky \aRat)/(\Risky \aRat + \chi_0)$ is the standardized adjustment wedge, and

$$\Psi_{11}=\frac{\chi_1}{\Risky\aRat+\chi_0}>0,\qquad
\Psi_{12}=-\frac{\chi_1\Risky(\aRat'+\chi_0)}{(\Risky\aRat+\chi_0)^2}<0,$$

where $\Psi_{ij}$ denotes the corresponding second partial derivative.

Define the wedge target $\tau \equiv v^{1}_{\aRat'}/v^{1}_{\lRat} - 1$; the inversion is

\begin{equation}
    u^* = \frac{\tau}{\chi_1}, \qquad \aEndFunc(\lRat, \aRat', e) = \frac{\aRat' - u^* \chi_0}{\Risky (1 + u^*)},
\end{equation}

a single multiplication and a ratio per grid point. The remaining endogenous quantities follow from the two accounting identities: endogenous market resources from the liquid-resources definition, $\mEndFunc = \lRat + \aRat' + \Psi(\aRat', \aEndFunc)$, and the endogenous liquid balance from the market-resources definition, $\bEndFunc = (\mEndFunc - \Risky \aEndFunc - y(e))/\Rliq$. There is no root-find and no iteration; the entire illiquid choice is closed form.[^psi2-envelope]

[^psi2-envelope]: The Stage 0 envelope returns $v^{0}_{\bRat} = \Rliq \, \cRat^{-\CRRA}$ and $v^{0}_{\aRat} = (\Risky - \Psi_2)\, \cRat^{-\CRRA}$, where $\Psi_2 = \partial \Psi/\partial \aRat = -\Risky\, \chi_1 u\,(1 + \tfrac{1}{2}u)$ collects the direct illiquid return and the indirect effect through the adjustment cost.

The inversion produces a curvilinear $(\bEndFunc,\aEndFunc)$ mesh while retaining the exogenous grid's adjacency. [Section 4](#multinterp) states the two geometric conditions ENGINE places on such a mesh, both of which I verify directly on every reported benchmark mesh. After ENGINE carries the illiquid choice to the regular state grid, the remaining policies follow by composition: the implied liquid resources pull the Stage 1 consumption function back to $(\bRat, \aRat, e)$, evaluated through the same Euler inversion, and the liquid choice is the budget residual (Online Appendix B). @fig:abrs_grids shows the result.[^abrs-grid-fig] The inversion bends the mesh most where the optimal adjustment is largest. Yet mesh lines of constant $\lRat$ and constant $a'$ never cross.

```{figure} ../figures/grid_transformation_abrs.*
:name: fig:abrs_grids
:align: center

Stage 0 grid transformation at the converged fixed point. (a) Exogenous grid of liquid resources $\lRat$ and illiquid choice $a'$. (b) Endogenous mesh of incoming states $(\bEndFunc, \aEndFunc)$, colored by consumption. Withdrawal-financed nodes with negative $\bEndFunc$ lie outside the state space and are never queried.
```

[^abrs-grid-fig]: The figure uses the benchmark calibration on a $20 \times 20$ state grid solved to tolerance $10^{-9}$, shown at the median of the three productivity states; the Stage 1 liquid-resources axis extends past the liquid state grid by the domain rule discussed below. Both axes are log-scaled away from zero because the underlying grids are log-spaced.

### Boundaries: Two Economic Edges and One Grid-Design Rule

The inversions handle every interior solution; the boundaries come in two kinds. The *economic* boundaries are the non-negativity constraints, neither of which costs a subproblem. The liquid edge ($\bRat' = 0$) is the consumption EGM's own borrowing constraint, handled inside Stage 1 by the prepended constraint point. The illiquid edge ($\aRat' = 0$) reduces the cell to a Stage 1 evaluation at the implied liquid resources $\lRat = \mRat - \Psi(0, \aRat)$, so the corner policy is read off Stage 1's own $\aRat' = 0$ slice, and each cell picks between the interior inversion and the corner by the sign of the unconstrained policy. With strictly concave utility and a convex adjustment cost, the stage policies are single-valued and continuous, so no discrete jumps arise and the solver never requires an upper envelope.

The *grid* boundary imposes a design rule with no analogue in one-block solvers. Sequential decomposition materializes the intermediate state $\lRat$ on a regular grid between the stages, and that grid must cover whatever the second inversion can request. Inherited liquid wealth does not bound it: a household rich in illiquid assets finances liquidity by withdrawing, and because the marginal value of illiquid wealth is non-negative the first-order condition bounds the withdrawal wedge by $u^* \geq -1/\chi_1$, so $\lRat$ reaches at most

\begin{equation}
    \overline{\lRat} = \Rliq \overline{\bRat} + \overline{y} + \frac{\Risky \overline{\aRat} + \chi_0}{2\chi_1},
\end{equation}

in closed form, where $\overline{\bRat}$ and $\overline{\aRat}$ are the largest nodes of the liquid and illiquid grids and $\overline{y}$ the highest income realization.

Truncating the $\lRat$-domain to the liquid state grid, the natural shortcut, strips mesh coverage at high illiquid wealth without warning and forces the realign step to extrapolate there. On this calibration the iteration then converges to a budget-infeasible pseudo-solution in the affected band, which a constraint-valid Euler diagnostic exposes at once ([Section 4](#multinterp)).[^lgrid-implementation]

[^lgrid-implementation]: The solver builds the $\lRat$-grid to this bound (a wider bound may be supplied when a calibration calls for one); with the rule in place, every realign query near the fixed point is interior. At the liquid edge, Euler-implied consumption is capped at the constraint-feasible maximum, and the liquid policy is its budget residual, so the delivered policies satisfy the within-period budget identity by construction.

## Comparison to Nested EGM

I compare the two solvers on total time to a converged household solution, at matched grids, matched tolerance ($10^{-9}$), and matched linear interpolation. All timings are net of one-time compilation cost, best of three repetitions from a warm start.[^matched-interp] @tab-timing-negm reports both solvers' times. The solvers share the consumption-stage EGM, the constrained segment, the post-decision integration, and the fixed-point iteration. NEGM solves the illiquid first-order condition directly on the regular state grid by a safeguarded bracketing iteration whose relative bracket tolerance must sit two orders below the outer norm, at $10^{-11}$. With the bracket loosened to $10^{-9}$, per-node root-find noise keeps the fixed-point iteration from meeting its own norm and the solve hits the iteration cap, so the comparator converges only with the tighter bracket (the archive's inner-tolerance table records the check). EGM$^n$ instead applies the closed-form inversion on its exogenous grid, then pays for a general ENGINE regrid and a regular-grid Euler inversion. The two backward steps are parallelized identically, so the comparison is matched at one thread and at sixteen.[^reproduce]

[^matched-interp]: The timing comparison measures the net effect of replacing NEGM's regular-grid root-find with the EGM$^n$ backward step, including the latter's extra Stage-0 regrid and regular-grid inversion. In the full-solve experiment of [Section 4](#multinterp) I hold the EGM$^n$ solver fixed and swap only that regrid to isolate ENGINE from the decomposition.

```{include} ../figures/tab_timing_negm.md
```

NEGM and EGM$^n$ converge in a comparable number of backward iterations (at $(40,60)$, $(80,120)$, and $(160,240)$, NEGM takes $473$, $481$, and $487$ iterations and EGM$^n$ takes $395$, $410$, and $424$). The timing gap comes from the illiquid step, where EGM$^n$ makes one closed-form evaluation and NEGM an iterated sequence of interpolated envelope evaluations. EGM$^n$ still wins after paying for the general Stage-0 ENGINE regrid and regular-grid Euler inversion that NEGM does not need, since NEGM's root-find returns the policy on the regular $(\bRat, \aRat)$ grid. Serially the factor is about thirteen at all three grids. At the largest, one serial NEGM solve takes just under three minutes against fourteen seconds for EGM$^n$. At matched sixteen threads the factor grows from about $3.7$ to $7.9$ with the grid size. The root-find carries enough arithmetic per node to scale almost linearly with threads, whereas the EGM$^n$ step is so cheap per node that at small grids its solve time is probably dominated by the fixed cost of coordinating the threads themselves. I expect the matched-thread ratio to keep climbing toward the serial factor at the grid sizes where computing time matters.

The two solvers also agree on the solved margins. Median relative distance between the converged policies stays below half a percent for both the liquid and the illiquid balance at every timing grid. At the $192\times160$ policy-benchmark grid the two asset margins agree to $0.16$ and $0.29$ percent. On the model's own liquid Euler equation, the comparison depends on where I take the residual. Off the grid, where both converged policies are interpolated and the residual puts the two methods on equal footing, EGM$^n$'s median is the smaller at six of the seven reported grids (three timing, three diagnostic, and the policy benchmark), trailing only at the $(80,120)$ timing grid, by a quarter of a percent. At the three diagnostic grids the medians differ by less than fifteen percent, and EGM$^n$'s advantage grows to factors of $1.4$ and $1.6$ at the two largest grids overall, the $(160,240)$ timing grid and the policy benchmark. On the regular grid, EGM$^n$'s residual runs one to two orders of magnitude below NEGM's, about $1.5\times 10^{-5}$ against $3.9\times 10^{-4}$ at $(40,60)$, because EGM$^n$ delivers regular-grid consumption as an Euler-inversion output where NEGM reads it off an interpolated policy (Online Appendix B).[^accuracy-negm]

[^accuracy-negm]: The script behind these readings and its output table are in the replication archive alongside the timing benchmark; both solvers run at the same tolerance, and the distances are medians over the regular $(\bRat, \aRat)$ grid and every income state.

[^reproduce]: The timing script and its output table are in the replication archive, whose README records the full benchmark environment and exact package versions (AMD Ryzen 9 5900X; Python 3.13, NumPy 2.3, Numba 0.62, SciPy 1.16); reported runs cap the thread count at sixteen. The archive's test suite validates that the solver's output is identical across thread counts (the backward step writes each grid cell independently, with no cross-cell reductions).

Each stage is itself a Bellman problem; every iteration of the backward step therefore also produces the marginal values $v^{1}_{\lRat}$ and $v^{1}_{\aRat'}$ of the consumption stage on a regular grid. Their ratio minus one is the wedge $\tau \equiv v^{1}_{\aRat'}/v^{1}_{\lRat} - 1$, the adjustment-cost shadow price of converting illiquid into liquid wealth. A modeler can read it off to answer questions that turn on this object (how the adjustment friction varies over the wealth distribution, where it binds), with no auxiliary computation. NEGM shares this property (its root-find runs on the same stage envelopes), but solvers that treat the period as a single block with two controls, the G2EGM family among them, deliver policies without ever constructing the intermediate-stage values. Recovering the same objects on the whole stage grid, beyond the optimal choices, requires re-deriving the stage decomposition that EGM$^n$ starts from.
