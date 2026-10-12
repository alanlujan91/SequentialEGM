
(method)=

# The Sequential Endogenous Grid Method

A household that simultaneously chooses consumption and labor supply presents the joint solver with a two-dimensional search at every state space point. But when decisions are separable in stages, though not necessarily globally, sequential decomposition lets EGM inversion solve each stage efficiently.

## Problem Setup

I demonstrate EGM$^n$ on a consumption-labor model in which the household chooses leisure and consumption under a risky wage. It is the smallest example that shows the chaining: two simultaneous decisions, a sequential ordering in which each stage sheds a state variable, and an EGM inversion at both stages, only one of which is the consumption margin EGM was built for. The household maximizes the present discounted lifetime utility of consumption and leisure over a finite horizon of $T+1$ periods, where $t = 0$ denotes the beginning of the planning horizon (the first period of economic life) and $t = T$ the terminal period:

\begin{equation}
    \VFunc_0(\aRat_{0}, \tShkEmp_0) = \max \Ex_{0}
    \left[ \sum_{t = 0}^{T} \DiscFac^{t} \utilFunc(\cRat_{t}, \leisure_{t})  \right],
\end{equation}

where $\VFunc_0$ is the lifetime value function at the start of period zero, $\aRat_{0}$ is the stock of liquid assets the household starts with, $\tShkEmp_0$ is the transitory wage shock, and $\Ex_0$ denotes the expectation conditional on information available at the start of the planning horizon.

Period utility is additively separable in consumption and leisure:

\begin{equation}
    \utilFunc(\cRat, \leisure) = \util(\cRat) + \hFunc(\leisure) = \frac{\cRat^{1-\CRRA}}{1-\CRRA} + \labShare \cdot \frac{\leisure^{1-\leiShare}}{1-\leiShare},
\end{equation}

where $\CRRA > 0$ is the coefficient of relative risk aversion, $\leiShare > 0$ determines the curvature of leisure utility (and thus the Frisch elasticity of labor supply) and $\labShare > 0$ scales leisure utility. I chose additive separability because it gives the leisure margin a marginal utility of its own, and therefore something to invert. The household solves

\begin{equation}
    \begin{split}
        \vFunc_{t}(\aRat_{t}, \tShkEmp_{t}) & = \max_{\cRat_{t} \geq 0, \, \leisure_{t} \in [0,1]}
        \utilFunc(\cRat_{t}, \leisure_{t}) + \DiscFac \Ex_{t} \left[ \vFunc_{t+1} (\aRat_{t+1}, \tShkEmp_{t+1}) \right] \\
        & \text{subject to} \\
        \labor_{t} = 1 - \leisure_{t}, \quad & \mRat_{t} = \Rfree \aRat_{t} + \tShkEmp_{t}\labor_{t}, \quad \aRat_{t+1} = \mRat_{t} - \cRat_{t},
    \end{split}
\end{equation}

where $\DiscFac$ is the discount factor and $\Rfree$ is the risk-free gross return. The terminal condition is $\vFunc_{T+1} \equiv 0$, so the household in the final period $T$ consumes all remaining resources; leisure then solves the static first-order condition (or its corners). Throughout, I assume standard constraint qualifications hold, so that interior solutions satisfy first-order conditions.[^constraint-qual] The intermediate quantities the constraints define are market resources $\mRat_{t}$ (incoming assets with their return, plus labor income) and end-of-period liquid assets $\aRat_{t+1}$.

[^constraint-qual]: Specifically: (i) utility and value functions are twice continuously differentiable in the interior of the constraint set; (ii) the Inada conditions $\lim_{c\to 0} \util'(c) = \infty$ and $\lim_{c\to\infty} \util'(c) = 0$ hold, so consumption stays away from zero; and (iii) constraint sets are convex with non-empty interior. These conditions ensure first-order conditions are necessary for optimality at interior solutions. Strict concavity of the utility and value functions further ensures sufficiency: any point satisfying the first-order conditions is a global maximum.


Although the two decisions are economically simultaneous, the dependence structure permits sequential solution. The labor-leisure choice determines market resources, and given those resources the consumption-saving choice determines liquid assets.

The sequential decomposition starts the period with the labor-leisure problem. To distinguish value functions at different stages, I introduce stage superscripts: the original problem has $\vFunc_t \equiv v^0_t$, the value at the first decision stage (labor-leisure), and $v^i_t$ is the value after the decisions at stages $0, 1, \ldots, i-1$. At this stage, the household observes the liquid assets $\aRat_{t}$ it carries into the period and the wage offer $\tShkEmp_{t}$, choosing leisure to maximize the sum of current leisure utility and the continuation value from market resources $\mRat_{t}$:

\begin{equation}
    v^{0}_{t}(\aRat_{t}, \tShkEmp_{t}) = \max_{\leisure_{t} \in [0, 1]}
    \hFunc(\leisure_{t}) + v^{1}_{t} (\mRat_{t}), \qquad
    \labor_{t} = 1 - \leisure_{t}, \quad
    \mRat_{t} = \Rfree \aRat_{t} + \tShkEmp_{t}\labor_{t}.
\end{equation}

Once market resources are realized, the pure consumption-saving problem determines how to allocate $\mRat_{t}$ between current consumption and liquid assets. The state space shrinks to a single dimension because the wage offer no longer matters:

\begin{equation}
    v^{1}_{t}(\mRat_{t}) = \max_{\cRat_{t} \in [0, \mRat_{t}]} \util(\cRat_{t}) + v^{2}_{t}(\aRat_{t+1}), \qquad
    \aRat_{t+1} = \mRat_{t} - \cRat_{t}.
\end{equation}

No choice remains after consumption, so the period closes with a post-decision value: the discounted expectation of next period's problem, taken over the wage draw the household does not yet know,

\begin{equation}
    v^{2}_{t}(\aRat_{t+1}) = \DiscFac \Ex_{t} \left[
        v^{0}_{t+1}(\aRat_{t+1}, \tShkEmp_{t+1}) \right].
\end{equation}

The sequential formulation follows the nested approaches of {cite:t}`Clausen2020` and {cite:t}`Druedahl2021` but composes EGM inversions without embedding optimization. Each choice is self-contained in a subproblem, with the structure chosen to minimize state variables at each stage. The sequential formulation is mathematically equivalent to the original joint problem by Bellman's principle of optimality {cite:p}`Bellman1957`: the expectation operator appears only in the final subproblem, so no uncertainty resolves between stages and every decision uses the same information set. Were shocks to resolve between stages, households would gain information unavailable in the original formulation, changing the problem's solution.

Limiting cases confirm that the decomposition nests simpler models. When $\labShare \to 0$, leisure utility vanishes and the labor-leisure stage degenerates: the household works full time regardless of wages, recovering the standard consumption-saving EGM of {cite:t}`Carroll2006`. When $\DiscFac \to 0$, the fully impatient household consumes all resources immediately, recovering the static problem.

## Sequential Solution

EGM$^n$ inverts both first-order conditions of the period in closed form, without grid search or root-finding, applying the standard EGM inversion to the consumption Euler equation and an analogous one to the leisure condition.

The consumption-saving stage is the standard EGM of {cite:t}`Carroll2006`. I take it as given and pause only to fix notation: bracketed variables ($\aMat$, $\mMat$) denote exogenous grids, gothic letters ($\cEndFunc$, $\mEndFunc$) denote endogenous quantities built by inverting a first-order condition, and primes are ordinary derivatives. Inverting the Euler equation $\util'(\cRat_t) = (v^{2}_{t})'(\aRat_{t+1})$ for consumption, which is possible because the marginal-utility map $\util'$ is strictly decreasing and hence one-to-one, gives consumption on the exogenous post-decision grid, and the budget identity returns the endogenous grid of market resources:

\begin{equation}
    \cEndFunc_{t}(\aMat) = (\util')^{-1}\left( (v^{2}_{t})'(\aMat) \right),
    \qquad
    \mEndFunc_{t}(\aMat) = \cEndFunc_{t}(\aMat) + \aMat.
\end{equation}

The pair $(\mEndFunc_{t}, \cEndFunc_{t})$ supports an interpolating consumption function for market-resources values off the grid. Through the envelope condition $(v^{1}_{t})'(\mRat_{t}) = \util'(\cRat_{t})$, the same pair gives the labor-leisure stage the marginal value it needs without additional computation.

The leisure choice has a first-order condition of its own. Substituting $\labor_t = 1 - \leisure_t$ and $\mRat_t = \Rfree \aRat_t + \tShkEmp_t \labor_t$ puts market resources inside $v^1_t$, so the whole trade-off between leisure and labor income runs through that one function:

\begin{equation}
    \hFunc'(\leisure_{t}) = (v^{1}_{t})'(\mRat_{t}) \tShkEmp_{t}.
\end{equation}

This equates the marginal utility of leisure to the opportunity cost of forgone labor income $\tShkEmp_t$, scaled by the marginal utility of consumption $(v^1_t)'(\mRat_t) = \util'(\cRat_t)$.

The marginal utility of leisure inverts in closed form,[^leisure-inverse] so on an exogenous grid of market resources $\mMat$ and wage shocks $\tShkMat$ leisure follows directly:

\begin{equation}
    \zEndFunc_{t}(\mMat, \tShkMat) = (\hFunc')^{-1}\left(
    (v^{1}_{t})'(\mMat) \tShkMat \right).
\end{equation}

Nothing in that step is special to consumption. The inverse marginal utility of leisure $(\hFunc')^{-1}$ carries an exogenous grid of market resources and wages to an endogenous grid of incoming assets, exactly as $(\util')^{-1}$ carries post-decision assets to market resources one stage later. EGM$^n$ chains the two.

Households with high market resources $\mRat_{t}$ and low wage offers $\tShkEmp_{t}$, however, may find the unconstrained optimum violates the feasibility constraint $\leisure_t \in [0,1]$. When this occurs, projecting the solution onto the constraint boundary defines the constrained optimal function $\hat{\zEndFunc}_{t}(\mMat, \tShkMat)$ as

\begin{equation}
    \hat{\zEndFunc}_{t}(\mMat, \tShkMat) = \max \left\{ \min \left\{ \zEndFunc_{t}(\mMat, \tShkMat), 1 \right\}, 0 \right\}.
\end{equation}

The projection keeps leisure within the time endowment; in practice only the upper bound binds.[^corner-solution] Interpolation handles the resulting kink by sorting the endogenous grid and fitting piecewise segments on either side of it.

[^leisure-inverse]: The marginal utility of leisure is $\hFunc'(\leisure) = \labShare\,\leisure^{-\leiShare}$, so its inverse at any positive marginal-utility value $\mu$ is $(\hFunc')^{-1}(\mu) = (\mu/\labShare)^{-1/\leiShare}$.

[^corner-solution]: With the functional form $\hFunc'(\leisure) = \labShare\,\leisure^{-\leiShare}$, the Inada condition $\lim_{\leisure \to 0} \hFunc'(\leisure) = \infty$ ensures the lower bound $\leisure_t = 0$ never binds for finite wages and marginal values. At the upper bound $\leisure_t = 1$, the household chooses full leisure when $\hFunc'(1) = \labShare \geq (v^{1}_t)'(\mRat_t)\tShkEmp_t$, which occurs for sufficiently low wages or high wealth. A zero wage draw carries no labor income at the margin, so there is no interior condition to invert and the projection assigns the full-leisure corner directly.

Labor follows as $1 - \hat{\zEndFunc}_{t}(\mRat_{t}, \tShkEmp_{t})$, and for each $\tShkEmp_{t}$ and $\mRat_{t}$ on the exogenous grid the endogenous grid of incoming assets is $\aEndFunc_{t}(\mRat_{t}, \tShkEmp_{t}) = \left( \mRat_{t} - \tShkEmp_{t}\bigl(1 - \hat{\zEndFunc}_{t}(\mRat_{t}, \tShkEmp_{t})\bigr) \right) / \Rfree$.

By the envelope theorem, the marginal value the previous period needs is $\partial v^{0}_{t}/\partial \aRat = \Rfree (v^{1}_{t})'(\mRat_{t})$, which completes the backward step. At interior solutions this marginal value equals $\Rfree \hFunc'(\leisure_{t})/\tShkEmp_{t}$. At a corner, however, only the first equality survives, because a Kuhn-Tucker inequality replaces the first-order condition.

The resulting endogenous grid for the labor-leisure problem is curvilinear rather than rectilinear.[^cgi-pedagogical]

[^cgi-pedagogical]: Simpler interpolation methods would suffice for the labor-leisure problem, since the grid warps along only one dimension. The replication code interpolates that stage by nested one-dimensional linear interpolation, which is enough here; the two-dimensional curvilinear meshes of [Section 3](#multdim-abrs) are where ENGINE is required.

```{figure} ../figures/consumption_labor_solution.*
:name: fig:consumption-labor
:align: center

Sequential decomposition produces well-behaved policy functions for the consumption-labor model ($T = 30$, parameters in footnote). Panel (a) shows the curvilinear endogenous grid from the labor-leisure stage at $t=0$ on a log-scaled asset axis, with points colored by optimal labor supply and the two mesh-line families identified in the legend. Panel (b) shows the consumption functions; it omits the terminal period and the final decision period, where the approaching horizon dominates the policy. Panel (c) shows that lower wages induce more leisure at $t=0$, since the opportunity cost of not working falls, with the unemployed household ($\tShkEmp = 0$) at full leisure throughout.
```

[^consumption-labor-params]: Parameters are $\CRRA = 5$ and $\DiscFac = 0.9$, with HARK's defaults $\Rfree = 1.03$ and a survival probability of $0.98$ that multiplies the discount factor. Wage shocks are lognormal with standard deviation $0.1$ and a five percent probability of a zero wage. Labor-specific parameters are $\leiShare = 2$ and $\labShare = 16$.

@fig:consumption-labor collects the full solution; no numerical optimization produced any of the policies it shows.[^consumption-labor-params] In the first period, the marginal propensity to consume falls from about $0.48$ at the borrowing constraint to about $0.024$ near $\mRat_t = 4.4$, then rises slowly to about $0.06$ at higher wealth. The slow rise plausibly mirrors the decline in the household's expected future labor, which reaches zero near $\mRat_t = 18$, shortly before the propensity levels off near $\mRat_t = 20$. Below $\mRat_t = 18$, saving most of an extra unit lets the household work less later in life. The labor-leisure inversion already produces a curvilinear grid, the simplest case of the meshes [Section 4](#multinterp) interpolates. The two-asset problem of [Section 3](#multdim-abrs), where two state variables persist across stages, provides the more demanding test.
