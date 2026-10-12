
(conditions)=

# When the Method Applies

When does EGM$^n$ apply? The consumption-labor problem ([Section 2](#method)) and two-asset problem ([Section 3](#multdim-abrs)) illustrated two enabling structures: separable utility functions and explicit inversion of a transition marginal followed by state recovery. In economic terms, separability means the marginal utility of one choice does not depend on the level of another, while the second structure recovers a choice or incoming state from how that choice changes the states carried forward. Below, *invertible transition marginal* is shorthand for this full marginal-inversion and state-recovery condition.

## Splitting the Problem into Subproblems

Decomposition begins by counting the problem's distinct controls. A problem with $n$ control variables might decompose into $n$ subproblems. Do not double-count: the consumption-saving choice ($\cRat + \aRat = \mRat$) is one decision, not two. Similarly, labor-leisure is a single choice despite involving two variables.

Each control variable requires a structure that permits EGM inversion. Two broad sufficient structures are separable utility and an invertible transition marginal in the shorthand just defined. More generally, an interior stage qualifies when its first-order condition has an explicitly evaluable, single-valued inverse, the target lies in its range, the transition identities recover the remaining endogenous state, and feasibility and curvature confirm the candidate as the unique optimum. Where a stage fails these conditions, it uses standard optimization, and the gains accrue only from the stages that invert.[^portfolio-fallback]

[^portfolio-fallback]: A risky portfolio share is a worked instance of the fallback. Appended to the consumption-labor model of [Section 2](#method) in the tradition of {cite:t}`Bodie1992`, the share stage carries no utility component of its own, and its first-order condition equates the expected product of the excess return and the marginal value of wealth to zero with the share inside the expectation, so no closed-form inversion exists. The implementation evaluates this condition on a one-dimensional grid of shares, linearly interpolates its first nonnegative-to-nonpositive crossing, and selects zero or one when the endpoint signs imply a corner.

The ordering of subproblems should shed state variables early. When subproblems affect disjoint state dimensions, the ordering is immaterial. When they share one, however, poor sequencing propagates unnecessary state variables through later stages. In the consumption-labor problem, placing labor-leisure first resolves the wage rate before the consumption stage, keeping that subproblem one-dimensional. Choosing consumption first would force the labor decision to track both liquid assets and wages, doubling its dimensionality.[^bad-ordering-example]

[^bad-ordering-example]: To see this concretely, placing consumption first makes its subproblem two-dimensional: $v^{0}(\aRat_t, \tShkEmp_t) = \max_{\cRat_t} \util(\cRat_t) + v^{1}(\tilde{\aRat}_t, \tShkEmp_t)$ with post-consumption balance $\tilde{\aRat}_t = \Rfree \aRat_t - \cRat_t \ge -\tShkEmp_t$, requiring interpolation on an $(\aRat_t, \tShkEmp_t)$ grid instead of on $\aRat_t$ alone. The labor-leisure subproblem would then carry the feasibility constraint through: $v^{1}(\tilde{\aRat}_t, \tShkEmp_t) = \max_{\leisure_t} \hFunc(\leisure_t) + v^{2}(\aRat_{t+1})$ subject to $0 \le \leisure_t \le 1$ and $\aRat_{t+1} = \tilde{\aRat}_t + \tShkEmp_t(1 - \leisure_t) \ge 0$.

## Formal Conditions

The formal statements below use uppercase calligraphic functions ($\VFunc$, $\WFunc$, $\UFunc$, $\TFunc$, $\GFunc$) for generic value, continuation, utility, within-stage transition, and next-period state functions, and $\Gamma$ for the feasible-choice correspondence, the set of controls the constraints permit at a given state, with generic state variables $\xRat$, $\yRat$, $\sRat$ in place of the model-specific variables: consumption $\cRat$, assets $\aRat$, and cash-on-hand $\mRat$ of preceding sections; the notation reference of the [Appendix](#appendix) collects the generic-stage symbols.

### Separable Utility

Once the problem is split into subproblems, the Endogenous Grid Method applies to each qualifying subproblem during backward iteration from the terminal period.

Consider a generic scalar subproblem with differentiable utility, transition, and continuation-value functions:

\begin{equation}
    \begin{split}
        \VFunc(\xRat) & = \max_{\aRat \in \Gamma(\xRat)} \UFunc(\xRat, \aRat) + \WFunc(\yRat) \\
        & \text{subject to} \\
        \yRat & = \TFunc(\xRat,\aRat),
    \end{split}
\end{equation}

where $\WFunc(\yRat) = \DiscFac \Ex[\VFunc_{+1}(\yRat)]$ is the continuation value.[^ez-egm] For an interior solution, the first-order condition is

\begin{equation}
    \frac{\partial \UFunc(\xRat, \aRat)}{\partial \aRat} + \WFunc'(\yRat) \frac{\partial \TFunc(\xRat,\aRat)}{\partial \aRat} = 0.
\end{equation}

[^ez-egm]: Time-additive expected utility is maintained throughout. Epstein-Zin preferences violate it, since the aggregator over current consumption and the certainty equivalent of continuation value is not additive; {cite:t}`Lujan2026` extends EGM to that case via a power transformation.

For a concave scalar stage with an interval constraint, a corner solution is obtained by projecting the interior candidate onto the feasible interval, as [Section 2](#method) demonstrates for leisure.

The question is when this first-order condition can be inverted to yield the optimal control directly. @prop-egm-utility gives the standard case used in [Section 2](#method): separable utility with a transition linear in consumption or labor.

```{prf:proposition} Separable Utility
:label: prop-egm-utility

Suppose $\UFunc(\xRat, \aRat) = \uFunc_{-}(\xRat) + \uFunc(\aRat)$ is additively separable, so $\phi(\aRat) \equiv \uFunc'(\aRat)$ depends only on the control. Assume $\phi^{-1}$ is single-valued and explicitly evaluable at the target, the recovered control is interior and feasible, and the full stage objective is strictly concave. If $\partial\TFunc/\partial\aRat=k\neq0$ depends only on quantities known before $\xRat$ is recovered, and if $\xRat\mapsto\TFunc(\xRat,\aRat)$ has an explicitly evaluable, single-valued inverse at each relevant $(\yRat,\aRat)$, the optimal control on an exogenous post-decision grid is

\begin{equation}
    \mathfrak{a}(\yMat) = \phi^{-1}
    \left( -k \cdot \WFunc'(\yMat) \right).
\end{equation}

The pre-decision state is the unique solution to $\TFunc(\mathfrak{x}(\yMat),\mathfrak{a}(\yMat))=\yMat$, written $\mathfrak{x}(\yMat)=[\TFunc(\mathord{\cdot},\mathfrak{a}(\yMat))]^{-1}(\yMat)$.
```

Each hypothesis does distinct work in the displayed formula. Because $\xRat$ is recovered endogenously, $\phi$ may not depend on $\xRat$. Without strict concavity the first-order condition can hold at a local minimum or at one of several stationary points, so the formula could return a candidate that is not the optimum, and nothing in the step would flag it. Without a single-valued inverse the step must select among branches, a selection the one-shot formula does not provide. If $k$ depended on $\xRat$, the right-hand side would contain the state the formula recovers, and the displayed inversion would not close.

Standard budget constraints in consumption-saving models satisfy the linearity assumption $\frac{\partial \TFunc}{\partial \aRat} = k$. For example, $\aRat = \mRat - \cRat$ implies $\frac{\partial \aRat}{\partial \cRat} = -1$, so $k = -1$. Similarly, the labor income constraint $\mRat = \Rfree \aRat + \tShkEmp \cdot \labor$ has $\frac{\partial \mRat}{\partial \labor} = \tShkEmp$. The wage shock may enter $k$ because it is observed before the stage is solved; the unrecovered endogenous state may not.

Computationally, the inversion replaces a numerical root-find with a single function evaluation at each grid point.

To see when the method fails, consider non-separable CES preferences (Cobb-Douglas within the period, so the elasticity of substitution is one) $\UFunc(\cRat, \leisure) = \frac{(\cRat^\xi \leisure^{1-\xi})^{1-\CRRA}}{1-\CRRA}$, where $\xi \in (0,1)$ is the weight on consumption. The first-order condition for consumption involves leisure: $\frac{\partial \UFunc}{\partial \cRat} = \xi \cRat^{\xi(1-\CRRA)-1} \leisure^{(1-\xi)(1-\CRRA)}$. Inverting for $\cRat$ requires knowing $\leisure$, and the symmetric condition for $\leisure$ requires knowing $\cRat$. The first-order conditions are coupled, so neither can be solved in isolation by EGM inversion, and the stages do not separate. Homotheticity, however, still allows a closed-form solution here: given the wage, the intratemporal condition makes leisure proportional to consumption, and substituting it back leaves a consumption marginal utility that inverts in closed form, the triangular case of {cite:t}`Iskhakov2015`. Without such a substitution, coupled conditions require a nested numerical solve (NEGM) or upper envelope techniques (G2EGM).[^ces-iegm]

[^ces-iegm]: The iEGM of {cite:t}`HallengreenJorgensenOlesen2024` would solve such a stage by numerical rather than analytical inversion of the marginal utility.

### Invertible Transition Marginals

Some subproblems have no within-period utility at all, only transitions affecting multiple post-decision states. The illiquid-choice stage of [Section 3](#multdim-abrs) is such a case, since moving funds between the accounts has no utility of its own and matters only through the two quantities it sets, the liquid resources left after the adjustment cost and the new illiquid balance. @prop-egm-transition shows that EGM still applies under the invertible-transition-marginal shorthand defined above, even without separable utility. Consider a problem with two endogenous state variables and two post-decision states, where the first transition may also carry the second state:

\begin{equation}
    \begin{split}
        \VFunc(\xRat_1, \xRat_2, \sRat) & = \max_{\aRat \in \Gamma(\xRat_1, \xRat_2, \sRat)} \WFunc(\yRat_1, \yRat_2, \sRat) \\
        & \text{subject to} \\
        \yRat_1 & = \TFunc_1(\xRat_1, \xRat_2, \aRat) \\
        \yRat_2 & = \TFunc_2(\xRat_2,\aRat),
    \end{split}
\end{equation}

where the differentiable continuation value is

\begin{equation}
    \WFunc(\yRat_1, \yRat_2, \sRat) = \DiscFac \Ex \left[ \VFunc_{+1}(\GFunc_1(\yRat_1, \sRat), \GFunc_2(\yRat_2, \sRat), \sRat') | \yRat_1, \yRat_2, \sRat \right].
\end{equation}

The first-order condition becomes

\begin{equation}
    \frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_1} \cdot \frac{\partial \TFunc_1(\xRat_1, \xRat_2, \aRat)}{\partial \aRat} + \frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_2} \cdot \frac{\partial \TFunc_2(\xRat_2,\aRat)}{\partial \aRat} = 0.
\end{equation}

```{prf:proposition} Invertible Transition Marginals
:label: prop-egm-transition

Suppose that, with post-decision states fixed, the first-order condition contains one unknown and has an explicitly evaluable, single-valued inverse whose target lies in its range. Suppose the transition identities then recover the remaining pre-decision quantities in closed form, and feasibility and curvature confirm the candidate as the unique optimum. The stage solves without root-finding. Two configurations deliver this structure.

**Case (i): off-grid control.** Suppose both transitions are additively separable in the control:
$$\TFunc_1(\xRat_1, \aRat) = f_1(\xRat_1) + k \cdot \aRat, \quad \TFunc_2(\xRat_2, \aRat) = f_2(\xRat_2) + \gFunc(\aRat),$$
where $k \neq 0$ is constant, $f_1$ and $f_2$ are invertible, $\gFunc'$ has an explicit single-valued inverse at the displayed target, and $\partial \WFunc / \partial \yRat_2 \neq 0$. Then the first-order condition yields

\begin{equation}
    \aRat = (\gFunc')^{-1}\left( -k \cdot \dfrac{\partial \WFunc(\yRat_1, \yRat_2, \sRat) / \partial \yRat_1}{\partial \WFunc(\yRat_1, \yRat_2, \sRat) / \partial \yRat_2} \right),
\end{equation}

and the pre-decision states follow by inverting $f_1$ and $f_2$.

**Case (ii): gridded control.** Suppose instead the control is itself the second post-decision state, and the first transition may couple the control with the not-yet-recovered state:
$$\TFunc_1(\xRat_1, \xRat_2, \aRat) = f_1(\xRat_1) + \kappa(\aRat, \xRat_2), \quad \TFunc_2(\xRat_2, \aRat) = \aRat,$$
where $f_1$ is invertible, $\partial \WFunc / \partial \yRat_1 \neq 0$, and the marginal $\kappa_{\aRat} \equiv \partial \kappa / \partial \aRat$ has an explicit single-valued inverse in $\xRat_2$ at the displayed target. Then

\begin{equation}
    \xRat_2 = \kappa_{\aRat}(\aRat, \cdot)^{-1}\left( - \dfrac{\partial \WFunc(\yRat_1, \yRat_2, \sRat) / \partial \yRat_2}{\partial \WFunc(\yRat_1, \yRat_2, \sRat) / \partial \yRat_1} \right), \qquad \xRat_1 = f_1^{-1}\big(\yRat_1 - \kappa(\aRat, \xRat_2)\big).
\end{equation}

The two configurations are sufficient, not exhaustive.[^g-monotone]
```

[^g-monotone]: Strict monotonicity ensures the inverse is well-defined in both cases: in case (i) $\gFunc'$ is strictly decreasing; in case (ii) the map $\aRat \mapsto \Psi_1(\aRat', \aRat) = \chi_1 (\aRat' - \Risky\aRat)/(\Risky\aRat + \chi_0)$, at fixed $\aRat'$, is strictly decreasing in the incoming stock ($\partial \Psi_1 / \partial \aRat = -\chi_1 \Risky (\aRat' + \chi_0)/(\Risky\aRat + \chi_0)^2 < 0$), with the inverse available in closed form as the single division $u^* = \tau/\chi_1$ followed by a linear solve for the stock.

In case (i), additive separability keeps unrecovered states out of the control derivative. Case (ii) permits state-control coupling because the control itself sits on the exogenous grid, leaving only the unrecovered state in the first-order condition.

The literature offers two neighboring standards for when EGM applies. {cite:t}`White2015` characterizes when EGM applies to the period problem as a whole, requiring that the endogenous transition factor through a post-decision state that absorbs the constraints (whether a choice was feasible can be read off the post-decision state alone), with the controls entering that state through a full-rank map. Under those conditions the expectation is computed once per gridpoint. But the first-order system they leave behind may still demand a multidimensional root-find. That residual root-find disappears under the "triangular" transition structure of {cite:t}`Iskhakov2015`. Case (i)'s additive separability is that structure, under which the system solves by substitution with no root-finding, though Triangular EGM requires the entire multidimensional first-order-condition system to be triangular. The stage conditions ask less than triangularity and more than the transition decomposition alone. Each stage's system must collapse to a scalar closed-form inversion. Meeting the stage requirement does not imply meeting the system-level one. In the two-asset application of [Section 3](#multdim-abrs) the stage conditions hold, but the full period first-order-condition system is not triangular.

The health-investment stage of [Online Appendix A](#appendix-online) is case (i); the illiquid stage of [Section 3](#multdim-abrs) is case (ii). ENGINE's separate geometric requirements are fold-freeness and index monotonicity, which [Section 4](#multinterp) tests cell by cell on every reported mesh. Neither EGM$^n$ nor ENGINE applies under nonconcave continuation, which breaks the curvature check that confirms the optimum and can fold the mesh. Under a discrete choice the policy is discontinuous, so the single-valued inversion the stage conditions require fails. The grids such a choice produces need an upper envelope rather than structured interpolation.

## Classification of Solver Classes

The four solver classes differ along three problem features identified by the EGM literature: whether each stage has additively separable utility, whether its transition marginal is invertible in the sense of @prop-egm-transition, and whether the stage has an occasionally binding constraint. @tab-solver-classification maps each (feature, solver) cell to the operation that solver applies: an inversion recovers the policy in closed form from the first-order condition, a nested root-find wraps an outer one-dimensional search around the next stage's solution, and an upper envelope adjudicates across candidate constraint regimes.

```{latex}
\begin{table}[htbp]
\centering
\caption{Solver class as a function of stage structure. The comparison covers Triangular EGM \citep{Iskhakov2015}, NEGM \citep{Druedahl2021}, G2EGM \citep{Druedahl2017}, and EGM$^n$. Each cell names the operation applied to a stage with the row's feature; the operations are defined in the text. An Invertible Transition Marginals stage has no within-period utility and couples post-decision states through a differentiable transition; Triangular EGM requires the full first-order-condition system to be triangular. ``Not developed'' means the published analysis assumes interior solutions, while EGM$^n$ selects the corner from the sign of the inverted policy (Section~3).}
\label{tab-solver-classification}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.20\linewidth}p{0.18\linewidth}p{0.15\linewidth}p{0.17\linewidth}p{0.18\linewidth}@{}}
\toprule
{\raggedright Stage feature\par} & {\raggedright Triangular EGM\par} & {\raggedright NEGM\par} & {\raggedright G2EGM\par} & {\raggedright EGM$^n$ (this paper)\par} \\
\midrule
{\raggedright Separable utility (Proposition~\ref{prop-egm-utility})\par} & {\raggedright inversion, if system triangular\par} & {\raggedright inversion (consumption only)\par} & {\raggedright inversion + upper envelope\par} & {\raggedright inversion\par} \\
\addlinespace
{\raggedright Invertible transition marginal (Proposition~\ref{prop-egm-transition})\par} & {\raggedright inversion, if system triangular\par} & {\raggedright nested root-find\par} & {\raggedright inversion + upper envelope\par} & {\raggedright inversion\par} \\
\addlinespace
{\raggedright Occasionally binding constraint\par} & {\raggedright not developed\par} & {\raggedright corner via outer root-find\par} & {\raggedright corner via upper envelope\par} & {\raggedright corner reads next stage\par} \\
\bottomrule
\end{tabular}
\end{table}
```

In the two-asset example, EGM$^n$ evaluates an explicit inverse where NEGM nests a root-find.

For $n$ independent scalar bounds under this sequential structure, a simultaneous upper envelope considers up to $2^n$ binding regimes, while the decomposition handles two branches in each of $n$ stages. At $n=2$ the counts coincide, so the Section 3 timing comparison does not measure this distinction. This is an architectural count, not a general complexity theorem: it requires that no cross-stage constraint regime be carried forward.

For a new model, the conditions work as a checklist: count the controls, order the stages to shed state variables early, and test each stage against the conditions stated at the start of this section, of which @prop-egm-utility and @prop-egm-transition are sufficient cases. A stage that passes inverts in closed form, while one that fails keeps standard optimization. ENGINE then interpolates any resulting mesh that is fold-free and index-monotone.
