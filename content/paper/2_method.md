
(method)=

# The Sequential Endogenous Grid Method

Models where households simultaneously choose consumption, labor supply, and portfolio allocation present a computational challenge. Joint optimization over all three choices requires evaluating a three-dimensional optimization at every state space point. Imposing strong separability restrictions speeds computation but may rule out economically interesting preference specifications. Sequential decomposition exploits partial separability: when decisions are separable in stages but not necessarily globally, each stage can be solved efficiently through EGM inversion.

## Problem Setup

The baseline problem which we use to demonstrate the Sequential Endogenous Grid Method (EGM$^n$) is a discrete time version of {cite:t}`Bodie1992` where a consumer has the ability to adjust their labor as well as their consumption in response to financial risk. The objective consists of maximizing the present discounted lifetime utility of consumption and leisure.

\begin{equation}
    \VFunc_0(\BLev_0, \tShkEmp_0) = \max \Ex_{t}
    \left[ \sum_{n = 0}^{T-t} \DiscFac^{n} \utilFunc(\CLev_{t+n}, \Leisure_{t+n})  \right].
\end{equation}

In particular, this example makes use of a utility function that is based on Example 1 in the paper, which is that of additively separable utility of labor and leisure as

\begin{equation}
    \utilFunc(\CLev, \Leisure) = \util(\CLev) + \h(\Leisure) = \frac{C^{1-\CRRA}}{1-\CRRA} + \labShare^{1-\CRRA}
    \frac{\Leisure^{1-\leiShare}}{1-\leiShare}
\end{equation}

where the term $\labShare^{1-\CRRA}$ scales the leisure utility to have the same curvature as consumption utility, following the approach in {cite:t}`Mertens2011`.[^alt-param] The use of additively separable utility is ad-hoc, as it will allow for the use of multiple EGM steps in the solution process, as we'll see later. For the remainder of the analysis, we work with normalized variables (lowercase) where consumption $\cRat = \CLev/\PLev$ and leisure $\leisure$ represent quantities relative to permanent income or in natural units.

[^alt-param]: An alternative formulation for the utility of leisure is to state it in terms of the disutility of labor as $\h(\labor) = - \leiShare \dfrac{\labor^{1+\labShare}}{1+\labShare}$, which gives $\h'(\leisure) = \leiShare(1-\leisure)^{\labShare}$ and $\h'^{-1}(\xRat) = 1 - (\xRat/\leiShare)^{1/\labShare}$. Note that this formulation does not support a balanced growth path because leisure utility is not homogeneous of degree $1-\CRRA$ in permanent income. A BGP-consistent specification would require $\h(\Leisure, \PLev) = (\labShare\PLev)^{1-\CRRA} \dfrac{\Leisure^{1-\leiShare}}{1-\leiShare}$, making leisure utility scale with permanent income.

This model represents a consumer who begins the period with a level of bank balances $\bRat_{t}$ and a given wage offer $\tShkEmp_{t}$. Simultaneously, they are able to choose consumption, labor intensity, and a risky portfolio share with the objective of maximizing their utility of consumption and leisure, as well as their future wealth.

Expressing the problem in normalized recursive form[^normalized-form] makes the stationarity of the decision rules apparent. The household solves

\begin{equation}
    \begin{split}
        \vFunc_{t}(\bRat_{t}, \tShkEmp_{t}) & = \max_{\{\cRat_{t},
            \leisure_{t}, \riskyshare_{t}\}} \utilFunc(\cRat_{t}, \leisure_{t}) +
        \DiscFac \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA}
            \vFunc_{t+1} (\bRat_{t+1},
            \tShkEmp_{t+1}) \right] \\
        & \text{s.t.} \\
        \labor_{t} & = 1 - \leisure_{t} \\
        \mRat_{t} & = \bRat_{t} + \tShkEmp_{t}\labor_{t} \\
        \aRat_{t} & = \mRat_{t} - \cRat_{t} \\
        \Rport_{t+1} & = \Rfree + (\Risky_{t+1} - \Rfree)
        \riskyshare_{t} \\
        \bRat_{t+1} & = \aRat_{t} \Rport_{t+1} / \PGro_{t+1}
    \end{split}
\end{equation}

where non-negativity constraints $\cRat_{t} \geq 0$, $\leisure_{t} \in [0,1]$, and $\riskyshare_{t} \in [0,1]$ restrict feasible choices. Throughout, we assume standard constraint qualifications hold such that interior solutions satisfy first-order conditions.[^constraint-qual] The constraints define a sequence of state transitions: labor supply $\labor_{t}$ determines market resources $\mRat_{t}$ (bank balances plus labor income), consumption determines liquid savings $\aRat_{t}$, and the portfolio choice $\riskyshare_{t}$ induces a stochastic return $\Rport_{t+1}$ that yields next period's normalized bank balances $\bRat_{t+1}$.

[^constraint-qual]: Specifically, we assume: (i) utility and value functions are twice continuously differentiable in the interior of the constraint set; (ii) the Inada conditions $\lim_{c\to 0} \util'(c) = \infty$ and $\lim_{c\to\infty} \util'(c) = 0$ hold, ensuring interior solutions away from zero consumption; and (iii) constraint sets are convex with non-empty interior. These conditions ensure first-order conditions are necessary for optimality at interior solutions.

[^normalized-form]: As in {cite:t}`Carroll2009`, where the utility of normalized consumption and leisure is defined as

    \begin{equation}
        \utilFunc(\cRat_{t}, \leisure_{t}) = \PLev_{t}^{1-\CRRA} \dfrac{\cRat_{t}^{1-\CRRA}}{1-\CRRA} + (\labShare\PLev_{t})
        ^{1-\CRRA} \dfrac{\leisure_{t}^{1-\leiShare}}{1-\leiShare}
    \end{equation}

Although the household makes all three decisions simultaneously from an economic perspective, the dependence structure permits sequential solution. The labor-leisure choice determines market resources; given those resources, the consumption-saving choice determines liquid assets; given liquid assets, the portfolio choice follows. This natural ordering reflects the problem's information flow rather than introducing artificial timing.

The decomposition proceeds as follows. The labor-leisure decision and wage realization jointly determine market resources. Given market resources, the consumption-saving decision determines liquid assets. Given liquid assets, the portfolio allocation follows. Each stage uses information from subsequent stages (through continuation values) while shedding state variables that later stages do not require.

The sequential decomposition begins at the start of the period with the labor-leisure problem.[^stage-notation] At this stage, the household observes bank balances $\bRat_{t}$ and the wage offer $\tShkEmp_{t}$, choosing leisure to maximize the sum of current leisure utility and the continuation value from market resources $\mRat_{t}$:

[^stage-notation]: We now introduce stage superscripts to distinguish value functions at different stages of the sequential decomposition. The original problem has $\vFunc_t \equiv v^0_t$, representing the value at the first decision stage (labor-leisure). Each subsequent stage $v^i_t$ represents the value function after making decisions at stages $0, 1, \ldots, i-1$.

\begin{equation}
    \begin{split}
        v^{0}_{t}(\bRat_{t}, \tShkEmp_{t}) & = \max_{ \leisure_{t}}
        \h(\leisure_{t}) + v^{1}_{t} (\mRat_{t}) \\
        & \text{s.t.} \\
        \leisure_{t} & \in [0, 1] \\
        \labor_{t} & = 1 - \leisure_{t} \\
        \mRat_{t} & = \bRat_{t} + \tShkEmp_{t}\labor_{t}.
    \end{split}
\end{equation}

Once market resources are realized, the pure consumption-saving problem determines how to allocate $\mRat_{t}$ between current consumption and liquid assets. The state space has been reduced to a single dimension since the wage offer no longer matters:

\begin{equation}
    \begin{split}
        v^{1}_{t}(\mRat_{t}) & = \max_{\cRat_{t}} \util(\cRat_{t}) + \DiscFac v^{2}_{t}(\aRat_{t}) \\
        & \text{s.t.} \\
        \cRat_{t} & \in [0, \mRat_{t}] \\
        \aRat_{t} & = \mRat_{t} - \cRat_{t}.
    \end{split}
\end{equation}

The final stage allocates liquid savings $\aRat_{t}$ between risk-free and risky assets. This portfolio problem involves no within-period utility, only the expected continuation value from next period's bank balances:

\begin{equation}
    \begin{split}
        v^{2}_{t}(\aRat_{t}) & = \max_{\riskyshare_{t}}
        \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA}
            v^{0}_{t+1}(\bRat_{t+1},
            \tShkEmp_{t+1}) \right] \\
        & \text{s.t.} \\
        \riskyshare_{t} & \in [0, 1] \\
        \Rport_{t+1} & = \Rfree + (\Risky_{t+1} - \Rfree)
        \riskyshare_{t} \\
        \bRat_{t+1} & = \aRat_{t} \Rport_{t+1} / \PGro_{t+1}.
    \end{split}
\end{equation}

The sequential formulation follows the nested approaches of {cite:t}`Clausen2020` and {cite:t}`Druedahl2021` but chains EGM inversions without embedding optimization. Each choice is self-contained in a subproblem, with the structure chosen to minimize state variables at each stage. The sequential formulation preserves the original problem because no uncertainty resolves between subproblems within a single period. From the agent's information set at time $t$, all three decisions are made before any time-$t+1$ shocks realize. The expectation operator appears only in the final subproblem, ensuring identical information across decisions. The sequential organization reduces computational cost while exposing intermediate economic quantities. The marginal value of wealth and the Frisch elasticity of labor emerge explicitly from the stage decomposition.

## Sequential Solution

Not every subproblem admits an EGM solution. The portfolio stage illustrates this limitation. The reorganization assigned leisure utility to the labor-leisure stage and consumption utility to the consumption-savings stage, exhausting the separable utility components. The portfolio subproblem lacks a separable utility term for the risky share. The risky share affects utility only through future wealth, not through contemporaneous utility. This subproblem requires standard convex optimization rather than EGM inversion.

Restating the problem in compact form gives

\begin{equation}
    v^{2}_{t}(\aRat_{t}) = \max_{\riskyshare_{t}} \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA}
    v^{0}_{t+1}\left(\aRat_{t}(\Rfree + (\Risky_{t+1} - \Rfree) \riskyshare_{t})/\PGro_{t+1}, \tShkEmp_{t+1}\right)
    \right].
\end{equation}

The first-order condition with respect to the risky portfolio share is then

\begin{equation}
    \Ex_{t}
    \left[ \PGro_{t+1}^{-\CRRA} \frac{\partial v^{0}_{t+1}}{\partial \bRat}\left(\bRat_{t+1}, \tShkEmp_{t+1}\right) (\Risky_{t+1} - \Rfree)  \right] =
    0.
\end{equation}

Finding the optimal risky share requires numerical optimization and root-solving of the first-order condition. To close out the problem, we can calculate the envelope condition as

\begin{equation}
    (v^{2}_{t})'(\aRat_{t}) = \Ex_{t}
    \left[ \PGro_{t+1}^{-\CRRA} \frac{\partial v^{0}_{t+1}}{\partial \bRat}\left(\bRat_{t+1}, \tShkEmp_{t+1}\right) \Rport_{t+1} \right].
\end{equation}

This completes the portfolio stage solution.[^alt-portfolio-formulation]

The consumption-saving EGM follows {cite:t}`Carroll2006` but we cover it for exposition. We can begin the solution process by restating the consumption-savings subproblem in a more compact form, substituting the market resources constraint and ignoring the no-borrowing constraint for now. The problem is:

\begin{equation}
    v^{1}_{t}(\mRat_{t}) = \max_{\cRat_{t}} \util(\cRat_{t}) +
    \DiscFac v^{2}_{t}(\mRat_{t}-\cRat_{t}).
\end{equation}

The first-order condition with respect to $\cRat_{t}$ yields the familiar Euler equation:

\begin{equation}
    \util'(\cRat_t) = \DiscFac (v^{2}_{t})'(\mRat_{t} - \cRat_{t}) = \DiscFac
    (v^{2}_{t})'(\aRat_{t})
\end{equation}

Inverting this equation is the (first) EGM step.[^inverse-monotone]

\begin{equation}
    \cEndFunc_{t}(\aRat_{t}) = \util'^{-1}\left( \DiscFac (v^{2}_{t})'(\aRat_{t})
    \right)
\end{equation}

[^inverse-monotone]: Invertibility of $\util'$ requires strict monotonicity, which holds when $\util'' < 0$. For CRRA utility with $\CRRA > 0$, we have $\util''(c) = -\CRRA c^{-\CRRA-1} < 0$, ensuring a one-to-one mapping between marginal utility and consumption levels.

Given the utility function above, the marginal utility of consumption and its inverse are

\begin{equation}
    \util'(\cRat) = \cRat^{-\CRRA} \qquad \util'^{-1}(\xRat) =
    \xRat^{-1/\CRRA}.
\end{equation}

{cite:t}`Carroll2006` demonstrates that by using an exogenous grid of $\aMat$ points we can find the unique $\cEndFunc_{t}(\aMat)$ that optimizes the consumption-saving problem.[^egm-notation] The strict concavity of $\util$ and $v^{2}_{t}$ (inherited from the value function) combined with the convex constraint set ensures the first-order condition is both necessary and sufficient[^foc-sufficient] for a unique optimum. Further, using the market resources constraint, we can recover the exact amount of market resources that is consistent with this consumption-saving decision as

[^egm-notation]: We adopt the notational convention that bracketed variables (e.g., $\aMat$, $\mMat$) denote exogenous grids of points on which we evaluate expectations and marginal values, while gothic (fraktur) letters (e.g., $\cEndFunc$, $\mEndFunc$) denote endogenous quantities constructed by inverting first-order conditions. This visual distinction emphasizes that grids are chosen inputs while gothic variables emerge from the EGM inversion step.

[^foc-sufficient]: Necessity follows from standard optimality conditions under differentiability. Sufficiency follows from the strict concavity of the objective function, which guarantees that any critical point is a global maximum. The strict concavity of CRRA utility and the inheritance of concavity through the continuation value ensure uniqueness of the solution.

\begin{equation}
    \mEndFunc_{t}(\aMat) = \cEndFunc_{t}(\aMat) + \aMat.
\end{equation}

This $\mEndFunc_{t}(\aMat)$ is the ``endogenous'' grid that is consistent with the exogenous decision grid $\aMat$. Now that we have a $(\mEndFunc_{t}(\aMat), \cEndFunc_{t}(\aMat))$ pair for each $\aRat \in \aMat$, we can construct an interpolating consumption function for market resources points that are off-the-grid.

The envelope condition[^envelope-thm] will be useful in the next section, but for completeness we define it here.

\begin{equation}
    (v^{1}_{t})'(\mRat_{t}) = \DiscFac (v^{2}_{t})'(\aRat_{t}) = \util'(\cRat_{t})
\end{equation}

[^envelope-thm]: Follows from the envelope theorem, valid when the value function is differentiable and the constraint set satisfies standard regularity conditions.

The labor-leisure subproblem can be restated more compactly as:

\begin{equation}
    v^{0}_{t}(\bRat_{t}, \tShkEmp_{t}) = \max_{ \leisure_{t}}
    \h(\leisure_{t}) + v^{1}_{t}(\bRat_{t} +
    \tShkEmp_{t}(1-\leisure_{t}))
\end{equation}

The first-order condition with respect to leisure is

\begin{equation}
    \h'(\leisure_{t}) = (v^{1}_{t})'(\mRat_{t}) \tShkEmp_{t}
\end{equation}

The marginal utility of leisure and its inverse are

\begin{equation}
    \h'(\leisure) = \labShare^{1-\CRRA}\leisure^{-\leiShare} \qquad
    \h'^{-1}(\xRat) = (\xRat/\labShare^{1-\CRRA})^{-1/\leiShare}
\end{equation}

Using an exogenous grid of $\mMat$ and $\tShkMat$, we can find leisure as

\begin{equation}
    \zEndFunc_{t}(\mMat, \tShkMat) = \h'^{-1}\left(
    (v^{1}_{t})'(\mMat) \tShkMat \right)
\end{equation}

However, agents with low market resources $\mRat_{t}$ and high wage offers $\tShkEmp_{t}$ may find the unconstrained optimum violates the feasibility constraint $\leisure_t \in [0,1]$. When this occurs, we project the solution onto the constraint boundary, defining the constrained optimal function $\hat{\zEndFunc}_{t}(\mMat, \tShkMat)$ as

\begin{equation}
    \hat{\zEndFunc}_{t}(\mMat, \tShkMat) = \max \left\{ \min \left\{ \zEndFunc_{t}(\mMat, \tShkMat), 1 \right\}, 0 \right\}
\end{equation}

This projection ensures feasibility.[^corner-solution] In regions where constraints bind, the Kuhn-Tucker conditions replace the unconstrained first-order condition. Care must be taken during interpolation to handle potential non-differentiabilities at constraint boundaries, though these typically affect only small regions of the state space.

[^corner-solution]: At the lower bound $\leisure_t = 0$, the Kuhn-Tucker condition is $\h'(0) \leq (v^{1}_t)'(\mRat_t)\tShkEmp_t$, with complementary slackness ensuring the constraint binds only when the marginal utility of leisure is insufficient to justify reduced labor supply. Similarly, at $\leisure_t = 1$, the agent chooses full leisure only when $\h'(1) \geq (v^{1}_t)'(\mRat_t)\tShkEmp_t$.

Then, we derive labor as $\lEndFunc_{t}(\mRat_{t}, \tShkEmp_{t}) = 1 - \hat{\zEndFunc}_{t}(\mRat_{t}, \tShkEmp_{t})$. Finally, for each $\tShkEmp_{t}$ and $\mRat_{t}$ as an exogenous grid, we can find the endogenous grid of bank balances as $\bEndFunc_{t}(\mRat_{t}, \tShkEmp_{t}) = \mRat_{t} - \tShkEmp_{t}\lEndFunc_{t}(\mRat_{t}, \tShkEmp_{t})$.

The envelope condition then provides the marginal value of bank balances as

\begin{equation}
    \frac{\partial v^{0}_{t}}{\partial \bRat}(\bRat_{t}, \tShkEmp_{t}) = (v^{1}_{t})'(\mRat_{t}) =
    \h'(\leisure_{t})/\tShkEmp_{t}.
\end{equation}

This envelope condition, together with the first-order condition, implicitly defines the heterogeneous Frisch elasticity of labor supply, which varies across states $(\bRat_{t}, \tShkEmp_{t})$.[^frisch-elasticity]

[^frisch-elasticity]: The Frisch elasticity of labor supply is defined as $\varepsilon_{\labor,\tShkEmp} = \dfrac{\partial \labor}{\partial \tShkEmp}\dfrac{\tShkEmp}{\labor}$ holding the marginal utility of wealth constant. From the first-order condition $\h'(\leisure_{t}) = (v^{1}_{t})'(\mRat_{t}) \tShkEmp_{t}$, we implicitly differentiate with respect to $\tShkEmp_{t}$ while holding $(v^{1}_{t})'(\mRat_{t})$ fixed: $\h''(\leisure_t)\dfrac{\partial \leisure_t}{\partial \tShkEmp} = (v^{1}_t)'(\mRat_t)$. Since $\labor_t = 1 - \leisure_t$, we obtain $\dfrac{\partial \labor_t}{\partial \tShkEmp} = -\dfrac{(v^{1}_t)'(\mRat_t)}{\h''(1-\labor_t)}$. For the CRRA leisure utility, $\h''(\leisure) = -\leiShare \labShare^{1-\CRRA} \leisure^{-\leiShare-1} < 0$, making the derivative positive. The elasticity $\varepsilon_{\labor,\tShkEmp} = -\dfrac{(v^{1}_t)'(\mRat_t)}{\h''(1-\labor_t)}\dfrac{\tShkEmp_t}{\labor_t}$ varies with the state because both $\h''(1-\labor_t)$ and the ratio $\tShkEmp_t/\labor_t$ depend on $(\bRat_t, \tShkEmp_t)$.

The resulting endogenous grid for the labor-leisure problem is curvilinear rather than rectilinear, requiring specialized interpolation methods. We defer the detailed discussion of interpolation on curvilinear grids to [Section %s](#multinterp).[^cgi-pedagogical]

[^cgi-pedagogical]: The labor-leisure problem could be solved using simpler interpolation methods since the grid warping occurs along only one dimension (wage offers). However, we use Curvilinear Grid Interpolation here for two pedagogical reasons: (1) it demonstrates the sequential decomposition that is the essence of EGM$^n$, and (2) it illustrates CGI in a transparent setting. CGI is robust to various types of grid warping, from simple one-dimensional stretching to complex multidimensional distortions. This makes it valuable to understand in this simpler context before encountering the genuinely unstructured grids of [Section %s](#multdim).

[^alt-portfolio-formulation]: An alternative formulation avoids taking expectations more than once. We could define the portfolio choice subproblem as $v^{2}_{t}(\aRat_{t}) = \max_{\riskyshare_{t}} \tilde{v}^{1}_{t}(\aRat_{t}, \riskyshare_{t})$ where $\tilde{v}^{1}_{t}(\aRat_{t}, \riskyshare_{t}) = \Ex_{t}[\PGro_{t+1}^{1-\CRRA} v^{0}_{t+1}(\bRat_{t+1}, \tShkEmp_{t+1})]$ with $\Rport_{t+1} = \Rfree + (\Risky_{t+1} - \Rfree) \riskyshare_{t}$ and $\bRat_{t+1} = \aRat_{t} \Rport_{t+1} / \PGro_{t+1}$. Given the next period's solution, we calculate the marginal value functions as $\frac{\partial \tilde{v}^{1}_{t}}{\partial \aRat}(\aRat_{t}, \riskyshare_{t}) = \Ex_{t}[\PGro_{t+1}^{-\CRRA} \frac{\partial v^{0}_{t+1}}{\partial \bRat}(\bRat_{t+1}, \tShkEmp_{t+1}) \Rport_{t+1}]$ and $\frac{\partial \tilde{v}^{1}_{t}}{\partial \riskyshare}(\aRat_{t}, \riskyshare_{t}) = \Ex_{t}[\PGro_{t+1}^{-\CRRA} \frac{\partial v^{0}_{t+1}}{\partial \bRat}(\bRat_{t+1}, \tShkEmp_{t+1}) \aRat_{t} (\Risky_{t+1} - \Rfree)]$. Both can be computed in one expectation step. The optimal risky share then satisfies $\frac{\partial \tilde{v}^{1}_{t}}{\partial \riskyshare}(\aRat_{t}, \riskyshare_{t}^{*}) = 0$ with envelope condition $(v^{2}_{t})'(\aRat_{t}) = \frac{\partial \tilde{v}^{1}_{t}}{\partial \aRat}(\aRat_{t}, \riskyshare_{t}^{*})$.

The labor-portfolio example demonstrates sequential EGM's core principle: decompose economically simultaneous decisions into computationally sequential stages, chaining EGM inversions without optimization. Sequential decomposition exploited separability in utility (leisure) and transitions (portfolio returns). The labor-leisure stage used EGM inversion; the portfolio stage required convex optimization. Each stage required at most one post-decision state variable, keeping dimensionality manageable. The resulting curvilinear grid from the labor-leisure inversion requires specialized interpolation—addressed in [Section %s](#multinterp). Retirement planning with multiple accounts requires tracking several state variables simultaneously across stages, presenting a more demanding test of the method's applicability.
