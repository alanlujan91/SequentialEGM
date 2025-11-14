
(conditions)=

# Conditions for using the Sequential Endogenous Grid Method

When does Sequential EGM apply? The labor-portfolio problem ([Section %s](#method)) and pension deposit problem ([Section %s](#multdim)) illustrated two key structures: separable utility functions and invertible transitions. This section formalizes these requirements and provides practical guidance for decomposing new problems. The examples demonstrated specific instances; we now characterize the general conditions that make sequential decomposition with EGM inversion possible.

## Splitting the problem into subproblems

**Step 1: Count independent control variables.** A problem with $n$ control variables typically decomposes into $n$ subproblems. Avoid double-counting: the consumption-savings choice ($\cRat + \aRat = \mRat$) represents one decision, not two. Similarly, labor-leisure is a single choice despite involving two variables.

**Step 2: Identify enabling structures.** Two mathematical structures permit EGM inversion:
- Separable, differentiable, and invertible utility functions (as in the leisure utility of [Section %s](#method))
- Differentiable and invertible transition functions (as in the pension deposit function of [Section %s](#multdim))

Match each control variable to its enabling structure. The labor-portfolio example features additive utility separability: leisure utility enables the labor-leisure EGM step, consumption utility enables the consumption-savings EGM step. When no structure applies (as in the portfolio choice stage), use standard optimization.

**Step 3: Order subproblems to shed state variables early.** Poor sequencing propagates unnecessary state variables through later stages. In the consumption-leisure-portfolio problem, placing labor-leisure first resolves the wage rate before the consumption stage, keeping that subproblem one-dimensional. Choosing consumption first would force the labor decision to track both bank balances and wages, doubling its dimensionality.[^bad-ordering-example] When subproblems are independent (consumption and pension deposit each affect separate accounts), ordering is immaterial.

We now formalize these requirements. Consider the utility function of the form

\begin{equation}
    \UFunc( \aRat) = \uFunc_{-i}( \aRat^{-i}) + \uFunc_i(\aRat^i)
\end{equation}

where $\aRat^{i}$ is the $i$-th control variable and $\aRat^{-i}$ is the vector of all control variables except the $i$-th one. This utility function is separable in the control variables that correspond to the index $i$.

\begin{equation}
    \begin{split}
    \VFunc(\xRat, \sRat) &= \max_{\aRat \in \Gamma(\xRat, \sRat)} \UFunc(\aRat)  + \DiscFac \Ex \left[ \VFunc'(\xRat', \sRat') | \yRat,  \sRat \right] \\
    & \text{s.t.} \\
    \yRat &= \TFunc(\xRat, \aRat) \\
    \xRat' &= \GFunc(\yRat, \sRat) \\
    \end{split}
\end{equation}

For simplicity, define

\begin{equation}
    \WFunc(\yRat, \sRat) = \DiscFac \Ex \left[ \VFunc'(\GFunc(\yRat, \sRat), \sRat') | \yRat, \sRat \right]
\end{equation}

then

\begin{equation}
    \begin{split}
    \VFunc(\xRat, \sRat) &= \max_{\aRat \in \Gamma(\xRat, \sRat)} \UFunc( \aRat)  +  \WFunc(\yRat, \sRat) \\
    & \text{s.t.} \\
    \yRat &= \TFunc(\xRat, \aRat)
    \end{split}
\end{equation}

the first-order condition (assuming an interior solution)

\begin{equation}
    \frac{\partial \UFunc( \aRat)}{\partial \aRat^i}  +  \sum_{j=1}^{n} \frac{\partial \WFunc(\yRat, \sRat)}{\partial \yRat^j} \frac{\partial \TFunc^j(\xRat, \aRat)}{\partial \aRat^i} = 0
\end{equation}

we require $\frac{\partial \TFunc^j(\xRat, \aRat)}{\partial \aRat^i} = 0$ for $j \neq i$ (separability in the transition) to be able to solve for $\aRat^i$ independently.

\begin{equation}
    \frac{\partial \UFunc( \aRat)}{\partial \aRat^i}  +   \frac{\partial \WFunc(\yRat, \sRat)}{\partial \yRat^i} \frac{\partial \TFunc^i(\xRat, \aRat)}{\partial \aRat^i} = 0
\end{equation}

The pension deposit problem in [Section %s](#multdim) illustrates another case where differentiable and invertible transitions enable an additional EGM step. The transition applies independently to a state variable unrelated to consumption, allowing it to be handled separately from the consumption subproblem. Interestingly, the ordering of these two subproblems proves immaterial because consumption and pension deposit each affect separate resource accounts: market resources and pension balance, respectively. Their independence means either ordering yields the same computational structure.

Minimize the information set passed forward at each stage. The labor-portfolio example illustrates this principle: the leisure-labor choice realizes market resources and sheds the wage rate before the consumption problem. The portfolio choice requires only liquid assets after consumption, simplifying the final stage. The sequence (leisure-labor, consumption-savings, portfolio allocation) reflects the information structure. When subproblems are independent, as with consumption and deposit in the pension problem, sequencing is immaterial.

## The Endogenous Grid Method for Subproblems

Once we have strategically split the problem into subproblems, we can use the Endogenous Grid Method in each applicable subproblem while iterating backwards from the terminal period. As demonstrated in [Section %s](#method) and [Section %s](#multdim), the EGM step can be applied when there is a separable, differentiable and invertible utility function in the subproblem or when there is a differentiable and invertible transition in the subproblem. We will discuss each of these cases in turn.

Consider a generic subproblem with a differentiable and invertible utility function:

\begin{equation}
    \begin{split}
        \VFunc(\xRat) & = \max_{\aRat \in \PGro(\xRat)} \UFunc(\xRat, \aRat) + \WFunc(\yRat) \\
        & \text{s.t.} \\
        \yRat & = \TFunc(\xRat,\aRat)
    \end{split}
\end{equation}

where $\WFunc(\yRat) = \DiscFac \Ex[\VFunc'(\yRat)]$ is the continuation value. For an interior solution, the first-order condition is

\begin{equation}
    \frac{\partial \UFunc(\xRat, \aRat)}{\partial \aRat} + \WFunc'(\yRat) \frac{\partial \TFunc(\xRat,\aRat)}{\partial \aRat} = 0
\end{equation}

When corner solutions occur (e.g., $\aRat$ at constraint boundaries), the unconstrained optimum from inverting the first-order condition must be projected onto the feasible set, as demonstrated in [Section %s](#method) for the leisure choice.

```{prf:proposition} Separable Utility
:label: prop-egm-utility

For interior solutions where the marginal utility $\partial \UFunc / \partial \aRat$ is strictly monotone in $\aRat$,[^egm-invertibility] the first-order condition can be inverted to obtain

\begin{equation}
    \aRat = \left(\frac{\partial \UFunc(\xRat, \aRat)}{\partial \aRat}\right)^{-1}
    \left[ -\WFunc'(\yRat) \frac{\partial \TFunc(\xRat,\aRat)}{\partial \aRat}\right]
\end{equation}

When the utility function is strictly concave in $\aRat$, the solution is unique.
```

By using an exogenous grid of the post-decision state $\yRat$, we can solve for the optimal decision rule $\aRat$ at each point on the grid. This is the Endogenous Grid Method step. The monotonicity requirement ensures that the mapping from the post-decision state to the control is well-defined, while concavity guarantees uniqueness of the optimal decision at each grid point.

## Applicability to Transition Functions

When the generic subproblem has no separable utility but instead has differentiable and invertible transitions that affect multiple post-decision states, the Endogenous Grid Method can still be applied. Consider a problem with two endogenous state variables and two post-decision states:

\begin{equation}
    \begin{split}
        \VFunc(\xRat_1, \xRat_2, \sRat) & = \max_{\aRat \in \PGro(\xRat_1, \xRat_2, \sRat)} \WFunc(\yRat_1, \yRat_2, \sRat) \\
        & \text{s.t.} \\
        \yRat_1 & = \TFunc_1(\xRat_1,\aRat) \\
        \yRat_2 & = \TFunc_2(\xRat_2,\aRat)
    \end{split}
\end{equation}

where the continuation value is

\begin{equation}
    \WFunc(\yRat_1, \yRat_2, \sRat) = \DiscFac \Ex \left[ \VFunc'(\GFunc_1(\yRat_1, \sRat), \GFunc_2(\yRat_2, \sRat), \sRat') | \yRat_1, \yRat_2, \sRat \right]
\end{equation}

The first-order condition becomes

\begin{equation}
    \frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_1} \cdot \frac{\partial \TFunc_1(\xRat_1,\aRat)}{\partial \aRat} + \frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_2} \cdot \frac{\partial \TFunc_2(\xRat_2,\aRat)}{\partial \aRat} = 0
\end{equation}

```{prf:proposition} Invertible Transitions
:label: prop-egm-transition

Suppose both transitions are additively separable in the control:
$$\TFunc_1(\xRat_1, \aRat) = f_1(\xRat_1) + k \cdot \aRat, \quad \TFunc_2(\xRat_2, \aRat) = f_2(\xRat_2) + \gFunc(\aRat)$$
where $k \neq 0$ is constant, $f_1$ and $f_2$ are invertible, and $\gFunc'$ is strictly monotone. Then the first-order condition yields

\begin{equation}
    \aRat = \gFunc'^{-1}\left( -k \cdot \dfrac{\partial \WFunc(\yRat_1, \yRat_2, \sRat) / \partial \yRat_1}{\partial \WFunc(\yRat_1, \yRat_2, \sRat) / \partial \yRat_2} \right)
\end{equation}

where strict monotonicity of $\gFunc'$ ensures existence and uniqueness of the inverse.[^g-monotone]
```

The additive separability in both transitions is essential: it allows the derivative with respect to $\aRat$ to not depend on the state variables $\xRat_1$ or $\xRat_2$, which we haven't yet recovered when solving the first-order condition on the exogenous grid of post-decision states. Once we obtain $\aRat$ from the inversion, we can recover the pre-decision states via $\xRat_1 = f_1^{-1}(\yRat_1 - k \cdot \aRat)$ and $\xRat_2 = f_2^{-1}(\yRat_2 - \gFunc(\aRat))$. The current formulation where one state variable enters linearly (e.g., $\TFunc_2 = \xRat_2 + \aRat + \gFunc(\aRat)$ with $f_2(\xRat_2) = \xRat_2$) is a common special case.

This additive separability defines what {cite:t}`Iskhakov2015` calls "triangular" structure in transitions. {cite:t}`Iskhakov2015` solves problems where the entire multidimensional structure is triangular, enabling simultaneous solution of all choices. Sequential EGM instead decomposes problems into stages, requiring only that each subproblem satisfy local EGM-compatibility conditions. A multistage problem can mix stages satisfying Proposition 1 (separable utility) with stages satisfying Proposition 2 (triangular transitions), and even include stages solved by standard optimization, expanding applicability beyond problems with uniform triangular structure.

[^bad-ordering-example]: To see this concretely, the consumption subproblem would become two-dimensional: $v^{0}(\bRat, \tShkEmp) = \max_{\cRat} \uFunc(\cRat) + v^{1}(\bRat', \tShkEmp)$ subject to $\bRat' = \bRat - \cRat \ge -\tShkEmp$, requiring interpolation on a $(\bRat, \tShkEmp)$ grid instead of just $\bRat$. The labor-leisure subproblem would then have the additional constraint: $v^{1}(\bRat', \tShkEmp) = \max_{\leisure} \h(\leisure) + v^{2}(\aRat)$ subject to $0 \le \leisure \le 1$ and $\aRat = \bRat' + \tShkEmp(1 - \leisure) \ge 0$. The poor ordering forces us to carry the wage state through both stages, doubling the dimensionality of the first stage.



[^egm-invertibility]: Strict monotonicity of the marginal utility ensures that the inverse function is well-defined and single-valued. This condition is satisfied by standard utility functions like CRRA utility where $\uFunc'(\cRat) = \cRat^{-\CRRA}$ is strictly decreasing in consumption.

[^g-monotone]: The monotonicity of $\gFunc'$ is crucial for inverting the first-order condition. In the pension deposit example, the matching function satisfies this property, allowing us to recover the optimal deposit from the marginal value ratio.
