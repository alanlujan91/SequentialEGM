
(multdim)=

# The EGM$^n$ in Higher Dimensions

The labor-portfolio problem in [Section %s](#method) features at most one post-decision state variable per stage, keeping dimensionality manageable. Problems where multiple state variables persist across stages present a more demanding test. Retirement planning with multiple accounts, durable goods choices, and human capital investment all require tracking several state variables simultaneously. The pension deposit problem demonstrates that EGM$^n$ extends to such settings, though the interpolation challenge intensifies: endogenous grids lose even their topological regularity, requiring more sophisticated interpolation methods.

## A more complex problem

A worker saving for retirement faces choices about both consumption and tax-advantaged retirement contributions, creating a natural two-dimensional state space. The worker begins each period with liquid market resources $\mRat_{t}$ and illiquid retirement savings $\nRat_{t}$, choosing consumption $\cRat_{t}$ and pension deposit $\dRat_{t}$ to maximize lifetime utility. The pension deposit enters a tax-advantaged account exposed to risky returns and cannot be liquidated until retirement, while post-consumption liquid assets earn a risk-free return. Income arrives each period subject to both permanent ($\PGro_{t+1}$) and transitory ($\tShkEmp_{t+1}$) shocks. Upon retirement at age 65, the illiquid account becomes accessible and the state collapses to a single dimension. The feasibility constraint $\mRat_{t} \geq \cRat_{t} + \dRat_{t}$ ensures non-negative liquid savings. The worker's recursive problem takes the form:

\begin{equation}
    \begin{split}
        \vFunc_{t}(\mRat_{t}, \nRat_{t}) & = \max_{\cRat_{t}, \dRat_{t}} \util(\cRat_{t}) + \DiscFac \Ex_{t}
        \left[ \PGro_{t+1}^{1-\CRRA} \vFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1}) \right] \\
        & \text{s.t.} \quad \cRat_{t} \ge 0, \quad \dRat_{t} \ge 0 \\
        \aRat_{t} & = \mRat_{t} - \cRat_{t} - \dRat_{t} \\
        \bRat_{t} & = \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t}) \\
        \mRat_{t+1} & = \aRat_{t} \Rfree / \PGro_{t+1} + \tShkEmp_{t+1} \\
        \nRat_{t+1} & = \bRat_{t} \Risky_{t+1} / \PGro_{t+1}
    \end{split}
\end{equation}

where

\begin{equation}
    \gFunc(\dRat) = \xFer \log(1+\dRat).
\end{equation}

This problem can subsequently be broken down into 3 stages: a pension deposit stage, a consumption stage, and an income shock stage.

## Sequential Decomposition

In the deposit stage, the worker begins with market resources and a retirement savings account. The worker must maximize their value of liquid wealth $\lRat_{t}$ and retirement balance $\bRat_{t}$ by choosing a pension deposit $\dRat_{t}$, which must be positive. The retirement balance $\bRat$ is the cash value of their retirement account plus their pension deposit and an additional amount $\gFunc(\dRat_{t})$ that provides an incentive to save for retirement. As we'll see, this additional term will allow us to use the Endogenous Grid Method to solve this subproblem. We now decompose $\vFunc_t$ into sequential stages, introducing stage superscripts where $v^0_t \equiv \vFunc_t$:[^multidim-stage-notation]

[^multidim-stage-notation]: As in [Section %s](#method), stage superscripts distinguish value functions at different stages of the sequential decomposition. Here $v^0_t$ represents the deposit decision stage, $v^1_t$ the consumption decision stage, and $v^2_t$ the expectation stage after all decisions are made.

\begin{equation}
    \begin{split}
        v^{0}_{t}(\mRat_{t}, \nRat_{t}) & = \max_{\dRat_{t}} v^{1}_{t}(\lRat_{t}, \bRat_{t}) \\
        & \text{s.t.} \quad \dRat_{t} \ge 0 \\
        \lRat_{t} & = \mRat_{t} - \dRat_{t} \\
        \bRat_{t} & = \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t})
    \end{split}
\end{equation}

After making their pension decision, the worker begins their consumption stage with liquid wealth $\lRat_{t}$ and retirement balance $\bRat_{t}$. From their liquid wealth, the worker must choose a level of consumption to maximize utility and continuation value $v^{2}_{t}$. After consumption, the worker is left with post-decision states that represent liquid assets $\aRat_{t}$ and retirement balance $\bRat_{t}$, which passes through this problem unaffected because it can't be liquidated until retirement.

\begin{equation}
    \begin{split}
        v^{1}_{t}(\lRat_{t}, \bRat_{t}) & = \max_{\cRat_{t}} \util(\cRat_{t}) + \DiscFac v^{2}_{t}(\aRat_{t}, \bRat_{t})  \\
        & \text{s.t.} \quad \cRat_{t} \ge 0 \\
        \aRat_{t} & = \lRat_{t} - \cRat_{t}
    \end{split}
\end{equation}

Finally, the post-decision value function $v^{2}_{t}$ represents the value of both liquid and illiquid account balances before the realization of uncertainty regarding the risky return and income shocks. Since we are dealing with a normalized problem, this stage handles the normalization of state variables and value functions into the next period.

\begin{equation}
    \begin{split}
        v^{2}_{t}(\aRat_{t}, \bRat_{t}) & = \Ex_{t}
        \left[ \PGro_{t+1}^{1-\CRRA} v^{0}_{t+1}(\mRat_{t+1}, \nRat_{t+1}) \right] \\
        & \text{s.t.} \quad \aRat_{t} \ge 0, \quad \bRat_{t} \ge 0 \\
        \mRat_{t+1} & = \aRat_{t} \Rfree / \PGro_{t+1} + \tShkEmp_{t+1} \\
        \nRat_{t+1} & = \bRat_{t} \Risky_{t+1} / \PGro_{t+1}
    \end{split}
\end{equation}

The advantage of conceptualizing this subproblem as a separate stage is that we can construct a function $v^{2}_{t}$ and use it in the prior optimization problems without having to worry about stochastic optimization and taking expectations repeatedly.

## Solution via Sequential EGM

As seen in the consumption stage above, the retirement balance $\bRat_{t}$ passes through the problem unaffected because it can't be liquidated until retirement. In this sense, it is already a post-decision state variable. To solve this problem, we can use a fixed grid of $\bMat$ and for each obtain endogenous consumption and ex-ante market resources using the simple Endogenous Grid Method for the consumption problem.

In the deposit stage, both the state variables and post-decision variables are different since both are affected by the pension deposit decision.

First, we can rewrite the pension deposit problem more compactly:

\begin{equation}
    v^{0}_{t}(\mRat_{t}, \nRat_{t}) = \max_{\dRat_{t}}
    v^{1}_{t}(\mRat_{t} - \dRat_{t}, \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t}))
\end{equation}

The first-order condition is

\begin{equation}
    \frac{\partial v^{1}_{t}}{\partial \lRat}(\lRat_{t}, \bRat_{t})(-1) +
    \frac{\partial v^{1}_{t}}{\partial \bRat}(\lRat_{t}, \bRat_{t})(1+\gFunc'(\dRat_{t})) = 0.
\end{equation}

This condition is necessary for interior optima, with sufficiency following from the concavity of $v^{1}_{t}$ inherited from the continuation value function and the convexity of the constraint set.[^foc-deposit] Rearranging yields

[^foc-deposit]: The concavity of $v^{1}_t$ in both arguments ensures the objective function is concave in $\dRat_t$, making any critical point a global maximum. The constraint $\dRat_t \geq 0$ is handled by checking feasibility of the unconstrained solution.

\begin{equation}
    \gFunc'(\dRat_{t}) = \dfrac{\partial v^{1}_{t} / \partial \lRat (\lRat_{t}, \bRat_{t})}{\partial v^{1}_{t} / \partial \bRat (\lRat_{t}, \bRat_{t})} - 1
\end{equation}

where

\begin{equation}
    \gFunc'(\dRat) =
    \frac{\xFer}{1+\dRat} \qquad \gFunc'^{-1}(y) = \xFer/y - 1
\end{equation}

Note that $\gFunc'(\dRat) > 0$ for all $\dRat > -1$, ensuring strict monotonicity and hence invertibility. We can find

\begin{equation}
    \dEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \gFunc'^{-1}\left(
    \dfrac{\partial v^{1}_{t} / \partial \lRat (\lRat_{t}, \bRat_{t})}{\partial v^{1}_{t} / \partial \bRat (\lRat_{t}, \bRat_{t})} - 1 \right)
\end{equation}

Using this, we can back out $\nRat_{t}$ as

\begin{equation}
    \nEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \bRat_{t} -
    \dEndFunc_{t}(\lRat_{t}, \bRat_{t}) - \gFunc(\dEndFunc_{t}(\lRat_{t},
        \bRat_{t}))
\end{equation}

and $\mRat_{t}$ as

\begin{equation}
    \mEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \lRat_{t} +
    \dEndFunc_{t}(\lRat_{t}, \bRat_{t})
\end{equation}

In sum, given an exogenous grid $(\lMat, \bMat)$ we obtain the triple $\left(\mEndFunc_{t}(\lRat_{t}, \bRat_{t}), \nEndFunc_{t}(\lRat_{t}, \bRat_{t}), \dEndFunc_{t}(\lRat_{t}, \bRat_{t})\right)$, which we can use to create an interpolator for the decision rule $\dRat_{t}$.

To close the solution method, the envelope conditions are

\begin{equation}
    \begin{split}
        \frac{\partial v^{0}_{t}}{\partial \mRat}(\mRat_{t}, \nRat_{t}) & =
        \frac{\partial v^{1}_{t}}{\partial \lRat}(\lRat_{t}, \bRat_{t}) \\
        \frac{\partial v^{0}_{t}}{\partial \nRat}(\mRat_{t}, \nRat_{t}) & =
        \frac{\partial v^{1}_{t}}{\partial \bRat}(\lRat_{t}, \bRat_{t}).
    \end{split}
\end{equation}

The resulting endogenous grid in this problem is irregular and unstructured, unlike the curvilinear grid in the labor-leisure problem of [Section %s](#method). The interpolation techniques required for this more complex case are discussed in detail in [Section %s](#multinterp).

Successful EGM application at each stage does not guarantee regular grids. The pension deposit problem generates highly irregular endogenous grids, presenting an interpolation challenge.
