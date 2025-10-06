
(conditions)=

# Conditions for using the Sequential Endogenous Grid Method

## Splitting the problem into subproblems

The first step in using the Sequential Endogenous Grid Method is to split the problem into subproblems. This process of splitting up the problem has to be strategic to not insert additional complexity into the original problem. If one is not careful when doing this, the subproblems can become more complex and intractable than the original problem.

To split up the problem, we first count the number of control variables or decisions faced by the agent. Ideally, if the agent has $n$ control variables, then the problem should be split into $n$ subproblems, each handling a different control variable. For counting the number of control variables, it is important to not double count variables which are equivalent and have market clearing conditions. For example, the decision of how much to consume and how much to save may seem like two different choices, but because of the market clearing condition $\cRat + \aRat = \mRat$ they are resolved simultaneously and count as only one decision variable. Similarly, the choice between labor and leisure are simultaneous and count as only one decision.

Having counted our control variables, we look for differentiable and invertible utility functions which are separable in the dynamic programming problem, such as in [Section %s](#method) of the paper, or differentiable and invertible functions in the transition, as in [Section %s](#multdim) of the paper. In [Section %s](#method), we have additively separable utility of consumption and leisure, which allows for each of these control variables to be handled by separate subproblems. So, it makes sense to split the utility between subproblems and attach one to the consumption subproblem and one to the leisure subproblem.

As mentioned in that section, however, there are only two separable utility functions in the problem which have been assigned to two subproblems already. This leaves one control variable without a separable utility function. In that case, there is not another Endogenous Grid Method step to exploit, and this subproblem has to be handled by standard convex optimization techniques such as maximization of the value function (VFI) or finding the root of the Euler equation (PFI).

Now that we have split the problem into conceptual subproblems, it is important to sequence them in such a way that they don't become more complex than the original problem. The key here is to avoid adding unnecessary state variables. For example, in the consumption-leisure-portfolio problem, if we were to choose consumption first, we would have to track the wage rate into the following leisure subproblem. This would mean that our consumption problem would be two-dimensional as well as our labor decision problem. As presented, the choice of order in [Section %s](#method) ensures that the consumption problem is one-dimensional, as we can shed the information about the wage rate offer after the agent has made their labor-leisure decision. If we did this the other way, the problem would be more complex and require additional computational resources.

The consumption subproblem would be two-dimensional instead of one-dimensional, adding more complexity,

\begin{equation}
    \begin{split}
        \vFunc(\bRat, \tShkEmp) & = \max_{\cRat} \uFunc(\cRat) + \vOpt(\bRat', \tShkEmp) \\
        & \text{s.t.}\\
        \bRat' & = \bRat - \cRat \ge - \tShkEmp
    \end{split}
\end{equation}

while the labor-leisure subproblem would have an additional constraint

\begin{equation}
    \begin{split}
        \vOpt(\bRat', \tShkEmp) & = \max_{\leisure} \h(\leisure) + \vEnd(\aRat) \\
        & \text{s.t.} \\
        0 & \le \leisure \le 1 \\
        \aRat & = \bRat' + \tShkEmp(1 - \leisure) \ge 0.
    \end{split}
\end{equation}

Therefore, strategic ordering of subproblems can greatly simplify the solution process and reduce the computational burden.

Consider the utility function of the form

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

In [Section %s](#multdim), we see that a problem with a differentiable and invertible transition can also be used to embed an additional Endogenous Grid Method step. Because the transition applies independently to a state variable that is not related to the other control variable, consumption, it can be handled separately from the consumption subproblem. In this particular problem, however, it turns out to make no difference how we order the two subproblems. This is because the control variables, consumption and pension deposit, each affect a separate resource account, namely market resources and pension balance. Because of this, the two subproblems are independent of each other and can be solved in any order.

A good rule of thumb is that when splitting up a problem into subproblems, we should try to reduce the information set that is passed onto the next subproblem. In [Section %s](#method), choosing leisure-labor and realizing total market resources before consumption allows us to shed the wage rate offer state variable before the consumption problem, and we know that for the portfolio choice we only need to know liquid assets after expenditures (consumption). Thus, the order makes intuitive sense; agent first chooses leisure-labor, realizing total market resources, then chooses consumption and savings, and finally chooses their risky portfolio choice. In [Section %s](#multdim), there are two expenditures that are independent of each other, consumption and deposit, and making one decision or the other first does not reduce the information set for the agent, thus the order of these subproblems does not matter.

## The Endogenous Grid Method for Subproblems

Once we have strategically split the problem into subproblems, we can use the Endogenous Grid Method in each applicable subproblem while iterating backwards from the terminal period. As demonstrated in [Section %s](#method) and [Section %s](#multdim), the EGM step can be applied when there is a separable, differentiable and invertible utility function in the subproblem or when there is a differentiable and invertible transition in the subproblem. We will discuss each of these cases in turn. A generic subproblem with a differentiable and invertible utility function can be characterized as follows:

\begin{equation}
    \begin{split}
        \VFunc(\xRat) & = \max_{\aRat \in \PGro(\xRat)} \UFunc(\xRat, \aRat) + \WFunc(\yRat) \\
        & \text{s.t.} \\
        \yRat & = \TFunc(\xRat,\aRat)
    \end{split}
\end{equation}

where $\WFunc(\yRat) = \DiscFac \Ex[\VFunc'(\yRat)]$ is the continuation value. For an interior solution, the first-order condition is thus

\begin{equation}
    \frac{\partial \UFunc(\xRat, \aRat)}{\partial \aRat} + \WFunc'(\yRat) \frac{\partial \TFunc(\xRat,\aRat)}{\partial \aRat} = 0
\end{equation}

When corner solutions occur (e.g., $\aRat$ at constraint boundaries), the unconstrained optimum from inverting the FOC must be projected onto the feasible set, as demonstrated in [Section %s](#method) for the leisure choice. If, for interior solutions, the utility function is differentiable and invertible, then the Endogenous Grid Method consists of

\begin{equation}
    \aRat = \left(\frac{\partial \UFunc(\xRat, \aRat)}{\partial \aRat}\right)^{-1}
    \left[ -\WFunc'(\yRat) \frac{\partial \TFunc(\xRat,\aRat)}{\partial \aRat}\right]
\end{equation}

By using an exogenous grid of the post-decision state $\yRat$, we can solve for the optimal decision rule $\aRat$ at each point on the grid. This is the Endogenous Grid Method step. Uniqueness of the solution is ensured when the utility function is strictly concave in $\aRat$.

## Applicability to Transition Functions

If the generic subproblem has no separable utility, but instead has differentiable and invertible transitions that affect multiple post-decision states, then the Endogenous Grid Method can still be used. Consider a problem with two endogenous state variables and two post-decision states:

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

Here, the first-order condition is

\begin{equation}
    \frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_1} \cdot \frac{\partial \TFunc_1(\xRat_1,\aRat)}{\partial \aRat} + \frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_2} \cdot \frac{\partial \TFunc_2(\xRat_2,\aRat)}{\partial \aRat} = 0
\end{equation}

If $\TFunc_2$ has the special structure $\TFunc_2(\xRat_2, \aRat) = \xRat_2 + \aRat + \gFunc(\aRat)$ where $\gFunc$ is differentiable with $\gFunc'$ strictly monotone, and $\frac{\partial \TFunc_1}{\partial \aRat}$ is constant, then we can rearrange the FOC to get

\begin{equation}
    \gFunc'(\aRat) = -\left(\frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_1} \middle/ \frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_2}\right) \cdot \frac{\partial \TFunc_1(\xRat_1,\aRat)}{\partial \aRat} - 1
\end{equation}

and the Endogenous Grid Method step is

\begin{equation}
    \aRat = \gFunc'^{-1}\left( -\left[\frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_1} \middle/ \frac{\partial \WFunc(\yRat_1, \yRat_2, \sRat)}{\partial \yRat_2}\right] \cdot \frac{\partial \TFunc_1(\xRat_1,\aRat)}{\partial \aRat} - 1 \right)
\end{equation}

where the strict monotonicity of $\gFunc'$ ensures existence and uniqueness of the inverse.
