
(conditions)=

# Conditions for using the Sequential Endogenous Grid Method

## Splitting the problem into subproblems

The first step in using the Sequential Endogenous Grid Method is to split the problem into subproblems. This process of splitting up the problem has to be strategic to not insert additional complexity into the original problem. If one is not careful when doing this, the subproblems can become more complex and intractable than the original problem.

% Comment: Need to decide if it is subproblem or subproblem # DONE

To split up the problem, we first count the number of control variables or decisions faced by the agent. Ideally, if the agent has $n$ control variables, then the problem should be split into $n$ subproblems, each handling a different control variable. For counting the number of control variables, it is important to not double count variables which are equivalent and have market clearing conditions. For example, the decision of how much to consume and how much to save may seem like two different choices, but because of the market clearing condition $\cRat + \aRat = \mRat$ they are resolved simultaneously and count as only one decision variable. Similarly, the choice between labor and leisure are simultaneous and count as only one decision.

Having counted our control variables, we look for differentiable and invertible utility functions which are separable in the dynamic programming problem, such as in [Section %s](#method) of the paper, or differentiable and invertible functions in the transition, as in [Section %s](#multdim) of the paper.

%note: Capitalize Section for all instances # DONE

### Separable utility functions

In [Section %s](#method), we have additively separable utility of consumption and leisure, which allows for each of these control variables to be handled by separate subproblems. So, it makes sense to split the utility between subproblems and attach one to the consumption subproblem and one to the leisure subproblem.

As mentioned in that section, however, there are only two separable utility functions in the problem which have been assigned to two subproblems already. This leaves one control variable without a separable utility function. In that case, there is not another Endogenous Grid Method step to exploit, and this subproblem has to be handled by standard convex optimization techniques such as maximization of the value function (VFI) or finding the root of the Euler equation (PFI).

%note: spell out small numbers except for when talking about a [Section %s](#multdim) etc

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

Therefore, strategic ordering of subproblems can greatly simplify the solution process and reduce computational the burden.

Consider the utility function of the form

\begin{equation}
    \UFunc( \yRat) = \uFunc_{-i}( \yRat^{-i}) + \uFunc_i(\yRat^i)
\end{equation}

where $\yRat^{i}$ is the $i$-th control variable and $\yRat^{-i}$ is the vector of all control variables except the $i$-th one.

which is separable in the state and control variables that correspond to the index $i$.

\begin{equation}
    \begin{split}
    \VFunc_{t}(\xRat_t, \sRat_t) &= \max_{\yRat_t \in \Gamma_t(\xRat_t, \sRat_t)} \UFunc(\yRat_t)  + \DiscFac \Ex_{t} \left[ \VFunc_{t+1}(\xRat_{t+1}, \sRat_{t+1}) | \tilde{\xRat}_t,  \sRat_t \right] \\
    & \text{s.t.} \\
    \tilde{\xRat}_t &= \TFunc_t(\xRat_t, \yRat_t) \\
    \xRat_{t+1} &= \GFunc_{t+1}(\tilde{\xRat}_t, \sRat_t) \\
    \end{split}
\end{equation}

For simplicity, define

\begin{equation}
    \WFunc_t(\tilde{\xRat}_t, \sRat_t) = \DiscFac \Ex_{t} \left[ \VFunc_{t+1}(\GFunc_{t+1}(\tilde{\xRat}_t, \sRat_t), \sRat_{t+1}) | \tilde{\xRat}_t, \sRat_t \right]
\end{equation}

then

\begin{equation}
    \begin{split}
    \VFunc_{t}(\xRat_t, \sRat_t) &= \max_{\yRat_t \in \Gamma_t(\xRat_t, \sRat_t)} \UFunc( \yRat_t)  +  \WFunc_t(\tilde{\xRat}_t, \sRat_t) \\
    & \text{s.t.} \\
    \tilde{\xRat}_t &= \TFunc_t(\xRat_t, \yRat_t)
    \end{split}
\end{equation}

the first order condition

\begin{equation}
    \frac{\partial \UFunc( \yRat_t)}{\partial \yRat_t^i}  +  \sum_{j=1}^{n} \frac{\partial \WFunc_t(\tilde{\xRat}_t, \sRat_t)}{\partial \tilde{\xRat}_{t}^j} \frac{\partial \TFunc_{t}^j(\xRat_t, \yRat_t)}{\partial \yRat_t^i} = 0
\end{equation}

we require $\frac{\partial \TFunc_{t}^j(\xRat_t, \yRat_t)}{\partial \yRat_t^i} = 0$ for $j \neq i$ to be able to solve for $\yRat_t^i$.

\begin{equation}
    \frac{\partial \UFunc( \yRat_t)}{\partial \yRat_t^i}  +   \frac{\partial \WFunc_t(\tilde{\xRat}_t, \sRat_t)}{\partial \tilde{\xRat}_{t}^i} \frac{\partial \TFunc_{t}^i(\xRat_t, \yRat_t)}{\partial \yRat_t^i} = 0
\end{equation}

### Differentiable and invertible transition

In [Section %s](#multdim), we see that a problem with a differentiable and invertible transition can also be used to embed an additional Endogenous Grid Method step. Because the transition applies independently to a state variable that is not related to the other control variable, consumption, it can be handled separately from the consumption subproblem.

%note to check the tense of the entire text

In this particular problem, however, it turns out to make no difference how we order the two subproblems. This is because the control variables, consumption and pension deposit, each affect a separate resource account, namely market resources and pension balance. Because of this, the two subproblems are independent of each other and can be solved in any order.

A good rule of thumb is that when splitting up a problem into subproblems, we should try to reduce the information set that is passed onto the next subproblem. In [Section %s](#method), choosing leisure-labor and realizing total market resources before consumption allows us to shed the wage rate offer state variable before the consumption problem, and we know that for the portfolio choice we only need to know liquid assets after expenditures (consumption). Thus, the order makes intuitive sense; agent first chooses leisure-labor, realizing total market resources, then chooses consumption and savings, and finally chooses their risky portfolio choice. In [Section %s](#multdim), there are two expenditures that are independent of each other, consumption and deposit, and making one decision or the other first does not reduce the information set for the agent, thus the order of these subproblems does not matter.

## The Endogenous Grid Method for Subproblems

Once we have strategically split the problem into subproblems, we can use the Endogenous Grid Method in each applicable subproblem while iterating backwards from the terminal period. As we discussed in Sections [Section %s](#method) and [Section %s](#multdim), the EGM step can be applied when there is a separable, differentiable and invertible utility function in the subproblem or when there is a differentiable and invertible transition in the subproblem. We will discuss each of these cases in turn.

### Utility function

A generic subproblem with a differentiable and invertible utility function can be characterized as follows:

\begin{equation}
    \begin{split}
        \VFunc(\xRat) & = \max_{\yRat \in \PGro(\xRat)} \UFunc(\xRat, \yRat) + \DiscFac \WFunc(\aRat) \\
        & \text{s.t.} \\
        \aRat & = \TFunc(\xRat,\yRat)
    \end{split}
\end{equation}

For an interior solution, the first-order condition is thus

\begin{equation}
    \frac{\partial \UFunc(\xRat, \yRat)}{\partial \yRat} + \DiscFac \WFunc'(\aRat) \frac{\partial \TFunc(\xRat,\yRat)}{\partial \yRat} = 0
\end{equation}

If, as we assumed, the utility function is differentiable and invertible, then the Endogenous Grid Method consists of

\begin{equation}
    \yRat = \left(\frac{\partial \UFunc(\xRat, \yRat)}{\partial \yRat}\right)^{-1}
    \left[ -\DiscFac \WFunc'(\aRat) \frac{\partial \TFunc(\xRat,\yRat)}{\partial \yRat}\right]
\end{equation}

By using an exogenous grid of the post-decision state $\aRat$, we can solve for the optimal decision rule $\yRat$ at each point on the grid. This is the Endogenous Grid Method step.

### Transition

If the generic subproblem has no separable utility, but instead has differentiable and invertible transitions that affect multiple post-decision states, then the Endogenous Grid Method can still be used. Consider a problem with two post-decision states:

\begin{equation}
    \begin{split}
        \VFunc(\xRat, \zRat) & = \max_{\yRat \in \PGro(\xRat, \zRat)} \WFunc(\aRat, \bRat) \\
        & \text{s.t.} \\
        \aRat & = \TFunc_1(\xRat,\yRat) \\
        \bRat & = \TFunc_2(\zRat,\yRat)
    \end{split}
\end{equation}

Here, the first-order condition is

\begin{equation}
    \WFunc^{\aRat}(\aRat, \bRat) \cdot \frac{\partial \TFunc_1(\xRat,\yRat)}{\partial \yRat} + \WFunc^{\bRat}(\aRat, \bRat) \cdot \frac{\partial \TFunc_2(\zRat,\yRat)}{\partial \yRat} = 0
\end{equation}

If $\TFunc_2$ has the special structure $\TFunc_2(\zRat, \yRat) = \zRat + \yRat + \gFunc(\yRat)$ where $\gFunc$ is differentiable and invertible, and $\frac{\partial \TFunc_1}{\partial \yRat}$ is constant, then we can rearrange the FOC to get

\begin{equation}
    \gFunc'(\yRat) = -\frac{\WFunc^{\aRat}(\aRat, \bRat)}{\WFunc^{\bRat}(\aRat, \bRat)} \cdot \frac{\partial \TFunc_1(\xRat,\yRat)}{\partial \yRat} - 1
\end{equation}

and the Endogenous Grid Method step is

\begin{equation}
    \yRat = \gFunc'^{-1}\left( -\frac{\WFunc^{\aRat}(\aRat, \bRat)}{\WFunc^{\bRat}(\aRat, \bRat)} \cdot \frac{\partial \TFunc_1(\xRat,\yRat)}{\partial \yRat} - 1 \right)
\end{equation}
