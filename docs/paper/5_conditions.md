(conditions)=

# Conditions for using the Sequential Endogenous Grid Method

## Splitting the problem into subproblems

The first step in using the Sequential Endogenous Grid Method is to split the problem into subproblems. This process of splitting up the problem has to be strategic to not insert additional complexity into the original problem. If one is not careful when doing this, the subproblems can become more complex and intractable than the original problem.

To split up the problem, we first count the number of control variables or decisions faced by the agent. Ideally, if the agent has $n$ control variables, then the problem should be split into $n$ subproblems, each handling a different control variable. For counting the number of control variables, it is important to not double count variables which are equivalent and have market clearing conditions. For example, the decision of how much to consume and how much to save may seem like two different choices, but because of the market clearing condition $\cRat + \aRat = \mRat$ they are resolved simultaneously and count as only one decision variable. Similarly, the choice between labor and leisure are simultaneous and count as only one decision.

Having counted our control variables, we look for differentiable and invertible utility functions which are separable in the dynamic programming problem, such as in [Section %s](#method) of the paper, or differentiable and invertible functions in the transition, as in [Section %s](#multdim) of the paper.

%note: Capitalize Section for all instances # DONE

### Separable utility functions

In [Section %s](#method), we have additively separable utility of consumption and leisure, which allows for each of these control variables to be handled by separate subproblems. So, it makes sense to split the utility between subproblems and attach one to the consumption subproblem and one to the leisure subproblem.

As mentioned in that section, however, there are only two separable utility functions in the problem which have been assigned to two subproblems already. This leaves one control variable without a separable utility function. In that case, there is not another Endogenous Grid Method step to exploit, and this subproblem has to be handled by standard convex optimization techniques such as maximization of the value function (VFI) or finding the root of the Euler equation (PFI).

Now that we have split the problem into conceptual subproblems, it is important to sequence them in such a way that they don't become more complex than the original problem. The key here is to avoid adding unnecessary state variables. For example, in the consumption-leisure-portfolio problem, if we were to choose consumption first, we would have to track the wage rate into the following leisure subproblem. This would mean that our consumption problem would be two-dimensional as well as our labor decision problem. As presented, the choice of order in [Section %s](#method) ensures that the consumption problem is one-dimensional, as we can shed the information about the wage rate offer after the agent has made their labor-leisure decision. If we did this the other way, the problem would be more complex and require additional computational resources.

The consumption subproblem would be two-dimensional instead of one-dimensional, adding more complexity,

while the labor-leisure subproblem would have an additional constraint

Therefore, strategic ordering of subproblems can greatly simplify the solution process and reduce computational the burden.

Consider the utility function of the form

where $\yRat^{i}$ is the $i$-th control variable and $\yRat^{-i}$ is the vector of all control variables except the $i$-th one.

which is separable in the state and control variables that correspond to the index $i$.

For simplicity, define

then

the first order condition

we require $\frac{\partial \TFunc_{t}^j(\xRat_t, \yRat_t)}{\partial \yRat_t^i} = 0$ for $j \neq i$ to be able to solve for $\yRat_t^i$.

### Differentiable and invertible transition

In [Section %s](#multdim), we see that a problem with a differentiable and invertible transition can also be used to embed an additional Endogenous Grid Method step. Because the transition applies independently to a state variable that is not related to the other control variable, consumption, it can be handled separately from the consumption subproblem.

%note to check the tense of the entire text

In this particular problem, however, it turns out to make no difference how we order the two subproblems. This is because the control variables, consumption and pension deposit, each affect a separate resource account, namely market resources and pension balance. Because of this, the two subproblems are independent of each other and can be solved in any order.

A good rule of thumb is that when splitting up a problem into subproblems, we should try to reduce the information set that is passed onto the next subproblem. In [Section %s](#method), choosing leisure-labor and realizing total market resources before consumption allows us to shed the wage rate offer state variable before the consumption problem, and we know that for the portfolio choice we only need to know liquid assets after expenditures (consumption). Thus, the order makes intuitive sense; agent first chooses leisure-labor, realizing total market resources, then chooses consumption and savings, and finally chooses their risky portfolio choice. In [Section %s](#multdim), there are two expenditures that are independent of each other, consumption and deposit, and making one decision or the other first does not reduce the information set for the agent, thus the order of these subproblems does not matter.

## The Endogenous Grid Method for Subproblems

Once we have strategically split the problem into subproblems, we can use the Endogenous Grid Method in each applicable subproblem while iterating backwards from the terminal period. As we discussed in Sections [Section %s](#method) and [Section %s](#multdim), the EGM step can be applied when there is a separable, differentiable and invertible utility function in the subproblem or when there is a differentiable and invertible transition in the subproblem. We will discuss each of these cases in turn.

### Utility function

A generic subproblem with a differentiable and invertible utility function can be characterized as follows:

For an interior solution, the first-order condition is thus

If, as we assumed, the utility function is differentiable and invertible, then the Endogenous Grid Method consists of

By using an exogenous grid of the post-decision state $\aRat$, we can solve for the optimal decision rule $\yRat$ at each point on the grid. This is the Endogenous Grid Method step.

### Transition

If the generic subproblem has no separable utility, but instead has a differentiable and invertible transition, then the Endogenous Grid Method can still be used.

Here, the first-order condition is

and the Endogenous Grid Method step is

## Extending the Triangular Dynamic Optimization Theorem

In this section, we extend the Triangular Dynamic Optimization Theorem from [](doi:10.1016/j.econlet.2015.07.033) to allow for a more general class of multidimensional dynamic optimization problems.

Consider a dynamic stochastic optimization problem with $m$ endogenous and continuous state variables $x_t = (x^1_t, x^2_t, \ldots, x^m_t)'$ and $n$ exogenous (and potentially discrete) state variables[^exog_agg] $\varepsilon_t = (\varepsilon^1_t, \varepsilon^2_t, \ldots, \varepsilon^n_t)'$ which follow a Markov process $\varepsilon_{t+1} \lvert \varepsilon_t \sim F$. The agent solves this problem by choosing $m$ continuous action variables $a_t = (a^1_t, a^2_t, \ldots, a^m_t)'$ simultaneously. Let $y_t = (y^1_t, y^2_t, \ldots, y^m_t)'$ be the vector of post-decision state variables, such that:

[^exog_agg]: The exogenous state vector can be composed of both discrete and continuous state variables, which will not affect the following analysis. Alternatively, we can think of this exogenous state vector as being aggregate shocks to the agent's environment.

$$
\begin{equation}
    y_t = g(x_t, a_t, \varepsilon_t).
\end{equation}
$$

This implies that our post-decision state vector is a *sufficient statistic* of the current period's state and action, conditional on the exogenous state vector, which means that it contains all the information necessary to determine the next period's state. This can be seen in the following expression

$$
\begin{equation}
    x_{t+1} = f(y_t, \xi_{t+1})
\end{equation}
$$

where $\xi_{t+1}$ is a vector of exogenous, independent, and idiosyncratic shocks to the agent's problem. We will call $g(\cdot)$ the *transition function* and $f(\cdot)$ the *next-period state function*.

The objective of the agent is to maximize the present discounted utility of its actions $u(a, \varepsilon)$ as follows:

$$
\begin{equation}
    v_0(x_0, \varepsilon_0) = \max_{a_t} \sum_{t=0}^T \beta^t u(a_t, \varepsilon_t).
\end{equation}
$$

Finally, we can define the recursive dynamic programming problem as:

$$
\begin{align}
v_t(x_t, \varepsilon_t) & = \max_{a_t \in \Gamma_t(x_t, \varepsilon_t)} \left\{ u(a_t, \varepsilon_t) + \beta \mathbb{E}_t \left[ v_{t+1}(x_{t+1}, \varepsilon_{t+1}) \lvert  x_t, a_t, \varepsilon_t \right] \right\} \\
    y_t & = g(x_t,  a_t, \varepsilon_t) \\
    x_{t+1} & = f(y_t, \xi_{t+1})
\end{align}
$$

where $\Gamma_t(\cdot)$ is the set of feasible actions at time $t$.

### The Continuation Value Function

The first set of assumptions we make will allow us to reduce the problem to a non-stochastic one.

:::{prf:assumption} Independence of Transition Probabilities

The transition probabilities $\mathbb{P}(\varepsilon_{t+1} | x_t, \varepsilon_t) = \mathbb{P}(\varepsilon_{t+1} | \varepsilon_t)$ are independent of $x_t$.

:::

This assumption allows us to factor the expectation out of the recursive problem since $\mathbb{E}_t[\cdot \lvert x_t, a_t, \varepsilon_t] = \mathbb{E}_t[\cdot \lvert y_t, \varepsilon_t]$. Then, we can define the following function

$$
\begin{align}
w(y_t, \varepsilon_t) & = \mathbb{E}_t \left[ v_{t+1}(x_{t+1}, \varepsilon_{t+1}) \lvert y_t, \varepsilon_t \right] \\
x_{t+1} & = f(y_t, \xi_{t+1})
\end{align}
$$

and substitute it into the original problem to obtain the following problem

$$
\begin{align}
v_t(x_t, \varepsilon_t) & = \max_{a_t \in \Gamma_t(x_t, \varepsilon_t)} \left\{ u(a_t, \varepsilon_t) + \beta w(y_t, \varepsilon_t) \right\} \\
    y_t & = g(x_t, a_t, \varepsilon_t)
\end{align}
$$

This is now a non-stochastic problem, which simplifies our analysis considerably. We will refer to $w(\cdot)$ as the *continuation value function*, which is the value function evaluated before the realization of any uncertainty. Our problem is now reduced to maximizing the trade-off between the utility of the action $a_t$ and the discounted continuation value given the consequences of said action.

:::{prf:assumption} Continuous and Differentiable Next Period State Function

The next period state function $f(\cdot)$ is continuous and differentiable. Alternatively, each of its component functions $f^i(\cdot)$ is continuous and differentiable.
:::

This assumption allows the existence of a continous derivative of $w(\cdot)$ with respect to $y_t$, which is necessary for the existence of the first-order condition.

$$
\begin{equation}
    \nabla w(y_t, \varepsilon_t) = \mathbb{E}_t \left[ \nabla v_{t+1}(f(y_t, \xi_{t+1}), \varepsilon_{t+1}) \lvert y_t, \varepsilon_t \right]
\end{equation}
$$

### Standard Assumptions for EGM

:::{prf:assumption} The Utility Function is Strictly Concave and Differentiable

The utility function $u(a_t)$ is strictly concave and continuously differentiable.
:::

As in the standard EGM, we require the utility function to be strictly concave and differentiable. This guarantees that the first-order condition is both necessary and sufficient for a maximum. In the examples presented in this paper, we used additively separable utility functions, which satisfy these conditions.

:::{prf:assumption} The Transition Function is Differentiable and Invertible

The transition function $g(\cdot)$ is differentiable and $g'(\cdot)$ is invertible. Moreover, $g'_i(\{x_t\}_1^i, \{a_t\}_1^i, \varepsilon_t)$ can be analytically solved for $x^i_t$ while holding all other arguments fixed.

$$
\begin{equation}
    x^i_t = {g'_i}^{-1}(\{x_t\}_1^{i-1}, \{a_t\}_1^i, \varepsilon_t)
\end{equation}
$$
:::

This condition allows us to work backwards from the post-decision state $y_t$ and the action $a_t$ to obtain the pre-decision state $x_t$ that optimally corresponds to them.

As [](doi:10.1016/j.econlet.2015.07.033) shows, more assumptions are needed for the multidimensional case.

### Triangular Assumptions

:::{prf:assumption} Triangular Utility
some theorem
:::

:::{prf:assumption} Triangular Transitions

Assume that the vector of post-decision state variables follows a triangular structure

$$
\begin{align}
y^1_t & = g^1(x_t, a_t, \varepsilon_t) = g^1(x^1_t, a^1_t, \varepsilon_t) \\
y^2_t & = g^2(x_t, a_t, \varepsilon_t) = g^2(x^1_t, a^1_t,  x^2_t, a^2_t, \varepsilon_t) \\
y^3_t & = g^3(x_t, a_t, \varepsilon_t) = g^3(x^1_t, a^1_t,  x^2_t, a^2_t, x^3_t, a^3_t, \varepsilon_t)
\end{align}
$$
:::

### Triangular Dynamic Optimization

:::{prf:theorem} Triangular Dynamic Optimization

Under assumptions 1-5, the dynamic optimization problem is triangular and admits a solution method which avoids all root-finding operations.
:::

:::{prf:proof}
$$
\begin{equation}
 u'_j(a_t) = \beta \frac{\partial w(y_t, \varepsilon_t)}{\partial y^j_t} \frac{\partial g^j(x_t, a_t, \varepsilon_t)}{\partial a^j_t}
\end{equation}
$$
:::

### Sequential Dyamic Optimization

:::{prf:theorem} Sequential Dynamic Optimization

Under assumptions 1-5, the dynamic optimization problem is triangular and can be solved sequentially.
:::

:::{prf:proof}

:::
