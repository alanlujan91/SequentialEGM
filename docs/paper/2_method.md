(method)=

# The Sequential Endogenous Grid Method

The Sequential Endogenous Grid Method (EGM$^n$) is a novel extension of the EGM to solve dynamic programming problems with multiple choice variables. The key insight is to break down the problem into a sequence of subproblems, each of which can be solved using a simple EGM step. To demonstrate the power of thinking sequentially, I first present a simple model with one choice variable that is traditionally solved using one EGM step.

## The Standard Incomplete Markets (SIM) Model as a sequential problem

Consider the standard one-asset incomplete markets model by [](doi:10.1086/250034), also known as the Bewley-Huggett-Aiyagari-Imrohoroglu-Zeldes-Deaton-Carroll model. The consumer's problem is to maximize the present discounted value of utility from consumption and savings, subject to uncertainty in labor productivity and an aggregate process of interest rates and wages. The Bellman equation is

:::{include} ../equations/sim.tex
:::

We can think of this problem as a sequence of two blocks: one where the household makes a consumption-saving decision, and another where the household takes an expectation of the future conditional on their savings choice. Define the continuation value of any savings choice as

$$
\begin{align}
W_t(e, k') = \mathbb{E}_t\left[V_{t+1}(e', k')|e \right].
\end{align}
$$

Importantly, this block does not require any optimization, as it is just an expectation step. The household can then use the value of any savings decision to optimize their consumption choice. The problem is then reduced to the following:

:::{include} ../equations/sim_0.tex
:::

Notice that this block has no uncertainty. This is a key advantage of thinking sequentially. By placing the expectation in a subsequent block, this block is now a simple optimization problem.

We have now dissected this problem into two blocks, an optimization block and an expectation block, each of which is easier to handle than the whole. We now present the EGM. The first-order condition with respect to consumption is

$$
\begin{align}
c(e, k') = (\beta W_t^k(e, k'))^{-1/\sigma}
\end{align}
$$

@SolvingMicroDSOPs refer to this as the "consumed function", because it provides the amount a household must have "consumed" to optimally achieve a certain level of savings $k'$ given their current productivity state $e$. From this function, we can derive the endogenous grid of capital $k$ that the agent must have started with to consume $c(e, k')$ and save $k'$ using the resource constraint.

$$
\begin{align}
k(e, k') = \frac{c(e, k') + k' - w_t e }{1+r_t}
\end{align}
$$

These two equations now jointly define a parameterized curve[^fedor] for the consumption function $c(e, k)$, which can be interpolated to find the optimal consumption for any level of savings and productivity. This is the EGM step.

[^fedor]: This comes from Fedor Iskhakov.

<!-- :::{raw} latex
\begin{algorithm}
\caption{The Endogenous Grid Method (EGM)}\label{alg:egm}
\begin{algorithmic}
\Require $V_{t+1}(e, k)$
\Ensure $y = x^n$
\State $y \gets 1$
\State $X \gets x$
\State $N \gets n$
\While{$N \neq 0$}
\If{$N$ is even}
\State $X \gets X \times X$
\State $N \gets \frac{N}{2}$  \\ Comment{This is a comment}
\ElsIf{$N$ is odd}
\State $y \gets y \times X$
\State $N \gets N - 1$
\EndIf
\EndWhile
\end{algorithmic}
\end{algorithm}
::: -->

## The Heterogeneous Agent New Keynesian (HANK) Model

The baseline problem which I will use to demonstrate the Sequential Endogenous Grid Method (EGM$^n$) is the one-asset HANK model with endogenous labor from [](doi:10.3982/ECTA17434).

In particular, this example makes use of an additively separable utility of consumption and disutility of labor as follows:

:::{include} ../equations/hank.tex
:::

The use of additively separable utility is important, as it will allow for the use of multiple EGM steps in the solution process, as we'll see later.

This model represents a consumer who begins the period with a level of bank balances $\bRat_{t}$ and a given wage offer $\tShkEmp_{t}$. Simultaneously, they are able to choose consumption, labor intensity, and a risky portfolio share with the objective of maximizing their utility of consumption and leisure, as well as their future wealth.

in which $\labor_{t}$ is the time supplied to labor net of leisure, $\mRat_{t}$ is the market resources totaling bank balances and labor income, $\aRat_{t}$ is the amount of saving assets held by the consumer, and $\riskyshare_{t}$ is the risky share of assets, which induce a $\Rport_{t+1}$ return on portfolio that results in next period's bank balances $\bRat_{t+1}$ normalized by next period's permanent income $\PGro_{t+1}$.

## Restating the problem sequentially

Before we continue, it will be useful to introduce a new notation for timing. In dynamic programming, we often think of the period-$t$ problem. However, when thinking sequentially, it will be useful to think about breaking the period into sub-periods, which we'll call moments. When using EGM$^n$, we will index these moments as $(t, \tau)$ for the period-$t$ and moment-$\tau$ subproblem. We will also use lowercase letters to denote the moment-specific value functions, $v_{(t,\tau)}(\cdot)$, and the decision rules, $\pi_{(t,\tau)}(\cdot)$.

We can make a few choices to create a sequential problem which will allow us to use multiple EGM steps in succession. First, the agent decides their labor-leisure trade-off and receives a wage. Their wage plus their previous bank balance then becomes their market resources. Second, given market resources, the agent makes a pure consumption-saving decision. Finally, given an amount of savings, the consumer then decides their risky portfolio share.

Starting from the beginning of the period, we can define the labor-leisure problem as

:::{include} ../equations/hank_1.tex
:::

The pure consumption-saving problem is then

:::{include} ../equations/hank_2.tex
:::

Finally, the expectation block is

:::{include} ../equations/hank_3.tex
:::

Notice that we started with moment $(t,1)$. We want to think of moment $(t,0)$ as the moment where uncertainty is resolved. Therefore, the agent first learns $r_t$ and $w_t(e)$ in moment $(t,0)$ and then uses this information to solve the consumption-savings problem in moment $(t,1)$.

This sequential approach is explicitly modeled after the nested approaches explored in {cite:t}`Clausen2020` and {cite:t}`Druedahl2021`. However, I will offer additional insights that expand on these methods. An important observation is that now, every single choice is self-contained in a subproblem, and although the structure is specifically chosen to minimize the number of state variables at every stage, the problem does not change by this structural imposition. This is because there is no additional information or realization of uncertainty that happens between decisions, as can be seen by the expectation operator being in the last subproblem. From the perspective of the consumer, these decisions are essentially simultaneous, but a careful organization into sub-period problems enables us to solve the model more efficiently and can provide key economic insights. In this problem, as we will see, a key insight will be the ability to explicitly calculate the marginal value of wealth and the Frisch elasticity of labor.

## The labor-leisure subproblem

The labor-leisure subproblem can be restated more compactly as:

$$
\begin{equation}
    \vFunc_{t}(\bRat_{t}, \tShkEmp_{t}) = \max_{ \leisure_{t}}
    \h(\leisure_{t}) + \vOpt_{t}(\bRat_{t} +
    \tShkEmp_{t}(1-\leisure_{t}))
\end{equation}
$$

The first-order condition with respect to leisure implies the labor-leisure Euler equation

$$
\begin{equation}
    \h'(\leisure_{t}) = \vOpt_{t}'(\mRat_{t}) \tShkEmp_{t}
\end{equation}
$$

The marginal utility of leisure and its inverse are

$$
\begin{equation}
    \h'(\leisure) = \labShare\leisure^{-\leiShare} \qquad
    \h'^{-1}(\xRat) = (\xRat/\labShare)^{-1/\leiShare}
\end{equation}
$$

Using an exogenous grid of $\mMat$ and $\tShkMat$, we can find leisure as

$$
\begin{equation}
    \zEndFunc_{t}(\mMat, \tShkMat) = \h'^{-1}\left(
    \vOpt_{t}'(\mMat) \tShkMat \right)
\end{equation}
$$

In this case, it's important to note that there are conditions for leisure itself. An agent with a small level of market resources $\mRat_{t}$ might want to work more than their available time endowment, especially at higher levels of income $\tShkEmp_{t}$, if the utility of leisure is not enough to compensate for their low wealth. In these situations, the optimal unconstrained leisure might be negative, so we must impose a constraint on the optimal leisure function. This is similar to the treatment of an artificial borrowing constraint in the pure consumption subproblem. From now on, let's call this constrained optimal function $\hat{\zEndFunc}_{t}(\mMat, \tShkMat)$, where

$$
\begin{equation}
    \hat{\zEndFunc}_{t}(\mMat, \tShkMat) = \max \left[ \min \left[ \zEndFunc_{t}(\mMat, \tShkMat), 1 \right], 0 \right]
\end{equation}
$$

Then, we derive labor as $\lEndFunc_{t}(\mRat_{t}, \tShkEmp_{t}) = 1 - \hat{\zEndFunc}_{t}(\mRat_{t}, \tShkEmp_{t})$. Finally, for each $\tShkEmp_{t}$ and $\mRat_{t}$ as an exogenous grid, we can find the endogenous grid of bank balances as $\bEndFunc_{t}(\mRat_{t}, \tShkEmp_{t}) = \mRat_{t} - \tShkEmp_{t}\lEndFunc_{t}(\mRat_{t}, \tShkEmp_{t})$.

The envelope condition then provides a heterogeneous Frisch elasticity of labor as simply

$$
\begin{equation}
    \vFunc_{t}^{b}(\bRat_{t}, \tShkEmp_{t}) = \vOpt_{t}'(\mRat_{t}) =
    \h'(\leisure_{t})/\tShkEmp_{t}.
\end{equation}
$$

:::{include} ../equations/solving_hank_1.tex
:::

## The consumption-saving subproblem

The consumption-saving EGM follows {cite:t}`Carroll2006` but I will cover it for exposition. We can begin the solution process by restating the consumption-savings subproblem in a more compact form, substituting the market resources constraint and ignoring the no-borrowing constraint for now. The problem is:

$$
\begin{equation}
    \vOpt_{t}(\mRat_{t}) = \max_{\cRat_{t}} \util(\cRat_{t}) +
    \DiscFac \vEnd_{t}(\mRat_{t}-\cRat_{t})
\end{equation}
$$

To solve, we derive the first-order condition with respect to $\cRat_{t}$ which gives the familiar Euler equation:

$$
\begin{equation}
    \utilFunc'(\cRat_t) = \DiscFac \vEnd_{t}'(\mRat_{t} - \cRat_{t}) = \DiscFac
    \vEnd_{t}'(\aRat_{t})
\end{equation}
$$

Inverting the above equation is the (first) EGM step.

$$
\begin{equation}
    \cEndFunc_{t}(\aRat_{t}) = \utilFunc'^{-1}\left( \DiscFac \vEnd_{t}'(\aRat_{t})
    \right)
\end{equation}
$$

Given the utility function above, the marginal utility of consumption and its inverse are

$$
\begin{equation}
    \utilFunc'(\cRat) = \cRat^{-\CRRA} \qquad \utilFunc'^{-1}(\xRat) =
    \xRat^{-1/\CRRA}.
\end{equation}
$$

{cite:t}`Carroll2006` demonstrates that by using an exogenous grid of $\aMat$ points we can find the unique $\cEndFunc_{t}(\aMat)$ that optimizes the consumption-saving problem, since the first-order condition is necessary and sufficient. Further, using the market resources constraint, we can recover the exact amount of market resources that is consistent with this consumption-saving decision as

$$
\begin{equation}
    \mEndFunc_{t}(\aMat) = \cEndFunc_{t}(\aMat) + \aMat.
\end{equation}
$$

This $\mEndFunc_{t}(\aMat)$ is the \`\`endogenous'' grid that is consistent with the exogenous decision grid $\aMat$. Now that we have a $(\mEndFunc_{t}(\aMat), \cEndFunc_{t}(\aMat))$ pair for each $\aRat \in \aMat$, we can construct an interpolating consumption function for market resources points that are off-the-grid.

The envelope condition will be useful in the next section, but for completeness is defined here.

$$
\begin{equation}
    \vOpt_{t}'(\mRat_{t}) = \DiscFac \vEnd_{t}'(\aRat_{t}) = \utilFunc'(\cRat_{t})
\end{equation}
$$

:::{include} ../equations/solving_hank_2.tex
:::
