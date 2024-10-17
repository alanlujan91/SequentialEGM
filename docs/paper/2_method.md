(method)=

# The Sequential Endogenous Grid Method

The Sequential Endogenous Grid Method (EGM$^n$) is a novel extension of the EGM to solve dynamic programming problems with multiple choice variables. The key insight is to break down the problem into a sequence of subproblems, each of which can be solved using a simple EGM step. To demonstrate the power of thinking sequentially, I first present a simple model with one choice variable that is traditionally solved using one EGM step.

## The Standard Incomplete Markets model as a sequential problem

Consider the standard one-asset incomplete markets model by [](doi:10.1086/250034), also known as the Bewley-Huggett-Aiyagari-Imrohoroglu-Zeldes-Deaton-Carroll model. The consumer's problem is to maximize the present discounted value of utility from consumption and savings, subject to uncertainty in labor productivity and an aggregate process of interest rates and wages. The Bellman equation is

$$
\begin{align}
V_t(e, k) = \max_{c, k'} &\left\{\frac{c^{1-\sigma}}{1-\sigma} + \beta \mathbb{E}_t\left[V_{t+1}(e', k')|e \right] \right\}
\\
c + k' &= (1 + r_t)k + w_t e
\\
k' &\geq 0
\end{align}
$$

We can think of this problem as a sequence of two blocks: one where the household makes a consumption-saving decision, and another where the household takes an expectation of the future conditional on their savings choice. Define the continuation value of any savings choice as

$$
\begin{align}
W_t(e, k') = \mathbb{E}_t\left[V_{t+1}(e', k')|e \right].
\end{align}
$$

Importantly, this block does not require any optimization, as it is just an expectation step. The household can then use the value of any savings decision to optimize their consumption choice. The problem is then reduced to the following:

$$
\begin{align}
V_t(e, k) = \max_{c, k'} &\left\{\frac{c^{1-\sigma}}{1-\sigma} + \beta W_t(e, k') \right\}
\\ c + k' &= (1 + r_t)k + w_t e
\\ k' &\geq 0
\end{align}
$$

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

:::{raw} latex
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
:::

## The Heterogeneous Agent New Keynesian (HANK) Model

The baseline problem which I will use to demonstrate the Sequential Endogenous Grid Method (EGM$^n$) is the one-asset HANK model with endogenous labor from [](doi:10.3982/ECTA17434).

In particular, this example makes use of an additively separable utility of consumption and disutility of labor as follows:

$$
\begin{align}
V_t(e, a) = \max_{c, n, a'} &\left\{\frac{c^{1-\sigma}}{1-\sigma} - \varphi \frac{n^{1+\nu}}{1+\nu} + \beta \mathbb{E}_t\left[V_{t+1}(e', a')|e\right] \right\}
\\
c + a' &= (1 + r_t)a + w_t(e) n + T_t(e)
\\
a' &\geq 0
\end{align}
$$

The use of additively separable utility is important, as it will allow for the use of multiple EGM steps in the solution process, as we'll see later.

This model represents a consumer who begins the period with a level of bank balances $\bRat_{t}$ and a given wage offer $\tShkEmp_{t}$. Simultaneously, they are able to choose consumption, labor intensity, and a risky portfolio share with the objective of maximizing their utility of consumption and leisure, as well as their future wealth.

in which $\labor_{t}$ is the time supplied to labor net of leisure, $\mRat_{t}$ is the market resources totaling bank balances and labor income, $\aRat_{t}$ is the amount of saving assets held by the consumer, and $\riskyshare_{t}$ is the risky share of assets, which induce a $\Rport_{t+1}$ return on portfolio that results in next period's bank balances $\bRat_{t+1}$ normalized by next period's permanent income $\PGro_{t+1}$.

## Restating the problem sequentially

We can make a few choices to create a sequential problem which will allow us to use multiple EGM steps in succession. First, the agent decides their labor-leisure trade-off and receives a wage. Their wage plus their previous bank balance then becomes their market resources. Second, given market resources, the agent makes a pure consumption-saving decision. Finally, given an amount of savings, the consumer then decides their risky portfolio share.

Starting from the beginning of the period, we can define the labor-leisure problem as

$$
\begin{align}
v_{(t,0)}(e, a) &= \max_{n} \left\{ - \varphi \frac{n^{1+\nu}}{1+\nu} + v_{(t,1)}(e,m) \right\}
\
m &= (1 + r_t)a + w_t(e) n + T_t(e)
\end{align}
$$

The pure consumption-saving problem is then

$$
\begin{align}
v_{(t,1)}(e, m) &= \max_{c, a'} \left\{\frac{c^{1-\sigma}}{1-\sigma}  + \beta v_{(t,2)}(e,a') \right\}
\
 a' &= m - c \geq 0
\end{align}
$$

Finally, the risky portfolio problem is

$$
\begin{align}
v_{(t,2)}(e, a') =  \mathbb{E}_t\left[V_{t+1}(e', a')|e\right]
\end{align}
$$

This sequential approach is explicitly modeled after the nested approaches explored in {cite:t}`Clausen2020` and {cite:t}`Druedahl2021`. However, I will offer additional insights that expand on these methods. An important observation is that now, every single choice is self-contained in a subproblem, and although the structure is specifically chosen to minimize the number of state variables at every stage, the problem does not change by this structural imposition. This is because there is no additional information or realization of uncertainty that happens between decisions, as can be seen by the expectation operator being in the last subproblem. From the perspective of the consumer, these decisions are essentially simultaneous, but a careful organization into sub-period problems enables us to solve the model more efficiently and can provide key economic insights. In this problem, as we will see, a key insight will be the ability to explicitly calculate the marginal value of wealth and the Frisch elasticity of labor.

## The portfolio decision subproblem

As useful as it is to be able to use the EGM step more than once, there are clear problems where the EGM step is not applicable. This basic labor-portfolio choice problem demonstrates where we can use an additional EGM step, and where we can not. First, we go over a subproblem where we can not use the EGM step.

In reorganizing the labor-portfolio problem into subproblems, we assigned the utility of leisure to the leisure-labor subproblem and the utility of consumption to the consumption-savings subproblem. There are no more separable convex utility functions to assign to this problem, and even if we re-organized the problem in a way that moved one of the utility functions into this subproblem, they would not be useful in solving this subproblem via EGM as there is no direct relation between the risky share of portfolio and consumption or leisure. Therefore, the only way to solve this subproblem is through standard convex optimization and root-finding techniques.

Restating the problem in compact form gives

$$

\begin{equation}
\vEnd_{t}(\aRat_{t}) = \max_{\riskyshare_{t}} \Ex_{t} \left\[ \PGro_{t+1}^{1-\CRRA}
\vFunc_{t+1}\left(\aRat_{t}(\Rfree + (\Risky_{t+1} - \Rfree) \riskyshare_{t}), \tShkEmp_{t+1}\right)
\right\].
\end{equation}
\$\$

The first-order condition with respect to the risky portfolio share is then

$$
\begin{equation}
    \Ex_{t}
    \left[ \PGro_{t+1}^{-\CRRA} \vFunc_{t+1}^{\bRat}\left(\bRat_{t+1}, \tShkEmp_{t+1}\right) (\Risky_{t+1} - \Rfree)  \right] =
    0.
\end{equation}
$$

Finding the optimal risky share requires numerical optimization and root-solving of the first-order condition. To close out the problem, we can calculate the envelope condition as

$$
\begin{equation}
    \vEnd_{t}'(\aRat_{t}) = \Ex_{t}
    \left[ \PGro_{t+1}^{-\CRRA} \vFunc_{t+1}^{\bRat}\left(\bRat_{t+1}, \tShkEmp_{t+1}\right) \Rport_{t+1} \right].
\end{equation}
$$

### A note on avoiding taking expectations more than once

We could instead define the portfolio choice subproblem as:

$$
\begin{equation}
    \vEnd_{t}(\aRat_{t}) = \max_{\riskyshare_{t}} \vOptAlt(\aRat_{t}, \riskyshare_{t})
\end{equation}
$$

where

$$
\begin{equation}
    \begin{split}
        \vOptAlt_{t}(\aRat_{t}, \riskyshare_{t}) & = \Ex_{t}
        \left[ \PGro_{t+1}^{1-\CRRA} \vFunc_{t+1}\left(\bRat_{t+1}, \tShkEmp_{t+1}\right)   \right] \
        \Rport_{t+1} & = \Rfree + (\Risky_{t+1} - \Rfree) \riskyshare_{t} \
        \bRat_{t+1} & = \aRat_{t} \Rport_{t+1} / \PGro_{t+1}
    \end{split}
\end{equation}
$$

In this case, the process is similar. The only difference is that we don't have to take expectations more than once. Given the next period's solution, we can calculate the marginal value functions as:

$$
\begin{equation}
    \begin{split}
        \vOptAlt_{t}^{\aRat}(\aRat_{t}, \riskyshare_{t}) & = \Ex_{t}
        \left[ \PGro_{t+1}^{-\CRRA} \vFunc_{t+1}'\left(\bRat_{t+1}, \tShkEmp_{t+1}\right) \Rport_{t+1} \right] \
        \vOptAlt_{t}^{\riskyshare}(\aRat_{t}, \riskyshare_{t}) & = \Ex_{t}
        \left[ \PGro_{t+1}^{-\CRRA} \vFunc_{t+1}'\left(\bRat_{t+1}, \tShkEmp_{t+1}\right) \aRat_{t} (\Risky_{t+1} - \Rfree)   \right] \
    \end{split}
\end{equation}
$$

If we are clever, we can calculate both of these in one step. Now, the optimal risky share can be found by the first-order condition and we can use it to evaluate the envelope condition.

$$
\begin{equation}
    \text{F.O.C.:} \qquad \vOptAlt_{t}^{\riskyshare}(\aRat_{t}, \riskyshare_{t}^{*})  = 0 \qquad
    \text{E.C.:} \qquad \vEnd_{t}^{\aRat}(\aRat_{t}) = \vOptAlt_{t}^{\aRat}(\aRat_{t}, \riskyshare_{t}^{*})
\end{equation}
$$

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

## Alternative Parametrization

An alternative formulation for the utility of leisure is to state it in terms of the disutility of labor as in (references)

$$
\begin{equation}
    \h(\labor) = - \leiShare \frac{\labor^{1+\labShare}}{1+\labShare}
\end{equation}
$$

In this case, we can restate the problem as

$$
\begin{equation}
    \h(\leisure) = - \leiShare
    \frac{(1-\leisure)^{1+\labShare}}{1+\labShare}
\end{equation}
$$

The marginal utility of leisure and its inverse are

$$
\begin{equation}
    \h'(\leisure) = \leiShare(1-\leisure)^{\labShare} \qquad
    \h'^{-1}(\xRat) = 1 - (\xRat/\leiShare)^{1/\labShare}
\end{equation}
$$
