(appendix)=

# Appendix: Solving the illustrative G2EGM model with EGM$^n$

## The problem for a retired household

I designate as $\wFunc_{t}(\mRat_{t})$ the problem of a retired household at time $t$ with total resources $\mRat$. The retired household solves a simple consumption-savings problem with no income uncertainty and a certain next period pension of $\underline{\tShkEmp}$.

\begin{equation}
    \begin{split}
        \wFunc_{t}(\mRat_{t}) & = \max_{\cRat_{t}} \util(\cRat_{t}) +
        \DiscFac \wFunc_{t+1}(\mRat_{t}) \\
        & \text{s.t.} \\
        \aRat_{t} & = \mRat_{t} - \cRat_{t} \\
        \mRat_{t+1} & = \Rfree_{\aRat} \aRat_{t} +
        \underline{\tShkEmp}
    \end{split}
\end{equation}

Notice that there is no uncertainty and the household receives a retirement income $\underline{\tShkEmp}$ every period until death.

## The problem for a worker household

The value function of a worker household is

\begin{equation}
    \VFunc_{t}(\mRat_{t}, \nRat_{t}) = \Ex_\error \max \left\{
    \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Work) + \sigma_{\error}
    \error_{\Work} ,
    \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Retire) + \sigma_{\error}
    \error_{\Retire} \right\}
\end{equation}

where the choice specific problem for a working household that decides to continue working is

\begin{equation}
    \begin{split}
        \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Work) & = \max_{\cRat_{t},
            \dRat_{t}} \util(\cRat_{t}) - \kapShare + \DiscFac
        \Ex_{t} \left[
            \VFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1})
            \right] \\
        & \text{s.t.} \\
        \aRat_{t} & = \mRat_{t} - \cRat_{t} - \dRat_{t} \\
        \bRat_{t} & = \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t}) \\
        \mRat_{t+1} & = \Rfree_{\aRat} \aRat_{t} + \tShkEmp_{t+1} \\
        \nRat_{t+1} & = \Rfree_{\bRat} \bRat_{t}
    \end{split}
\end{equation}

and the choice specific problem for a working household that decides to retire is

\begin{equation}
    \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Retire) =
    \wFunc_{t}(\mRat_{t}+\nRat_{t})
\end{equation}

## Applying the Sequential EGM

The first step is to define a post-decision value function. Once the household decides their level of consumption and pension deposits, they are left with liquid assets they are saving for the future and illiquid assets in their pension account which they can't access again until retirement. The post-decision value function can be defined as

\begin{equation}
    \begin{split}
        \vEnd_{t}(\aRat_{t}, \bRat_{t}) & = \DiscFac
        \Ex_{t} \left[ \VFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1}) \right] \\
        & \text{s.t.} \\
        \mRat_{t+1} & = \Rfree_{\aRat} \aRat_{t} + \tShkEmp_{t+1} \\
        \nRat_{t+1} & = \Rfree_{\bRat} \bRat_{t}
    \end{split}
\end{equation}

Then redefine the working agent's problem as

\begin{equation}
    \begin{split}
        \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Work) & = \max_{\cRat_{t},
            \dRat_{t}} \util(\cRat_{t})  - \kapShare + \vEnd_{t}(\aRat_{t},
        \bRat_{t}) \\
        \aRat_{t} & = \mRat_{t} - \cRat_{t} - \dRat_{t} \\
        \bRat_{t} & = \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t}) \\
    \end{split}
\end{equation}

Clearly, the structure of the problem remains the same, and this is the problem that G2EGM solves. We've only moved some of the stochastic mechanics out of the problem. Now, we can apply the sequential EGM$^n$ method. Let the agent first decide $\dRat_{t}$, the deposit amount into their retirement; we will call this the deposit problem, or outer loop. Thereafter, the agent will have net liquid assets of $\lRat_{t}$ and pension assets of $\bRat_{t}$.

\begin{equation}
    \begin{split}
        \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Work) & = \max_{\dRat_{t}}
        \vOpt_{t}(\lRat_{t}, \bRat_{t}) \\
        & \text{s.t.} \\
        \lRat_{t} & = \mRat_{t} - \dRat_{t} \\
        \bRat_{t} & = \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t})
    \end{split}
\end{equation}

Now, the agent can move on to picking their consumption and savings; we can call this the pure consumption problem or inner loop.

\begin{equation}
    \begin{split}
        \vOpt_{t}(\lRat_{t}, \bRat_{t}) & = \max_{\cRat_{t}}
        \util(\cRat_{t}) - \kapShare + \vEnd_{t}(\aRat_{t}, \bRat_{t}) \\
        & \text{s.t.} \\
        \aRat_{t} & = \lRat_{t} - \cRat_{t} \\
    \end{split}
\end{equation}

Because we've already made the pension decision, the amount of pension assets does not change in this loop and it just passes through to the post-decision value function.

## Solving the problem

### Solving the Inner Consumption Saving Problem

Let's start with the pure consumption-saving problem, which we can summarize by substitution as

\begin{equation}
    \vOpt_{t}(\lRat_{t}, \bRat_{t}) = \max_{\cRat_{t}} \util(\cRat_{t}) - \kapShare +
    \vEnd_{t}(\lRat_{t} - \cRat_{t}, \bRat_{t})
\end{equation}

The first-order condition is

\begin{equation}
    \util'(\cRat_{t}) = \vEnd_{t}^{\aRat}(\lRat_{t}-\cRat_{t}, \bRat_{t}) =
    \vEnd_{t}^{\aRat}(\aRat_{t}, \bRat_{t})
\end{equation}

We can invert this Euler equation as in standard EGM to obtain the consumption function.

\begin{equation}
    \cEndFunc_{t}(\aRat_{t}, \bRat_{t}) =
    \util'^{-1}\left(\vEnd_{t}^{\aRat}(\aRat_{t}, \bRat_{t})\right)
\end{equation}

Again as before, $\lEndFunc_{t}(\aRat_{t}, \bRat_{t}) =
    \cEndFunc_{t}(\aRat_{t}, \bRat_{t}) + \aRat_{t}$. To sum up, using an exogenous grid of $(\aRat_{t}, \bRat_{t})$ we obtain the trio $(\cEndFunc_{t}(\aRat_{t},
    \bRat_{t}), \lEndFunc_{t}(\aRat_{t},
    \bRat_{t}), \bRat_{t})$ which provides an interpolating function for our optimal consumption decision rule over the
$(\lRat, \bRat)$ grid. Without loss of generality, assume $\lEndFunc_{t} =
    \lEndFunc_{t}(\aRat_{t}, \bRat_{t})$ and define the interpolating function as

\begin{equation}
    \cTarg_{t}(\lEndFunc_{t}, \bRat_{t}) \equiv \cEndFunc_{t}(\aRat_{t},
    \bRat_{t})
\end{equation}

For completeness, we derive the envelope conditions as well, and as we will see, these will be useful when solving the next section.

\begin{equation}
    \begin{split}
        \vOpt_{t}^{\lRat}(\lRat_{t}, \bRat_{t}) & =
        \vEnd_{t}^{\aRat}(\aRat_{t}, \bRat_{t}) = \util'(\cRat_{t}) \\
        \vOpt_{t}^{\bRat}(\lRat_{t}, \bRat_{t}) & =
        \vEnd_{t}^{\bRat}(\aRat_{t}, \bRat_{t})
    \end{split}
\end{equation}

### Solving the Outer Pension Deposit Problem

Now, we can move on to solving the deposit problem, which we can also summarize as

\begin{equation}
    \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Work) = \max_{\dRat_{t}}
    \vOpt_{t}(\mRat_{t}
    - \dRat_{t}, \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t}))
\end{equation}

The first-order condition is

\begin{equation}
    \vOpt_{t}^{\lRat}(\lRat_{t}, \bRat_{t})(-1) +
    \vOpt_{t}^{\bRat}(\lRat_{t}, \bRat_{t})(1+\gFunc'(\dRat_{t})) = 0
\end{equation}

Rearranging this equation gives

\begin{equation}
    \gFunc'(\dRat_{t}) = \frac{\vOpt_{t}^{\lRat}(\lRat_{t},
        \bRat_{t})}{\vOpt_{t}^{\bRat}(\lRat_{t}, \bRat_{t})} - 1
\end{equation}

Assuming that $\gFunc'(\dRat)$ exists and is invertible, we can find

\begin{equation}
    \dEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \gFunc'^{-1}\left(
    \frac{\vOpt_{t}^{\lRat}(\lRat_{t},
        \bRat_{t})}{\vOpt_{t}^{\bRat}(\lRat_{t},
        \bRat_{t})} - 1 \right)
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

In sum, given an exogenous grid $(\lRat_{t}, \bRat_{t})$ we obtain the triple
$\left(\mEndFunc_{t}(\lRat_{t}, \bRat_{t}), \nEndFunc_{t}(\lRat_{t},
        \bRat_{t}), \dEndFunc_{t}(\lRat_{t}, \bRat_{t})\right)$, which we can use to create an interpolator for the decision rule $\dRat_{t}$.

To close the solution method, the envelope conditions are

\begin{equation}
    \begin{split}
        \vFunc_{t}^{\mRat}(\mRat_{t}, \nRat_{t}, \Work) & =
        \vOpt_{t}^{\lRat}(\lRat_{t}, \bRat_{t}) \\
        \vFunc_{t}^{\nRat}(\mRat_{t}, \nRat_{t}, \Work) & =
        \vOpt_{t}^{\bRat}(\lRat_{t}, \bRat_{t})
    \end{split}
\end{equation}

## Is g invertible?

We've already seen that $\util'(\cdot)$ is invertible, but is $\gFunc$?

\begin{equation}
    \gFunc(\dRat) = \xFer \log(1+\dRat) \qquad \gFunc'(\dRat) =
    \frac{\xFer}{1+\dRat} \qquad \gFunc'^{-1}(y) = \xFer/y - 1
\end{equation}

## The Post-Decision Value and Marginal Value Functions

\begin{equation}
    \begin{split}
        \vEnd_{t}(\aRat, \bRat) & = \DiscFac \Ex_{t} \left[
            \VFunc(\mRat_{t+1}, \nRat_{t+1}) \right] \\
        & \text{s.t.} \\
        \mRat_{t+1} & = \Rfree_{\aRat} \aRat_{t} + \tShkEmp_{t+1} \\
        \nRat_{t+1} & = \Rfree_{\bRat} \bRat_{t}
    \end{split}
\end{equation}

and

\begin{equation}
    \begin{split}
        \vEnd_{t}^{\aRat}(\aRat_{t}, \bRat_{t}) & = \DiscFac
        \Rfree_{\aRat} \Ex_{t} \left[ \VFunc^{\mRat}_{t+1}(\mRat_{t+1},
            \nRat_{t+1})
            \right] \\
        & \text{s.t.} \\
        \mRat_{t+1} & = \Rfree_{\aRat} \aRat_{t} + \tShkEmp_{t+1} \\
        \nRat_{t+1} & = \Rfree_{\bRat} \bRat_{t}
    \end{split}
\end{equation}

and

\begin{equation}
    \begin{split}
        \vEnd_{t}^{\bRat}(\aRat_{t}, \bRat_{t}) & = \DiscFac
        \Rfree_{\bRat} \Ex_{t} \left[ \VFunc^{\nRat}_{t+1}(\mRat_{t+1},
            \nRat_{t+1})
            \right] \\
        & \text{s.t.} \\
        \mRat_{t+1} & = \Rfree_{\aRat} \aRat_{t} + \tShkEmp_{t+1} \\
        \nRat_{t+1} & = \Rfree_{\bRat} \bRat_{t}
    \end{split}
\end{equation}

## Taste Shocks

From discrete choice theory and from DCEGM paper, we know that

\begin{equation}
    \Ex_{t} \left[
        \VFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1}, \error_{t+1}) \right] =
    \sigma \log \left[ \sum_{\Decision \in \{\Work, \Retire\}} \exp \left(
        \frac{\vFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1},
            \Decision)}{\sigma_\error} \right)  \right]
\end{equation}

and

\begin{equation}
    \Prob_{t}(\Decision ~ \lvert ~ \mRat_{t+1}, \nRat_{t+1}) = \frac{\exp
        \left(
        \vFunc_{t + 1}(\mRat_{t+1}, \nRat_{t+1}, \Decision) /
        \sigma_\error
        \right)
    }{ \sum\limits_{\Decision \in \{\Work, \Retire\}} \exp \left(
        \frac{\vFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1},
            \Decision)}{\sigma_\error} \right)}
\end{equation}

the first-order conditions are therefore

\begin{equation}
    \vOptAlt_{t}^{\mRat}(\mRat_{t+1}, \nRat_{t+1}) = \sum_{\Decision \in
        \{\Work, \Retire\}} \Prob_{t}(\Decision ~
    \lvert ~
    \mRat_{t+1}, \nRat_{t+1}) \vFunc_{t+1}^{\mRat}(\mRat_{t+1},
    \nRat_{t+1},
    \Decision)
\end{equation}
