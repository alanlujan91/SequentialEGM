(appendix)=

# Appendix: Solving the G2EGM Model with EGM$^n$

## Problem Formulation

The retired household at time $t$ with resources $\mRat_{t}$ solves

\begin{equation}
    \begin{split}
        \wFunc_{t}(\mRat_{t}) & = \max_{\cRat_{t}} \util(\cRat_{t}) +
        \DiscFac \wFunc_{t+1}(\mRat_{t+1}) \\
        & \text{s.t.} \\
        \aRat_{t} & = \mRat_{t} - \cRat_{t} \\
        \mRat_{t+1} & = \Rfree_{\aRat} \aRat_{t} +
        \underline{\tShkEmp}.
    \end{split}
\end{equation}

The worker's value function features discrete choice:

\begin{equation}
    \VFunc_{t}(\mRat_{t}, \nRat_{t}) = \Ex_\error \max \left\{
    \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Work) + \sigma_{\error}
    \error_{\Work} ,
    \vFunc_{t}(\mRat_{t}, \nRat_{t}, \Retire) + \sigma_{\error}
    \error_{\Retire} \right\}
\end{equation}

where continuing work yields

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

and retiring gives $\vFunc_{t}(\mRat_{t}, \nRat_{t}, \Retire) = \wFunc_{t}(\mRat_{t}+\nRat_{t})$.

## Sequential Decomposition

Define the post-decision value function[^appendix-stage-notation]:

[^appendix-stage-notation]: Stage superscripts track the sequential decomposition: $v^1_t \equiv \vFunc_t(\cdot, \cdot, \Work)$ is the deposit stage conditional on working, $v^2_t$ the consumption stage, and $v^3_t$ the expectation stage.

\begin{equation}
    \begin{split}
        v^{3}_{t}(\aRat_{t}, \bRat_{t}) & = \DiscFac
        \Ex_{t} \left[ \VFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1}) \right] \\
        & \text{s.t.} \\
        \mRat_{t+1} & = \Rfree_{\aRat} \aRat_{t} + \tShkEmp_{t+1}, \quad
        \nRat_{t+1} = \Rfree_{\bRat} \bRat_{t}.
    \end{split}
\end{equation}

Decompose the work problem into deposit choice followed by consumption:

\begin{equation}
    \begin{split}
        v^{1}_{t}(\mRat_{t}, \nRat_{t}, \Work) & = \max_{\dRat_{t}}
        v^{2}_{t}(\lRat_{t}, \bRat_{t}) \\
        & \text{s.t.} \\
        \lRat_{t} & = \mRat_{t} - \dRat_{t}, \quad
        \bRat_{t} = \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t})
    \end{split}
\end{equation}

where

\begin{equation}
    \begin{split}
        v^{2}_{t}(\lRat_{t}, \bRat_{t}) & = \max_{\cRat_{t}}
        \util(\cRat_{t}) - \kapShare + v^{3}_{t}(\aRat_{t}, \bRat_{t}) \\
        & \text{s.t.} \quad
        \aRat_{t} = \lRat_{t} - \cRat_{t}.
    \end{split}
\end{equation}

## Consumption Stage Solution

The consumption first-order condition is $\util'(\cRat_{t}) = \partial v^{3}_{t}/\partial \aRat(\aRat_{t}, \bRat_{t})$, which inverts to

\begin{equation}
    \cEndFunc_{t}(\aRat_{t}, \bRat_{t}) =
    \util'^{-1}\left(\frac{\partial v^{3}_{t}}{\partial \aRat}(\aRat_{t}, \bRat_{t})\right).
\end{equation}

Endogenous net resources: $\lEndFunc_{t}(\aRat_{t}, \bRat_{t}) = \cEndFunc_{t}(\aRat_{t}, \bRat_{t}) + \aRat_{t}$. The envelope conditions are

\begin{equation}
        \frac{\partial v^{2}_{t}}{\partial \lRat}(\lRat_{t}, \bRat_{t}) =
        \util'(\cRat_{t}), \quad
        \frac{\partial v^{2}_{t}}{\partial \bRat}(\lRat_{t}, \bRat_{t}) =
        \frac{\partial v^{3}_{t}}{\partial \bRat}(\aRat_{t}, \bRat_{t}).
\end{equation}

## Deposit Stage Solution

The deposit first-order condition is

\begin{equation}
    \gFunc'(\dRat_{t}) = \dfrac{\partial v^{2}_{t} / \partial \lRat (\lRat_{t}, \bRat_{t})}{\partial v^{2}_{t} / \partial \bRat (\lRat_{t}, \bRat_{t})} - 1.
\end{equation}

For $\gFunc(\dRat) = \xFer \log(1+\dRat)$ with $\xFer > 0$ and $\dRat > -1$, strict monotonicity of $\gFunc'(\dRat) = \xFer/(1+\dRat)$ ensures invertibility:[^g-invertible]

[^g-invertible]: The inverse derivative is $\gFunc'^{-1}(y) = \xFer/y - 1$.

\begin{equation}
    \dEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \gFunc'^{-1}\left(
    \dfrac{\partial v^{2}_{t} / \partial \lRat (\lRat_{t}, \bRat_{t})}{\partial v^{2}_{t} / \partial \bRat (\lRat_{t}, \bRat_{t})} - 1 \right).
\end{equation}

Recover endogenous states:

\begin{equation}
    \begin{split}
    \nEndFunc_{t}(\lRat_{t}, \bRat_{t}) & = \bRat_{t} -
    \dEndFunc_{t}(\lRat_{t}, \bRat_{t}) - \gFunc(\dEndFunc_{t}(\lRat_{t},
        \bRat_{t})) \\
    \mEndFunc_{t}(\lRat_{t}, \bRat_{t}) & = \lRat_{t} +
    \dEndFunc_{t}(\lRat_{t}, \bRat_{t}).
    \end{split}
\end{equation}

Envelope conditions:

\begin{equation}
        \frac{\partial v^{1}_{t}}{\partial \mRat}(\mRat_{t}, \nRat_{t}, \Work) =
        \frac{\partial v^{2}_{t}}{\partial \lRat}(\lRat_{t}, \bRat_{t}), \quad
        \frac{\partial v^{1}_{t}}{\partial \nRat}(\mRat_{t}, \nRat_{t}, \Work) =
        \frac{\partial v^{2}_{t}}{\partial \bRat}(\lRat_{t}, \bRat_{t}).
\end{equation}

## Discrete Choice Integration

Post-decision marginal values are

\begin{equation}
    \begin{split}
        \frac{\partial v^{3}_{t}}{\partial \aRat}(\aRat_{t}, \bRat_{t}) & = \DiscFac
        \Rfree_{\aRat} \Ex_{t} \left[ \frac{\partial \VFunc_{t+1}}{\partial \mRat}(\Rfree_{\aRat} \aRat_{t} + \tShkEmp_{t+1},
            \Rfree_{\bRat} \bRat_{t})
            \right] \\
        \frac{\partial v^{3}_{t}}{\partial \bRat}(\aRat_{t}, \bRat_{t}) & = \DiscFac
        \Rfree_{\bRat} \Ex_{t} \left[ \frac{\partial \VFunc_{t+1}}{\partial \nRat}(\Rfree_{\aRat} \aRat_{t} + \tShkEmp_{t+1},
            \Rfree_{\bRat} \bRat_{t})
            \right].
    \end{split}
\end{equation}

From DCEGM ({cite:t}`Iskhakov2017`):

\begin{equation}
    \Ex_{t} \left[
        \VFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1}, \error_{t+1}) \right] =
    \sigma_{\error} \log \left[ \sum_{\Decision \in \{\Work, \Retire\}} \exp \left(
        \dfrac{\vFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1},
            \Decision)}{\sigma_{\error}} \right)  \right]
\end{equation}

with choice probabilities

\begin{equation}
    \Prob_{t}(\Decision ~ \lvert ~ \mRat_{t+1}, \nRat_{t+1}) = \frac{\exp
        \left(
        \vFunc_{t + 1}(\mRat_{t+1}, \nRat_{t+1}, \Decision) /
        \sigma_{\error}
        \right)
    }{ \sum\limits_{\Decision \in \{\Work, \Retire\}} \exp \left(
        \dfrac{\vFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1},
            \Decision)}{\sigma_{\error}} \right)}.
\end{equation}

The marginal value is

\begin{equation}
    \frac{\partial \tilde{\VFunc}_{t}}{\partial \mRat}(\mRat_{t+1}, \nRat_{t+1}) = \sum_{\Decision \in
        \{\Work, \Retire\}} \Prob_{t}(\Decision ~
    \lvert ~
    \mRat_{t+1}, \nRat_{t+1}) \frac{\partial \vFunc_{t+1}}{\partial \mRat}(\mRat_{t+1},
    \nRat_{t+1},
    \Decision).
\end{equation}
