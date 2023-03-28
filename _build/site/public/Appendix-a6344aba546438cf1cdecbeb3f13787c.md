\input{./econtexRoot.texinput}
\documentclass[\econtexRoot/SequentialEGM]{subfiles}
\onlyinsubfile{\externaldocument{\econtexRoot/SequentialEGM}}
\usepackage{\econtexSetup,\econark,\econtexShortcuts}

\begin{document}

\addcontentsline{toc}{section}{Appendices} % label the section "Appendices"

\hypertarget{Appendices}{} % Allows link to [url-of-paper]#Appendices
\ifthenelse{\boolean{Web}}{}{% Web version has no page headers
\chead[Appendices]{Appendices} % but PDF version does
\appendixpage % Reset formatting for appendices
}

\hypertarget{Estimating-discount-factor-distributions-for-different-interest-rates}{}\par\section{Solving the illustrative G2EGM model with EGM$^n$}
\notinsubfile{\label{app:DF_R}}

\subsection{The problem for a retired household}

I designate as $\wFunc_{t}(\mRat_{t})$ the problem of a retired household at time $t$ with total resources $\mRat$. The retired household solves a simple consumption-savings problem with no income uncertainty and a certain next period pension of $\underline{\tShkEmp}$.

\begin{equation}
\begin{split}
\wFunc*{t}(\mRat*{t}) & = \max*{\cRat*{t}} \util(\cRat*{t}) +
\DiscFac \wFunc*{t+1}(\mRat*{t}) \\
& \text{s.t.} \\
\aRat*{t} & = \mRat*{t} - \cRat*{t} \\
\mRat*{t+1} & = \Rfree*{\aRat} \aRat\_{t} +
\underline{\tShkEmp}
\end{split}
\end{equation}

Notice that there is no uncertainty and the household receives a retirement
income $\underline{\tShkEmp}$ every period until death.

\subsection{The problem for a worker household}

The value function of a worker household is

\p

\begin{equation}
\VFunc*{t}(\mRat*{t}, \nRat*{t}) = \Ex*\error \max \left\{
\vFunc*{t}(\mRat*{t}, \nRat*{t}, \Work) + \sigma*{\error}
\error*{\Work} ,
\vFunc*{t}(\mRat*{t}, \nRat*{t}, \Retire) + \sigma*{\error}
\error*{\Retire} \right\}
\end{equation}

where the choice specific problem for a working household that decides to
continue working is

\begin{equation}
\begin{split}
\vFunc*{t}(\mRat*{t}, \nRat*{t}, \Work) & = \max*{\cRat*{t},
\dRat*{t}} \util(\cRat*{t}) - \kapShare + \DiscFac
\Ex*{t} \left[
\VFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1})
\right] \\
& \text{s.t.} \\
\aRat*{t} & = \mRat*{t} - \cRat*{t} - \dRat*{t} \\
\bRat*{t} & = \nRat*{t} + \dRat*{t} + \gFunc(\dRat*{t}) \\
\mRat*{t+1} & = \Rfree*{\aRat} \aRat*{t} + \tShkEmp*{t+1} \\
\nRat*{t+1} & = \Rfree*{\bRat} \bRat\_{t}
\end{split}
\end{equation}

and the choice specific problem for a working household that decides to retire
is

\begin{equation}
\vFunc*{t}(\mRat*{t}, \nRat*{t}, \Retire) =
\wFunc*{t}(\mRat*{t}+\nRat*{t})
\end{equation}

\subsection{Applying the Sequential EGM}

The first step is to define a post-decision value function. Once the household
decides their level of consumption and pension deposits, they are left with
liquid assets they are saving for the future and illiquid assets in their
pension account which they can't access again until retirement. The
post-decision value function can be defined as

\begin{equation}
\begin{split}
\vEnd*{t}(\aRat*{t}, \bRat*{t}) & = \DiscFac
\Ex*{t} \left[ \VFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1}) \right] \\
& \text{s.t.} \\
\mRat*{t+1} & = \Rfree*{\aRat} \aRat*{t} + \tShkEmp*{t+1} \\
\nRat*{t+1} & = \Rfree*{\bRat} \bRat\_{t}
\end{split}
\end{equation}

Then redefine the working agent's problem as

\begin{equation}
\begin{split}
\vFunc*{t}(\mRat*{t}, \nRat*{t}, \Work) & = \max*{\cRat*{t},
\dRat*{t}} \util(\cRat*{t}) - \kapShare + \vEnd*{t}(\aRat*{t},
\bRat*{t}) \\
\aRat*{t} & = \mRat*{t} - \cRat*{t} - \dRat*{t} \\
\bRat*{t} & = \nRat*{t} + \dRat*{t} + \gFunc(\dRat*{t}) \\
\end{split}
\end{equation}

Clearly, the structure of the problem remains the same, and this is the problem
that G2EGM solves. We've only moved some
of the stochastic mechanics out of the problem. Now, we can apply the
sequential EGM$^n$ method. Let the agent first decide $\dRat_{t}$, the deposit
amount into their retirement; we will call this the deposit problem, or outer loop. Thereafter, the
agent will have net liquid assets
of $\lRat_{t}$ and pension assets of $\bRat_{t}$.

\begin{equation}
\begin{split}
\vFunc*{t}(\mRat*{t}, \nRat*{t}, \Work) & = \max*{\dRat*{t}}
\vOpt*{t}(\lRat*{t}, \bRat*{t}) \\
& \text{s.t.} \\
\lRat*{t} & = \mRat*{t} - \dRat*{t} \\
\bRat*{t} & = \nRat*{t} + \dRat*{t} + \gFunc(\dRat\_{t})
\end{split}
\end{equation}

Now, the agent can move on to picking their consumption and savings; we can call this
the pure consumption problem or inner loop.

\begin{equation}
\begin{split}
\vOpt*{t}(\lRat*{t}, \bRat*{t}) & = \max*{\cRat*{t}}
\util(\cRat*{t}) - \kapShare + \vEnd*{t}(\aRat*{t}, \bRat*{t}) \\
& \text{s.t.} \\
\aRat*{t} & = \lRat*{t} - \cRat*{t} \\
\end{split}
\end{equation}

Because we've already made the pension decision, the amount of pension assets
does not change in this loop and it just passes through to the post-decision
value function.

\subsection{Solving the problem}

\subsubsection{Solving the Inner Consumption Saving Problem}

Let's start with the pure consumption-saving problem, which we can summarize by
substitution as

\begin{equation}
\vOpt*{t}(\lRat*{t}, \bRat*{t}) = \max*{\cRat*{t}} \util(\cRat*{t}) - \kapShare +
\vEnd*{t}(\lRat*{t} - \cRat*{t}, \bRat*{t})
\end{equation}

The first-order condition is

\begin{equation}
\util'(\cRat*{t}) = \vEnd*{t}^{\aRat}(\lRat*{t}-\cRat*{t}, \bRat*{t}) =
\vEnd*{t}^{\aRat}(\aRat*{t}, \bRat*{t})
\end{equation}

We can invert this Euler equation as in standard EGM to obtain the consumption
function.

\begin{equation}
\cEndFunc*{t}(\aRat*{t}, \bRat*{t}) =
\util'^{-1}\left(\vEnd*{t}^{\aRat}(\aRat*{t}, \bRat*{t})\right)
\end{equation}

Again as before, $\lEndFunc_{t}(\aRat_{t}, \bRat_{t}) =
    \cEndFunc_{t}(\aRat_{t}, \bRat_{t}) + \aRat_{t}$. To sum up, using an
exogenous
grid of $(\aRat_{t}, \bRat_{t})$ we obtain the trio $(\cEndFunc_{t}(\aRat_{t},
    \bRat_{t}), \lEndFunc_{t}(\aRat_{t},
    \bRat_{t}), \bRat_{t})$ which
provides an
interpolating function for our optimal consumption decision rule over the
$(\lRat, \bRat)$ grid. Without loss of generality, assume $\lEndFunc_{t} =
    \lEndFunc_{t}(\aRat_{t}, \bRat_{t})$ and define the interpolating
function as

\begin{equation}
\cTarg*{t}(\lEndFunc*{t}, \bRat*{t}) \equiv \cEndFunc*{t}(\aRat*{t},
\bRat*{t})
\end{equation}

For completeness, we derive the envelope conditions as well, and as we will
see, these will be useful when solving the next section.

\begin{equation}
\begin{split}
\vOpt*{t}^{\lRat}(\lRat*{t}, \bRat*{t}) & =
\vEnd*{t}^{\aRat}(\aRat*{t}, \bRat*{t}) = \util'(\cRat*{t}) \\
\vOpt*{t}^{\bRat}(\lRat*{t}, \bRat*{t}) & =
\vEnd*{t}^{\bRat}(\aRat*{t}, \bRat\_{t})
\end{split}
\end{equation}

\subsubsection{Solving the Outer Pension Deposit Problem}

Now, we can move on to solving the deposit problem, which we can also summarize
as

\begin{equation}
\vFunc*{t}(\mRat*{t}, \nRat*{t}, \Work) = \max*{\dRat*{t}}
\vOpt*{t}(\mRat*{t} - \dRat*{t}, \nRat*{t} + \dRat*{t} + \gFunc(\dRat\_{t}))
\end{equation}

The first-order condition is

\begin{equation}
\vOpt*{t}^{\lRat}(\lRat*{t}, \bRat*{t})(-1) +
\vOpt*{t}^{\bRat}(\lRat*{t}, \bRat*{t})(1+\gFunc'(\dRat\_{t})) = 0
\end{equation}

Rearranging this equation gives

\begin{equation}
\gFunc'(\dRat*{t}) = \frac{\vOpt*{t}^{\lRat}(\lRat*{t},
\bRat*{t})}{\vOpt*{t}^{\bRat}(\lRat*{t}, \bRat\_{t})} - 1
\end{equation}

Assuming that $\gFunc'(\dRat)$ exists and is invertible, we can find

\begin{equation}
\dEndFunc*{t}(\lRat*{t}, \bRat*{t}) = \gFunc'^{-1}\left(
\frac{\vOpt*{t}^{\lRat}(\lRat*{t},
\bRat*{t})}{\vOpt*{t}^{\bRat}(\lRat*{t},
\bRat\_{t})} - 1 \right)
\end{equation}

Using this, we can back out $\nRat_{t}$ as

\begin{equation}
\nEndFunc*{t}(\lRat*{t}, \bRat*{t}) = \bRat*{t} -
\dEndFunc*{t}(\lRat*{t}, \bRat*{t}) - \gFunc(\dEndFunc*{t}(\lRat*{t},
\bRat*{t}))
\end{equation}

and $\mRat_{t}$ as

\begin{equation}
\mEndFunc*{t}(\lRat*{t}, \bRat*{t}) = \lRat*{t} +
\dEndFunc*{t}(\lRat*{t}, \bRat\_{t})
\end{equation}

In sum, given an exogenous grid $(\lRat_{t}, \bRat_{t})$ we obtain the triple
$\left(\mEndFunc_{t}(\lRat_{t}, \bRat_{t}), \nEndFunc_{t}(\lRat_{t},
        \bRat_{t}), \dEndFunc_{t}(\lRat_{t}, \bRat_{t})\right)$, which
we can use to
create an interpolator for the decision rule $\dRat_{t}$.

To close the solution method, the envelope conditions are

\begin{equation}
\begin{split}
\vFunc*{t}^{\mRat}(\mRat*{t}, \nRat*{t}, \Work) & =
\vOpt*{t}^{\lRat}(\lRat*{t}, \bRat*{t}) \\
\vFunc*{t}^{\nRat}(\mRat*{t}, \nRat*{t}, \Work) & =
\vOpt*{t}^{\bRat}(\lRat*{t}, \bRat*{t})
\end{split}
\end{equation}

\subsection{Is g invertible?}

We've already seen that $\util'(\cdot)$ is invertible, but is $\gFunc$?

\begin{equation}
\gFunc(\dRat) = \xFer \log(1+\dRat) \qquad \gFunc'(\dRat) =
\frac{\xFer}{1+\dRat} \qquad \gFunc'^{-1}(y) = \xFer/y - 1
\end{equation}

\subsection{The Post-Decision Value and Marginal Value Functions}

\begin{equation}
\begin{split}
\vEnd*{t}(\aRat, \bRat) & = \DiscFac \Ex*{t} \left[
\VFunc(\mRat_{t+1}, \nRat_{t+1}) \right] \\
& \text{s.t.} \\
\mRat*{t+1} & = \Rfree*{\aRat} \aRat*{t} + \tShkEmp*{t+1} \\
\nRat*{t+1} & = \Rfree*{\bRat} \bRat\_{t}
\end{split}
\end{equation}

and

\begin{equation}
\begin{split}
\vEnd*{t}^{\aRat}(\aRat*{t}, \bRat*{t}) & = \DiscFac
\Rfree*{\aRat} \Ex*{t} \left[ \VFunc^{\mRat}*{t+1}(\mRat*{t+1},
\nRat*{t+1})
\right] \\
& \text{s.t.} \\
\mRat*{t+1} & = \Rfree*{\aRat} \aRat*{t} + \tShkEmp*{t+1} \\
\nRat*{t+1} & = \Rfree*{\bRat} \bRat\_{t}
\end{split}
\end{equation}

and

\begin{equation}
\begin{split}
\vEnd*{t}^{\bRat}(\aRat*{t}, \bRat*{t}) & = \DiscFac
\Rfree*{\bRat} \Ex*{t} \left[ \VFunc^{\nRat}*{t+1}(\mRat*{t+1},
\nRat*{t+1})
\right] \\
& \text{s.t.} \\
\mRat*{t+1} & = \Rfree*{\aRat} \aRat*{t} + \tShkEmp*{t+1} \\
\nRat*{t+1} & = \Rfree*{\bRat} \bRat\_{t}
\end{split}
\end{equation}

\subsection{Taste Shocks}

From discrete choice theory and from DCEGM paper, we know that

\begin{equation}
\Ex*{t} \left[
\VFunc*{t+1}(\mRat*{t+1}, \nRat*{t+1}, \error*{t+1}) \right] =
\sigma \log \left[ \sum*{\Decision \in \{\Work, \Retire\}} \exp \left(
\frac{\vFunc*{t+1}(\mRat*{t+1}, \nRat*{t+1},
\Decision)}{\sigma*\error} \right) \right]
\end{equation}

and

\begin{equation}
\Prob*{t}(\Decision ~ \lvert ~ \mRat*{t+1}, \nRat*{t+1}) = \frac{\exp
\left(
\vFunc*{t + 1}(\mRat*{t+1}, \nRat*{t+1}, \Decision) /
\sigma*\error
\right)
}{ \sum\limits*{\Decision \in \{\Work, \Retire\}} \exp \left(
\frac{\vFunc*{t+1}(\mRat*{t+1}, \nRat*{t+1},
\Decision)}{\sigma*\error} \right)}
\end{equation}

the first-order conditions are therefore

\begin{equation}
\vOptAlt*{t}^{\mRat}(\mRat*{t+1}, \nRat*{t+1}) = \sum*{\Decision \in
\{\Work, \Retire\}} \Prob*{t}(\Decision ~
\lvert ~
\mRat*{t+1}, \nRat*{t+1}) \vFunc*{t+1}^{\mRat}(\mRat*{t+1},
\nRat*{t+1},
\Decision)
\end{equation}

\onlyinsubfile{\input{\LaTeXInputs/bibliography_blend}}

\ifthenelse{\boolean{Web}}{}{
\onlyinsubfile{\captionsetup[figure]{list=no}}
\onlyinsubfile{\captionsetup[table]{list=no}}
\end{document} \endinput
}
