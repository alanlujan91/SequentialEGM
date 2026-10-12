---
title: "Appendix"
---

(appendix)=

# Appendix

(proof-grid-homeomorphism)=
## The Grid Homeomorphism Theorem

```{prf:lemma} Vertical-Line Monotonicity
:label: lem-vertical-line

Let a fold-free curvilinear grid satisfy IPM. Extend its index map $\Xi$, defined by $\Xi(j,k) = (x_{jk}, y_{jk})$, to $\tilde\Xi$ by bilinear interpolation within each cell, parameterizing each cell by coordinates $(s,t) \in [0,1]^2$ with $s$ along the row index and $t$ along the column index. Then within every cell $\partial x/\partial s > 0$, $\partial y/\partial t > 0$, and $\mathcal{J}\equiv\det D\tilde\Xi > 0$. Call the interval from the largest first-node abscissa to the smallest last-node abscissa of a set of rows their common span. For an abscissa $x_0$ in the common span of the rows bounding a run of bands, the preimage of $x=x_0$ crosses those bands in a single continuous curve along which the $y$-value increases strictly. Symmetrically, the $x$-value along the preimage of an ordinate in the common span of a run of columns, defined from the columns' first and last ordinates, increases strictly across those column bands.
```

```{prf:proof}
:label: prf-vertical-line

Within each cell $[j, j+1] \times [k, k+1]$, the bilinear map in parametric coordinates $(s,t) \in [0,1]^2$ takes the form $x(s,t) = (1-s)(1-t)x_{jk} + s(1-t)x_{j+1,k} + (1-s)t\,x_{j,k+1} + st\,x_{j+1,k+1}$, and similarly for $y$. The partial derivative $\partial x/\partial s$ is a convex combination of the row increments $x_{j+1,k} - x_{jk}$ and $x_{j+1,k+1} - x_{j,k+1}$, both strictly positive by IPM row monotonicity, so $\partial x/\partial s > 0$ throughout the cell; the column argument gives $\partial y/\partial t > 0$. The determinant $\mathcal{J}$ is an affine function of $(s,t)$, so its minimum over the cell occurs at a corner; fold-freeness therefore gives $\mathcal{J}>0$ throughout.

Fix a band between rows $k$ and $k+1$ and write $t \in [0,1]$ for the position between them. At each fixed $t$ the section $s \mapsto x(s,t)$ is continuous and strictly increasing, since $\partial x/\partial s > 0$ throughout the band, and its endpoint values are the linear interpolations in $t$ of the two rows' first and last abscissas. An $x_0$ in the two rows' common span therefore lies between the section's endpoints at every $t$, so the intermediate value theorem yields a crossing $x(s(t), t) = x_0$, unique by strict monotonicity. Within each cell the crossing function $s(\cdot)$ is differentiable with $ds/dt = -(\partial x/\partial t)/(\partial x/\partial s)$ by the implicit function theorem, so the $y$-value along the curve changes at rate $dy/dt = \mathcal{J}(s,t)/(\partial x/\partial s) > 0$. Because adjacent bilinear pieces agree on shared edges, $s(\cdot)$ is continuous across cell columns, and the crossings form one continuous curve over the band along which $y$ increases strictly. The curve ends on the shared row, which is the next band's own lower edge, so when $x_0$ also lies in the next pair's common span the construction continues and the concatenated curve's $y$-value increases strictly across bands. The symmetric argument, with rate $\mathcal{J}(s,t)/(\partial y/\partial t) > 0$ along the horizontal level curve, gives the column statement.
```

```{prf:theorem} Grid Homeomorphism
:label: thm-grid-homeomorphism

A fold-free curvilinear grid satisfying IPM is homeomorphic to the rectangular index grid. That is, the mapping $\Xi: \{1,\ldots,J\} \times \{1,\ldots,K\} \to \mathbb{R}^2$ defined by $\Xi(j,k) = (x_{jk}, y_{jk})$ is injective, and extending $\Xi$ via bilinear interpolation within each cell yields a homeomorphism from $[1,J] \times [1,K]$ to the grid's image.
```

```{prf:proof}
:label: prf-grid-homeomorphism

The bilinear extension $\tilde{\Xi}: [1,J] \times [1,K] \to \mathbb{R}^2$ is continuous and piecewise $C^1$. Within each cell, @lem-vertical-line gives
$$
\frac{\partial x}{\partial s}>0,\qquad
\frac{\partial y}{\partial t}>0,\qquad
\det D\tilde\Xi>0.
$$
Thus every cell-interior Jacobian has positive diagonal entries and a positive determinant. Four positive corner Jacobians are already the classical necessary and sufficient condition for invertibility of a single bilinear cell map {cite:p}`Knupp1990`; the remaining question is global: can two distant cells map to the same physical point? At a point interior to a cell the map is differentiable and its Jacobian carries that sign pattern. At every other point of the closed rectangle, on interior seams and grid vertices and the outer boundary alike, the nonsmooth form of the theorem works with the convex combinations of the one-sided Jacobians the incident cells contribute, and the sign pattern survives the mixing. Both diagonal entries do because each is a convex combination of positive edge increments. The determinant does for a structural reason. Across a seam the two incident Jacobians share the column tangent to that seam, since the two bilinear pieces agree along it, so the determinant of any mixture is the same mixture of the two positive one-sided determinants. At a grid vertex, any mixture of the four incident Jacobians has one column that is a convex combination of the two horizontal edge vectors meeting there and one that is a convex combination of the two vertical edge vectors, so its determinant expands into a positively weighted sum of the four incident corner determinants, each positive by fold-freeness. The theorem of {cite:t}`Gale1965`, in the nonsmooth form of {cite:t}`Gowda2000`, therefore applies on the compact rectangle and makes $\tilde\Xi$ globally one-to-one. Finally, a continuous injection from a compact space into the plane is a homeomorphism onto its image.
```

The direct four-corner calculation verifies fold-freeness on every cell of each finite mesh reported in Section 3, kink-straddling cells included.

(appendix-engine-algorithm)=
## The ENGINE Algorithm

Section 4 develops ENGINE and states its probe costs; this section records the single-query listing and the boundary behavior. The dense regrids in the benchmarks run the same two passes with brackets carried forward across sorted targets, as Section 4 describes.

**Algorithm (ENGINE Two-Pass Interpolation).** *Inputs:* grid $\{(x_{jk}, y_{jk}, f_{jk})\}$ for $j = 1, \ldots, J$ and $k = 1, \ldots, K$; query point $(x^*, y^*)$. *Output:* interpolated value $f(x^*, y^*)$. The listing uses subscripts $l$, $m$, and $h$ for the low, mid, and high indices. The routine $\mathrm{BracketSearch}$ verifies that the bracket it returns contains $x^*$ and falls back to a full row search on a miss.

*Pass 1: binary search on rows.*

1. Set $k_l \leftarrow 1$ and $k_h \leftarrow K$.
2. Locate the initial bracket at the bottom row, $j_l \leftarrow \mathrm{BinarySearch}(x^*, \{x_{j,k_l}\}_j)$, and compute the intermediate coordinate $\hat{y}_l \leftarrow \mathrm{Interp1D}(x^*;\, x_{j_l,k_l}, x_{j_l+1,k_l}, y_{j_l,k_l}, y_{j_l+1,k_l})$.
3. While $k_h - k_l > 1$:
   1. Set $k_m \leftarrow \lfloor (k_l + k_h)/2 \rfloor$.
   2. Find $j_m \leftarrow \mathrm{BracketSearch}(x^*, \{x_{j,k_m}\}_j, j_l)$, searching near $j_l$ within the configured bracket-displacement window.
   3. Compute $\hat{y}_m \leftarrow \mathrm{Interp1D}(x^*;\, x_{j_m,k_m}, x_{j_m+1,k_m}, y_{j_m,k_m}, y_{j_m+1,k_m})$.
   4. If the intermediate ordinate satisfies $\hat{y}_m \leq y^*$, set $k_l \leftarrow k_m$, $j_l \leftarrow j_m$, and $\hat{y}_l \leftarrow \hat{y}_m$; otherwise set $k_h \leftarrow k_m$.

*Pass 2: final interpolation.*

1. Find $j_h \leftarrow \mathrm{BracketSearch}(x^*, \{x_{j,k_h}\}_j, j_l)$.
2. Compute $\hat{y}_h \leftarrow \mathrm{Interp1D}(x^*;\, x_{j_h,k_h}, x_{j_h+1,k_h}, y_{j_h,k_h}, y_{j_h+1,k_h})$.
3. Compute $\hat{f}_l \leftarrow \mathrm{Interp1D}(x^*;\, x_{j_l,k_l}, x_{j_l+1,k_l}, f_{j_l,k_l}, f_{j_l+1,k_l})$.
4. Compute $\hat{f}_h \leftarrow \mathrm{Interp1D}(x^*;\, x_{j_h,k_h}, x_{j_h+1,k_h}, f_{j_h,k_h}, f_{j_h+1,k_h})$.
5. Set $t \leftarrow (y^* - \hat{y}_l)/(\hat{y}_h - \hat{y}_l)$ and return $(1-t)\,\hat{f}_l + t\,\hat{f}_h$.

EGM applications require an extrapolation rule: simulation occasionally produces state values slightly outside the solution grid because of shock realizations or numerical precision. Each 1D interpolation step handles out-of-bounds queries by extending the boundary slope, a continuous extension beyond the grid's convex hull: for $x^* < x_1$, the value is $f_1 + (x^* - x_1)(f_2 - f_1)/(x_2 - x_1)$, and symmetrically for $x^* > x_J$.

Linear extrapolation is unreliable near binding constraints, where policies have kinks and high curvature, and accurate in wealth-rich regions, where precautionary motives fade and policies become nearly linear. The asset grids in this paper extend well past their constraint regions, which confines any extrapolation to that smooth region.

```{raw} latex
% The \clearpage is load-bearing: a deferred float after the final text page
% stales the CAS class's lastpage ("Page N of N-1" at any pass count). It sits
% before the heading so the heading and the [!h] table below share a page.
% Verify the footer against pdfinfo whenever appendix length changes.
\clearpage
```

## Notation Reference

```{latex}
% The \providecommand lines cover symbols that appear only in this raw table
% (and in the separately built online appendix): MyST emits a macro's
% \newcommand only when it sees the macro in math it parses, and raw blocks
% are invisible to that scan.
\providecommand{\HRat}{H}
\providecommand{\hRat}{h}
\providecommand{\nRat}{n}
\providecommand{\nEndFunc}{\mathfrak{n}}
\begin{table}[!h]
\centering
\caption{Notation reference.}
\label{tab:notation}
\footnotesize
\begin{tabular}{@{}ll@{}}
\toprule
Symbol & Meaning \\
\midrule
\multicolumn{2}{@{}l}{\itshape Conventions} \\
$\mRat'$, $\aRat'$ & Prime: next-period quantity \\
$\cEndFunc$, $\mEndFunc$, $\aEndFunc$ & Gothic: endogenous grid, built by inverting a condition \\
$\aMat$, $\mMat$ & Brackets: exogenous grid \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Section 2, consumption-labor} \\
$\CRRA$, $\DiscFac$, $\Rfree$ & Risk aversion; discount factor; gross risk-free return \\
$\mRat_t$, $\aRat_t$ & Market resources; liquid assets carried in \\
$\leisure_t$, $\labor_t$ & Leisure; labor supply ($\labor = 1 - \leisure$) \\
$\hFunc$, $\labShare$, $\leiShare$ & Leisure utility; its scale and curvature \\
$\zEndFunc_t$ & Endogenous leisure from the first-order-condition inversion \\
$\tShkEmp_t$ & Wage shock \\
$v^{0}$, $v^{1}$, $v^{2}$ & Stage value functions; the superscript labels the stage \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Section 3, two-asset household} \\
$\aRat$, $\aRat'$ & Illiquid balance; illiquid choice (liquid in Section 2) \\
$\bRat$, $\bRat'$ & Liquid balance; liquid choice \\
$\Risky$, $\Rliq$ & Gross illiquid and liquid returns \\
$\mRat$, $\lRat$ & Market resources; liquid resources after the illiquid choice \\
$\lEndFunc$ & Endogenous liquid-resources grid ($\bRat' + \cEndFunc$) \\
$e$, $\Pi(e' \mid e)$ & Productivity state; Markov transition \\
$\Psi$, $\chi_0$, $\chi_1$ & Illiquid adjustment cost ($\Psi_1$, $\Psi_{12}$: argument partials); its offset and scale \\
$u$, $\tau$ & Standardized adjustment wedge; wedge target, the adjustment-cost shadow price \\
$v^{1}_{\lRat}$, $v^{1}_{\aRat'}$ & Stage-1 partials (subscripts; stage label in the superscript); $\wFunc$ replaces $v^{2}$ here \\
$\wFunc$, $\wFunc^{\bRat'}$ & End-of-period value; its superscripts denote partials (no stage label) \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Section 4, ENGINE} \\
$j$, $k$, $J$, $K$ & Exogenous-grid column and row indices; their sizes \\
$P$, $Q$, $N$ & Target-grid sizes of the dense regrid; total mesh size $N = JK$ \\
$\alpha$, $\hat{\alpha}$ & Bracket displacement between adjacent rows; its measured value \\
\addlinespace
\multicolumn{2}{@{}l}{\itshape Section 5 and Online Appendix A, generic stage} \\
$\xRat$, $\yRat$, $\sRat$, $\aRat$ & Pre-decision, post-decision, exogenous state; control \\
$\VFunc$, $\UFunc$, $\TFunc$, $\WFunc$, $\GFunc$ & Stage value; utility; within-stage transition; continuation value; next-period state map \\
$\Gamma$ & Feasible-choice correspondence \\
$\gFunc$, $\kappa$ & Control-only and state-coupled transition terms (Invertible Transition Marginals proposition) \\
$\phi$ & Marginal utility of the control, $\phi(\aRat) \equiv \uFunc'(\aRat)$ \\
$k$, $f_1$, $f_2$ & Constant transition slope; invertible transition components (the two propositions) \\
$\HRat$, $\hRat$ & Second state, post- and pre-transition; health in the testbed \\
$\nRat$, $\nEndFunc$ & Health investment (Online Appendix A; letter reuse with Section 2 labor $\labor$) \\
\bottomrule
\end{tabular}
\end{table}
```
