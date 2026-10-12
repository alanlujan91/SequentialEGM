
(multinterp)=

# Multivariate Interpolation on Curvilinear Grids

EGM's efficiency comes from working on exogenous grids on which calculations are straightforward, then using the inverted Euler equation to recover endogenous state variables. That recovery, however, gives up the grid's rectilinear geometry, so standard multilinear interpolation, which assumes each dimension varies independently, cannot handle the curvilinear grid that results.

ENGINE relies on the index structure that endogenous grids inherit from exogenous grids despite geometric warping. It reduces multidimensional interpolation to a sequence of one-dimensional interpolations along these inherited indices, because row $k$ remains row $k$ after warping and neighbors remain neighbors, even when geometric regularity (uniform spacing, rectilinear alignment) is lost.

Consider a curvilinear grid indexed by $j \in \{1,\ldots,J\}$ and $k \in \{1,\ldots,K\}$, where each "row" (fixed $k$) contains $J$ points. The curvilinear interpolation of {cite:t}`White2015` (the curvilinear method below) locates the containing quadrilateral cell via sector-walking, then solves for normalized coordinates within that cell (a quadratic equation in 2D, Newton iteration in higher dimensions). ENGINE sidesteps both operations: for each of the $K$ rows it asks where a vertical line at $x^*$ intersects that row, a one-dimensional interpolation problem that finds where $x^*$ falls among the $J$ $x$-coordinates and interpolates the $y$-coordinate to produce an intermediate point. A final 1D interpolation across these $K$ intermediate points yields the answer. In implementation, a single query touches only the $O(\log K)$ rows its binary search probes, and the dense regrid reuses brackets across sorted targets.

## The ENGINE Algorithm

ENGINE works in index space, where bracketing is efficient, and can map back to physical coordinates because each endogenous point $(x_{jk}, y_{jk})$ keeps the indices $(j,k)$ of the exogenous post-decision grid point that generated it through the EGM inversion. This index-to-mesh correspondence is a homeomorphism under conditions developed below, a result the appendix states and proves (@thm-grid-homeomorphism). @fig:grid-homeomorphism makes the correspondence concrete. Because the mapping preserves neighborhoods, a bracket search on the rectilinear index grid is a valid search of the warped physical mesh.

```{figure} ../figures/grid_homeomorphism.*
:name: fig:grid-homeomorphism
:align: center

The grid homeomorphism between index and physical space. (a) Curvilinear physical grid. (b) Rectilinear index grid. The shaded cell has the same four nodes in both.
```

The full two-pass listing and boundary-extrapolation behavior are recorded in the [Appendix](#appendix). @fig:engine_visual traces one query through both passes.

```{figure} ../figures/engine_interpolation_visual.*
:name: fig:engine_visual
:align: center

The two passes of ENGINE. (a) Pass 1: a 1D bracket search on each $k$-slice at $x = x^*$ gives the intermediate ordinates $\hat{y}_k$ (circles). (b) Pass 2: one 1D interpolation across the $\hat{y}_k$ gives the value at the query point (star).
```

For a query point $(x^*, y^*)$, the interpolated value is

\begin{equation}
\hat{f}(x^*, y^*) = \texttt{ENGINE}_k\Bigl(y^*;\; \bigl\{\hat{y}_k\bigr\}_{k=1}^{K},\; \bigl\{\hat{f}_k\bigr\}_{k=1}^{K}\Bigr),
\end{equation}

where $\texttt{ENGINE}_k$ denotes Pass 2's one-dimensional linear interpolation across the row index $k$, and $\hat{y}_k$ and $\hat{f}_k$ are the intermediate coordinates and values from Pass 1.

On a warped mesh the two passes do not commute. Walking the regular axis first moves the brackets less between rows.

For sorted targets, each search resumes from the previous target's bracket and never moves backward, so total work is proportional to the number of targets plus their total bracket displacement.

## When ENGINE Applies

The pure consumption-saving problem illustrates the basic phenomenon. An exogenous grid of post-decision liquid assets $\aMat$ maps through the inverted Euler equation to an endogenous grid of market resources $\mMat$ with different spacing, the nonlinearity of marginal utility making the mapping non-uniform and binding constraints adding kinks. In one dimension, non-uniform linear interpolation handles the distortion. Higher-dimensional problems, however, inherit the warping in each dimension simultaneously. How much of the grid's structure survives then determines the appropriate interpolation method.

Previous approaches to this problem include Delaunay triangulation {cite:p}`Ludwig2018` and curvilinear interpolation {cite:p}`White2015`. Delaunay triangulation constructs a mesh of simplices over the scattered points and performs barycentric interpolation within each simplex. Although general and robust, triangulation is computationally expensive. Construction from unstructured point sets requires at least order $N \log N$ operations in the algebraic decision-tree model, a lower bound rather than an implementation artifact: sorting $N$ numbers reduces to triangulating $N$ points on a parabola {cite:p}`PreparataShamos1985`. The triangulation must also be recomputed when grids change during iterative solution procedures. The curvilinear method exploits the grid structure to map irregular quadrilateral sectors to the unit square for bilinear interpolation; sector location uses a visibility walk algorithm, and coordinate calculation in 2D uses a closed-form quadratic solution (3D and higher require Newton iteration).

ENGINE requires structural conditions on the endogenous grid. Analytic invertibility of a stage first-order condition (@prop-egm-utility, @prop-egm-transition) is not enough on its own, because the recovery map defined in the [Introduction](#introduction) can still fold when continuation is nonconcave or when cross effects reverse orientation. Discrete choices and nonconcave regions are the leading exclusions.

ENGINE imposes two requirements, both geometric: cells cannot fold, and each coordinate must increase with its own index. A third property, a bound on bracket displacement between adjacent rows, governs how far ENGINE must search. The implementation verifies every windowed bracket and falls back to a full row search on a miss, so the search margin affects only speed.

Nonlinear EGM mappings can distort cells severely enough to invert them. The [fold-free](#def-fold-free) condition, the classical validity condition from structured mesh generation {cite:p}`Knupp2001`, ensures each cell maintains proper orientation.

```{prf:definition} Fold-Free Cell
:label: def-fold-free

Each grid cell with corners $(x_{jk}, y_{jk})$, $(x_{j+1,k}, y_{j+1,k})$, $(x_{j,k+1}, y_{j,k+1})$, $(x_{j+1,k+1}, y_{j+1,k+1})$ is *fold-free* (also called *valid* or *orientation-preserving*) if the Jacobian determinant $\mathcal{J}(s,t)$ of its bilinear coordinate map is positive at all four corners $(s,t) \in \{0,1\}^2$. At the lower-left corner, this reduces to

$$\mathcal{J}_{jk} \equiv (x_{j+1,k} - x_{jk})(y_{j,k+1} - y_{jk}) - (x_{j,k+1} - x_{jk})(y_{j+1,k} - y_{jk}) > 0.$$

The remaining three conditions use the cell's edge vectors at the corresponding vertices. A grid is *fold-free* when every one of its cells is.
```

Four-corner positivity rules out a fold; @lem-vertical-line shows that it delivers $\mathcal{J}(s,t)>0$ throughout the cell.[^fold-free-jacobian]

[^fold-free-jacobian]: The determinant $\mathcal{J}_{jk}$ equals twice the signed area of the triangle formed by the lower-left, lower-right, and upper-left vertices. The four-corner condition is standard in structured mesh generation {cite:p}`Knupp2001`, is stronger than lower-left positivity alone, and is exactly the classical necessary and sufficient condition for invertibility of the bilinear cell map {cite:p}`Knupp1990`.

Fold-freeness ensures local cell validity, but ENGINE also requires that bracketing along rows and columns be well-defined. [IPM](#def-ipm) ensures coordinates increase monotonically along each index direction.

```{prf:definition} Index-Preserved Monotonicity (IPM)
:label: def-ipm

A grid $\{(x_{jk}, y_{jk}) : j = 1,\ldots,J, \; k = 1,\ldots,K\}$ satisfies Index-Preserved Monotonicity if:

1. **Row monotonicity:** for each fixed $k$, the first coordinate increases along the row, $x_{j+1,k} > x_{jk}$ for all $j \in \{1,\ldots,J-1\}$.
2. **Column monotonicity:** for each fixed $j$, the second coordinate increases along the column, $y_{j,k+1} > y_{jk}$ for all $k \in \{1,\ldots,K-1\}$.

Strict inequalities ensure non-degenerate grid spacing. Weak IPM (allowing equalities) accommodates constraint regions where optimal policies are constant, but requires care to avoid degenerate cells.[^degenerate-cells]
```

[^degenerate-cells]: Where a constraint holds a policy constant along a grid dimension, adjacent grid points can share coordinates. I skip zero-width intervals in the bracket search, take one-sided limits at the edges of such regions, and treat each constant-policy region as one effective grid point.

Together, [fold-free](#def-fold-free) and [IPM](#def-ipm) ensure the grid mapping is a homeomorphism, a continuous bijection with continuous inverse from index space to physical space, so the sequential brackets are well-defined within the interpolation region.[^grid-terminology] Efficiency also depends on how far a query's bracket can move between adjacent rows. The per-row maximum of that drift, the constant $\alpha$, determines the search window, and a smaller $\alpha$ reduces search work. Economically, $\alpha$ tends to grow where policies change rapidly relative to grid spacing, near occasionally binding constraints or under high value-function curvature, and when one axis's spacing is much finer than the other's. Refining one axis alone creates exactly that mismatch, so refinement should proceed at a common rate with comparable spacing across axes. The constant governs the single-query search window. By contrast, the cost of the dense regrid in the backward step depends on how far the brackets move in total across its sorted targets.

[^grid-terminology]: IPM is a new term introduced here. The standard characterization of a valid curvilinear coordinate system is that the grid mapping is a homeomorphism.

After each solve I compute the empirical estimate $\hat\alpha$, the largest bracket displacement observed between adjacent rows over a dense set of test points, as a mesh diagnostic. On the health-model meshes in [Online Appendix A](#appendix-online), $\hat\alpha$ ranges from $10$ to $16$, well above the default margin of two. Margins matched to those measurements avoid the fallback searches, and ENGINE completes every reported solve.[^alpha-implementation]

[^alpha-implementation]: A dense measurement is an empirical diagnostic, not a guarantee for untested queries. A guarantee would require a theoretical bound or an exhaustive check of a declared finite query set.

I verify directly whether an inverted stage's recovery map delivers a qualifying mesh. On the two-asset block, exhaustive four-corner and row-and-column monotonicity tests confirm both conditions on every cell of the converged mesh at every reported benchmark refinement ([Section 3](#multdim-abrs)). On the health model of [Online Appendix A](#appendix-online), every reported mesh of the two structured methods satisfies IPM, and the only folds on those meshes are sliver cells at the lowest health node. In two high-mortality calibrations of that appendix's warp sweep, under half a percent of evaluation points fall in those cells, and in the other calibrations none do; the reported residuals include every final query inside one. The measured $\hat\alpha$ stays between two and three on every reported two-asset mesh, well below the health meshes' ten to sixteen. Yet where it reaches three it exceeds the default margin of two, which the containment fallback absorbs. On the health meshes I set the margin from the measurement.

Occasionally binding constraints introduce kinks where the recovery map is not smooth. The implementations build the constrained and interior branches separately; on the two-asset block the checks above then verify the assembled mesh, kink-straddling cells included. In the health model of [Online Appendix A](#appendix-online) the kink falls on a mesh line, so no cell straddles it, and the cells next to it are fold-free in every reported mesh of the two structured methods. Discrete choices instead create jumps that no continuous interpolant should cross. Stages with discrete choices therefore belong to upper-envelope methods such as G2EGM or DCEGM.

## Accuracy and Complexity

On rectangular grids the two passes reduce algebraically to standard bilinear interpolation, whose error for twice continuously differentiable functions is second order in the mesh width {cite:p}`deBoor2001`. I measure accuracy on warped meshes directly. Under successive grid doublings the robust off-grid Euler residual falls by about a factor of five per doubling, and the converged policy's distance to a high-resolution benchmark falls about threefold per doubling (Online Appendix B).

For the dense regrid onto a $P \times Q$ target grid in each backward step, each target costs at most the single-query bound, so the regrid is $O(PQ\log J\cdot\log K)$. Sorted target axes, however, let each bracket search resume where the previous one ended, so in the scaling benchmark the cost falls far below that bound. On meshes with monotone row shifts, whose measured $\hat\alpha$ nevertheless rises with mesh size, doubling $J=K$ from $32$ to $1024$ gives $N^{0.98}$ scaling in $N=JK$, about ten nanoseconds per output value at the largest mesh.

The two-dimensional implementation needs neither a construction step nor Newton iteration.[^implementation]

[^implementation]: The implementation uses NumPy {cite:p}`Harris2020` with Numba {cite:p}`Lam2015` just-in-time compilation within the Econ-ARK HARK toolkit {cite:p}`Carroll2018`, so comparisons against the toolkit's curvilinear routine (`Curvilinear2DInterp`) are between equivalently compiled implementations. The comparison is also matched on warm starts: the curvilinear sector walk begins each query at the previously found sector, just as ENGINE resumes its brackets across sorted targets. When multiple functions share the same endogenous grid, the implementation reuses Pass 1's cell-location work across all of them.

### Choosing the Interpolation Order

Practitioners should walk the *regular* axis first, the one mapped to a strictly monotone endogenous coordinate with smaller bracket displacement. In the two-asset Stage 0 recovery domain, this is the liquid-resources axis $\lRat$; adjustment costs warp the incoming illiquid coordinate.

One limiting case makes the second pass trivial. If the second coordinate is aligned across rows ($y_{jk}=y_k$) with the shared ordinates increasing ($y_{k+1}>y_k$), and every row satisfies $x_{j+1,k}>x_{jk}$, the aligned family supplies column monotonicity while the ordered rows supply row monotonicity and positive cell orientation.[^separable-engine]

[^separable-engine]: On a target grid sharing the aligned $y_k$ axis, Pass 2 evaluates at those same ordinates, so only one-dimensional non-uniform interpolation along the ordered rows remains. Beyond index alignment, the aligned ordinates and the other coordinate family must also be strictly monotone.

On warped meshes, walking the regular axis first reduces bracket displacement and the effective constant $\alpha$. Reversal keeps IPM in the tested meshes but raises search work toward the $O(\log J\cdot\log K)$ worst case. It also changes the interpolant because sequential one-dimensional interpolation does not commute across warped coordinate lines. Re-solving the two-asset block in reverse order changes total solve time by less than ten percent at every grid, while the median relative consumption difference stays below $0.15$ percent and robust Euler medians remain within two percent (Online Appendix B). In this application, order governs the worst-case search cost rather than converged accuracy.

### Higher Dimensions

The sequential construction extends algorithmically to higher dimensions through a single one-dimensional pass per dimension. Its efficiency would require controlling bracket drift along each added dimension; the implementation and quantitative claims here are two-dimensional.

## Comparison with Alternative Interpolants

The three approaches to multidimensional EGM interpolation differ in how much of the endogenous grid's structure they exploit. On a fold-free mesh satisfying IPM, the endogenous grid is the image of the rectilinear exogenous grid under a homeomorphism (@thm-grid-homeomorphism), so the rectangular index lattice survives intact and every row and column remains sorted. Delaunay triangulation uses none of this structure; the curvilinear method uses adjacency (which cell neighbors which) but not order; ENGINE uses both:

```{latex}
\begin{table}[htbp]
\centering
\caption{Computational cost and grid requirements of multidimensional interpolants.}
\label{tab-interpolant-comparison}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & Construction & Query cost & Interpolation & Grid requirements \\
\midrule
ENGINE (this paper) & $O(1)$ & $O(\log J \cdot \log K)$ & Sequential linear & Fold-free + IPM \\
Curvilinear \citep{White2015} & $O(JK)$ & $O(J+K)$ & Bilinear & Fold-free only \\
Delaunay \citep{Ludwig2018} & $O(N \log N)$ & $O(\log N)$ & Piecewise linear & None \\
\bottomrule
\end{tabular}
\end{table}
```

Among these three methods, construction cost rises as the grid requirements fall, because a method that assumes less structure spends computation reconstructing it. ENGINE's construction is free because EGM supplies the index lattice. The curvilinear method recomputes its per-cell sector data and the triangulation rebuilds order at each backward step. In the scaling benchmark the dense matched regrid runs at output size, while unstructured triangulation carries a construction cost of at least order $N \log N$ before its first query. A lattice-aware $O(N)$ triangulation would rely on the same index structure ENGINE exploits and would still be undefined outside the mesh's convex hull.

@fig:partition contrasts the geometric work each method performs to reach the same query point.

```{figure} ../figures/partition_comparison.*
:name: fig:partition
:align: center

Cell location under three interpolation approaches. The curvilinear method searches for the containing quadrilateral (panel a) and Delaunay triangulation constructs simplices from scratch (panel b); both build a spatial partition and locate the query geometrically. ENGINE needs no spatial partition: it interpolates sequentially along indexed $k$-slices (highlighted curves) before combining intermediate points (circles) to reach the query point (star) in panel (c).
```

The solve-time ratio understates the per-query advantage because interpolation is one of several cost components of a backward step, alongside the expectation step and the EGM inversions, so even a large per-query gain translates into a bounded reduction in total solve time. The asymptotic comparison in @tab-interpolant-comparison covers only the interpolation component, where ENGINE can at best tie triangulation. Any comparison-based point location needs at least order $\log N$ comparisons to find the containing cell among $N$ candidates; Delaunay point location attains that floor, and ENGINE's $O(\log J \cdot \log K)$ sits at most a row-search factor above it. The two passes multiply their logarithms rather than add them. Locating the query between rows takes $O(\log K)$ probes, each requiring its own $O(\log J)$ bracket search within a row.

Unlike the solver comparison of Section 3, the full-solve diagnostic of Online Appendix B holds the EGM$^n$ solver fixed and swaps only its Stage-0 regrid between the two structured interpolants, scoring both by total solve time to convergence and by a robust Euler error that is valid at constraint-bound queries, since it compares consumption with the lesser of the Euler-implied and constraint-feasible levels.

On accuracy the diagnostic (@tab-solvetime-abrs) finds ENGINE and the curvilinear method interchangeable, agreeing on the robust Euler error within two percent at every grid and quantile. On speed, ENGINE completes the same solve roughly three to four and a half times faster, widening with the grid. The difference is pure interpolation-method cost, including the comparator's per-step construction, since everything but the regrid step is identical across the solves. Measured directly at the converged meshes, one realign of the regular target grid costs ENGINE $0.02$ to $0.26$ milliseconds across the three grids against the curvilinear method's $0.54$ to $4.7$, factors of $18$ to $27$. Both solves run this step on one thread, even in the sixteen-thread diagnostic, so these timings apply to it unchanged. Realigning alone accounts for more than four fifths of the per-iteration difference between the two solves. The asymptotic entry in @tab-interpolant-comparison does not capture this per-call cost. At the converged fixed point, exactly one regular Stage-0 target lies outside the endogenous mesh at each grid. Online Appendix B reports the target totals and outside shares, the protocol, and boundary-sensitivity readings. Extending both axis maxima by half changes medium-grid consumption on the common interior set by $0.05$ percent at the median query and by $4.3$ percent at the worst.

```{include} ../figures/tab_solvetime_abrs.md
```

The two-asset diagnostic includes only the two structured interpolants. [Online Appendix A](#appendix-online) adds the third alternative, Delaunay triangulation, on the health investment model where it and the curvilinear method were originally proposed and tested. I use that model because it favors those two methods, so any advantage ENGINE shows there is conservative. There EGM$^n$ with ENGINE solves $2.8$ to $4.7$ times faster than the simultaneous solver with the curvilinear interpolation of {cite:t}`White2015`, and swapping only the interpolant on that solver reproduces the bulk of that speedup. Across nine calibrations chosen to bend the endogenous meshes hardest, the triangulation's worst consumption residual lies between $0.25$ and $0.66$, while ENGINE's never exceeds $0.20$. Against the curvilinear method, however, the tails split: ENGINE's 95th-percentile consumption residual is smaller at seven of nine calibrations, while its maximum is larger at six of nine.

On fold-free meshes that satisfy IPM, ENGINE matches the curvilinear method's accuracy at lower cost; when IPM fails, ENGINE does not apply. [Section 5](#conditions) asks which economic problems produce the qualifying stages and recovery maps.
