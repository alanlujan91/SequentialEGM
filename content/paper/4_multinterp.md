
(multinterp)=

# Multivariate Interpolation on Non-Rectilinear Grids

EGM's efficiency comes from working on exogenous grids where calculations are straightforward, then using the inverted Euler equation to recover endogenous state variables. This efficiency creates an interpolation challenge: the resulting endogenous grid is warped, stretched, or even unstructured. Standard multilinear interpolation assumes a rectilinear grid where each dimension varies independently, rendering it inapplicable to EGM-generated grids. We need interpolation methods that respect the actual geometry of these grids. Some methods must recognize when topological structure persists despite geometric distortion, while others must handle cases where even topology breaks down. For curvilinear grids that preserve topological structure, Curvilinear Grid Interpolation exploits that structure for efficiency. For fully unstructured grids where regularity is lost entirely, we turn to Gaussian Process Regression.

Two distinct interpolation challenges arise in EGM applications. Curvilinear grids retain regular topological structure despite geometric warping: points that are neighbors in index space remain neighbors in physical space, even as Euclidean distances are distorted. Fully unstructured grids lose even this topological regularity, with neighborhood relationships destroyed by the nonlinear mapping. We present Curvilinear Grid Interpolation (CGI) for the former case, improving upon the interpolation approach in {cite:t}`White2015`, and a machine learning approach based on Gaussian Process Regression as in {cite:t}`Scheidegger2019` for the latter.

## Motivation for Alternative Methods

The interpolation challenge arises because first-order conditions induce nonlinear mappings from exogenous to endogenous grids. Highly nonlinear or non-monotonic Euler equations, along with binding constraints that create kinks in policy functions, can severely distort grid structure. The pure consumption-savings problem illustrates the basic phenomenon: an exogenous grid of post-decision liquid assets $\aMat$ maps through the inverted Euler equation to an endogenous grid of market resources $\mMat$ with different spacing. The nonlinearity of marginal utility ensures this mapping is non-uniform. In one dimension, non-uniform linear interpolation handles this distortion without difficulty.

Higher-dimensional problems inherit this warping in each dimension simultaneously, potentially destroying the regular structure assumed by standard multilinear interpolation. The degree of structural preservation determines the appropriate interpolation method. Curvilinear grids retain topological regularity (points that are index-neighbors remain geometric neighbors), permitting efficient specialized methods that exploit preserved structure. Fully unstructured grids, as arise in the pension deposit problem of [Section %s](#multdim), lose even topological regularity, requiring more sophisticated approaches that make no assumptions about grid geometry.

A similar approach using Delaunay triangulation was presented in {cite:t}`Ludwig2018`. However, this approach is not well suited for our purposes because triangulation can be computationally intensive and slow (often offsetting the efficiency gains from the Endogenous Grid Method). As an alternative, we introduce the use of Gaussian Process Regression (GPR) along with the Endogenous Grid Method for unstructured grids. GPR is computationally efficient, and tools exist to easily parallelize and take advantage of hardware such as Graphics Processing Units (GPU) {cite:p}`Gardner2018`.

## Interpolation on Curvilinear Grids

We begin with the case of curvilinear grids, which arise in problems like the labor-leisure example in [Section %s](#method). In this case, standard multi-linear interpolation is inapplicable because the resulting endogenous grid is non-rectilinear. Instead, we introduce Curvilinear Grid Interpolation (CGI), which exploits the preserved topological structure of curvilinear grids to achieve computational efficiency superior to triangulation-based or dense interpolation methods.

```{figure} ../../docs/figures/LaborSeparableWarpedGrid.*
:name: fig:LaborSeparableWarpedGrid
:align: center

Warped curvilinear grid that results from multivariate EGM. This grid can be interpolated by CGI.
```

```{prf:definition} Curvilinear Grid
:label: def-curvilinear

A grid of points $\{(x_{ij}, y_{ij}) : i = 1,\ldots,I, \; j = 1,\ldots,J\}$ in $\mathbb{R}^2$ is curvilinear if there exists a continuous and piecewise differentiable mapping $\phi: [0,I-1] \times [0,J-1] \to \mathbb{R}^2$ such that $\phi(i,j) = (x_{ij}, y_{ij})$ and $\phi$ is locally invertible almost everywhere.[^curvilinear-def] The grid is topologically regular if $\phi$ preserves the ordering of indices: for adjacent indices $(i,j)$ and $(i',j')$ in the index space, the corresponding points in physical space remain geometrically adjacent.
```

Consider a function $f: \mathbb{R}^2 \to \mathbb{R}$ for which we observe values $z_{ij} = f(x_{ij}, y_{ij})$ at a curvilinear grid of points $\{(x_{ij}, y_{ij})\}_{i,j}$. The points $(x_{ij}, y_{ij})$ are not evenly spaced and do not form a rectilinear grid, yet they retain their matrix structure through the continuous mapping $\phi$ from index space $(i,j)$ to physical space $(x,y)$. This topological regularity is crucial: neighborhood relationships are preserved even though Euclidean distances are distorted.

```{figure} ../../docs/figures/WarpedInterpolation.*
:name: fig:warped_interp
:align: center

True function and curvilinear grid of points for which we know the value of the function.
```

[Figure %s](#fig:warped_interp) displays the true function in three-dimensional space along with the observed grid points. Connecting points along each row and column reveals the piecewise affine structure characteristic of curvilinear grids. The key mathematical insight is that there exists a homotopy (a continuous deformation) between the curvilinear grid in physical space and the rectilinear grid in index space, as shown in [Figure %s](#fig:homotopy).

```{figure} ../../docs/figures/Homotopy.*
:name: fig:homotopy
:align: center

Homotopy between the curvilinear grid and the index coordinates of the matrix.
```

```{prf:algorithm} Curvilinear Grid Interpolation
:label: alg-cgi

To evaluate $f$ at an arbitrary point $(x^*, y^*)$ within the convex hull of the grid, we proceed in three stages. For each row $i \in \{1,\ldots,I\}$, we construct a piecewise linear interpolator $g_i: [y_{i1}, y_{iJ}] \to \mathbb{R}$ connecting the points $\{(y_{ij}, x_{ij})\}_{j=1}^J$. This allows us to find the value $\tilde{x}_i = g_i(y^*)$ such that the horizontal line at height $y = y^*$ intersects row $i$ at position $\tilde{x}_i$. Similarly, we construct interpolators $h_i$ for the function values to obtain $\tilde{z}_i = h_i(y^*)$. Having obtained $I$ pairs $\{(\tilde{x}_i, \tilde{z}_i)\}_{i=1}^I$ representing intersections of the horizontal line $y = y^*$ with each row of the curvilinear grid, we construct a piecewise linear interpolator $\hat{h}: [\tilde{x}_1, \tilde{x}_I] \to \mathbb{R}$ through these points and evaluate $\hat{f}(x^*, y^*) = \hat{h}(x^*)$. We confirm that $(x^*, y^*)$ lies within the convex hull of the grid to ensure the interpolation is well-defined.[^cgi-convexhull]
```

The choice of dimension order (rows versus columns) can affect accuracy when the mapping $\phi$ is highly anisotropic. [Figure %s](#fig:mapping) illustrates this process: the vertical line at $x = x^*$ intersects each column at specific $y$-coordinates (shown as circles), and interpolating across these intersection points yields the final estimate. The method leverages the grid's topological structure by using fast one-dimensional interpolation sequentially, avoiding the computational cost of triangulation.[^cgi-examples]

```{figure} ../../docs/figures/Mapping.*
:name: fig:mapping
:align: center

CGI intersects the horizontal line at $y = y^*$ with each row of the curvilinear grid (circles), then interpolates across these intersections to evaluate $f(x^*, y^*)$.
```

CGI exploits the preserved matrix structure of curvilinear grids to achieve $O(I + J)$ complexity per evaluation, compared to $O(IJ \log(IJ))$ for Delaunay triangulation construction or $O((IJ)^2)$ for dense grid methods.[^cgi-complexity] Moreover, CGI naturally extends to higher dimensions through recursive application, whereas triangulation-based methods suffer from the curse of dimensionality in simplex construction.

[^cgi-complexity]: The complexity analysis assumes binary search for locating the relevant grid cell in each dimension. In practice, if evaluations are performed along a continuous path (as in policy function iteration), the search can be accelerated using the previous evaluation's location as an initial guess, reducing amortized complexity.

## Interpolation on Unstructured Grids

We now turn to the more challenging case of fully unstructured grids, as arise in the pension deposit problem of [Section %s](#multdim). In this case, the endogenous grid loses even its topological regularity, making curvilinear interpolation methods inapplicable. We use Gaussian Process Regression to handle this case.

```{prf:definition} Unstructured Grid
:label: def-unstructured

A grid of points $\{(\mathbf{x}_k, z_k) : k = 1,\ldots,N\}$ in $\mathbb{R}^d \times \mathbb{R}$ is unstructured if it cannot be represented by a continuous mapping from a regular index space that preserves local neighborhoods. Formally, no continuous and piecewise differentiable mapping $\phi: [0,n_1-1] \times \cdots \times [0,n_d-1] \to \mathbb{R}^d$ exists such that the grid points correspond to $\phi$ evaluated at integer lattice points while preserving adjacency relationships.[^unstructured-def] Equivalently, the grid is unstructured when points that are neighbors in physical space may correspond to arbitrarily distant indices in any attempted regular indexing scheme, or when multiple grid points map to overlapping regions of the domain.
```

```{figure} ../../docs/figures/SparsePensionExogenousGrid.*
:name: fig:exog
:align: center

A regular, rectilinear exogenous grid of pension balances after deposit $\bRat_{t}$ and liquid assets after consumption $\lRat_{t}$.
```

Starting from a regular and rectilinear exogenous grid of liquid assets post-consumption $\lRat_{t}$ and pension balances post-deposit $\bRat_{t}$ shown in [Figure %s](#fig:exog), we obtain [Figure %s](#fig:endog) which shows an irregular and unstructured endogenous grid of market resources $\mRat_{t}$ and pension balances pre-deposit $\nRat_{t}$.

```{figure} ../../docs/figures/PensionEndogenousGrid.*
:name: fig:endog
:align: center

An irregular, unstructured endogenous grid of market resources $\mRat_{t}$ and pension balances before deposit $\nRat_{t}$.
```

On unstructured grids, multiple exogenous points may map to overlapping regions of the endogenous space.[^gpr-uniqueness] To interpolate a function defined on such grids, we use Gaussian Process Regression (GPR) as in {cite:t}`Scheidegger2019`. GPR handles this by finding the maximum likelihood function given all observed points, effectively averaging over conflicting information in a principled manner.

[^gpr-uniqueness]: Unlike Delaunay triangulation, which requires preprocessing to remove duplicate or nearly-collinear points, GPR naturally accommodates such degeneracies through its probabilistic framework. The posterior mean provides a smooth approximation even when the endogenous grid contains local irregularities or overlapping regions.

A Gaussian Process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. Formally, a GP is completely specified by its mean function $m(\mathbf{x})$ and covariance (kernel) function $k(\mathbf{x}, \mathbf{x}')$. For any finite set of points, we have

\begin{equation}
    \begin{gathered}
        \mathbf{X} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma}) \quad \text{s.t.} \quad x_i \sim \mathcal{N}(\mu_i, \sigma_{ii}) \\
        \text{and} \quad \sigma_{ij} = \Ex[(x_i - \mu_i)(x_j - \mu_j)] \quad \forall i,j \in \{1, \ldots, n\}.
    \end{gathered}
\end{equation}

where

\begin{equation}
    \mathbf{X} = \begin{bmatrix}
        x_1    \\
        x_2    \\
        \vdots \\
        x_n
    \end{bmatrix}
    \quad
    \mathbf{\mu} = \begin{bmatrix}
        \mu_1  \\
        \mu_2  \\
        \vdots \\
        \mu_n
    \end{bmatrix}
    \quad
    \mathbf{\Sigma} = \begin{bmatrix}
        \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1n} \\
        \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2n} \\
        \vdots      & \vdots      & \ddots & \vdots      \\
        \sigma_{n1} & \sigma_{n2} & \cdots & \sigma_{nn}
    \end{bmatrix}.
\end{equation}

A Gaussian Process can be used to represent a probability distribution over the space of functions in $n$ dimensions. Thus, Gaussian Process Regression (GPR) finds the posterior distribution over functions given observed data points. The posterior distribution is

\begin{equation}
    \mathbb{P}(\mathbf{f} | \mathbf{X}) = \mathcal{N}(\mathbf{f} | \mathbf{m}, \mathbf{K})
\end{equation}

where $\mathbf{f}$ is the vector of function values at the points $\mathbf{X}$, $\mathbf{m}$ is the posterior mean vector, and $\mathbf{K}$ is the covariance matrix determined by the kernel function $k(\cdot, \cdot)$ that describes the covariance between function values at different points.

A standard kernel function is the squared exponential (or radial basis function) kernel, which is defined as

\begin{equation}
    k(\mathbf{x}_i, \mathbf{x}_j) = \sigma^2_f \exp\left(-\frac{1}{2l^2} (\mathbf{x}_i - \mathbf{x}_j)' (\mathbf{x}_i -
    \mathbf{x}_j)\right)
\end{equation}

where $\sigma_f^2$ is the signal variance and $l$ is the length-scale parameter. This kernel is infinitely differentiable and assumes smooth functions. Using GPR to interpolate a function $f$, we can both predict the value of the function at a point $\mathbf{x}_*$ and quantify the uncertainty in the prediction via the posterior variance, which provides useful information about approximation accuracy.

In [Figure %s](#fig:true_function), we see the function we are trying to approximate along with a sample of data points for which we know the value of the function. In practice, the value of the function is unknown and/or expensive to compute, so we must use a limited amount of data to approximate it.

```{figure} ../../docs/figures/GPR_True_Function.*
:name: fig:true_function
:align: center

The true function that we are trying to approximate and a sample of data points.
```

A Gaussian Process is an infinite dimensional random process which can be used to represent a probability distribution over the space of functions. In [Figure %s](#fig:gpr_sample), we see a random sample of functions from the GPR posterior, which is a Gaussian Process conditioned on fitting the data. From this small sample of functions, we can see that the GP generates functions that fit the data well, and the goal of GPR is to find the one function that best fits the data given some hyperparameters by minimizing the negative log-likelihood of the data.

```{figure} ../../docs/figures/GPR_Posterior_Sample.*
:name: fig:gpr_sample
:align: center

A random sample of functions from the GPR posterior that fit the data. The goal of GPR is to find the function that best fits the data.
```

In [Figure %s](#fig:gpr), we see the result of GPR with a particular parametrization[^gpr-kernel] of the kernel function. The dotted line shows the true function, while the blue dots show the known data points. GPR provides the mean function which best fits the data, represented in the figure as an orange line. The shaded region represents a 95\% confidence interval, which is the uncertainty of the predicted function. Along with finding the best fit of the function, GPR provides the uncertainty of the prediction, which is useful information as to the accuracy of the approximation.

[^gpr-kernel]: The specific hyperparameters (signal variance $\sigma_f^2$ and length-scale $l$) are optimized by maximizing the marginal likelihood of the observed data. Implementation details and code for reproducing these figures are provided in the accompanying computational notebooks. The interpolation methods presented in this section are implemented using [`scipy`](https://www.scipy.org/) {cite:p}`Virtanen2020` for efficient numerical computations, [`numpy`](https://www.numpy.org/) {cite:p}`Harris2020` for array operations, [`numba`](https://numba.pydata.org/) {cite:p}`Lam2015` for just-in-time compilation to accelerate performance-critical loops, and [`scikit-learn`](https://scikit-learn.org/) {cite:p}`Pedregosa2011` for Gaussian Process Regression.

```{figure} ../../docs/figures/GaussianProcessRegression.*
:name: fig:gpr
:align: center

GPR finds the function that best fits the data given some hyperparameters. GPR then optimizes over the parameter space to find the function that minimizes the negative log-likelihood of the data.
```
