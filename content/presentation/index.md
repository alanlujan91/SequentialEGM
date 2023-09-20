---
title: "EGM<sup>n</sup>"
subtitle: "The Sequential Endogenous Grid Method"
author: "Alan E. Lujan Solis"
institute: "Johns Hopkins University <br> Econ-ARK"
date: "September 29, 2023"
date-format: long
bibliography: https://paperpile.com/eb/jAmePXcJLW/paperpile.bib
format:
  revealjs:
    fontsize: "25pt"
    theme: serif
    logo: econ-ark-logo.png
    footer: "Powered by **Econ-ARK**"
---

# Functional Approximation <br> in Economics

$$
\newcommand{\DiscFac}{\beta}
\newcommand{\utilFunc}{\mathrm{u}}
\newcommand{\VFunc}{\mathrm{V}}
\newcommand{\Leisure}{Z}
\newcommand{\tShk}{\xi}
\newcommand{\util}{u}
\newcommand{\tShkEmp}{\theta}
\newcommand{\BLev}{B}
\newcommand{\CLev}{C}
\newcommand{\MLev}{M}
\newcommand{\ALev}{A}
\newcommand{\Ex}{\mathbb{E}}
\newcommand{\CRRA}{\rho}
\newcommand{\labShare}{\nu}
\newcommand{\leiShare}{\zeta}
\newcommand{\h}{h}
\newcommand{\bRat}{b}
\newcommand{\leisure}{z}
\newcommand{\cRat}{c}
\newcommand{\PLev}{P}
\newcommand{\vFunc}{\mathrm{v}}
\newcommand{\Rfree}{\mathsf{R}}
\newcommand{\wage}{\mathsf{w}}
\newcommand{\riskyshare}{\varsigma}
\newcommand{\PGro}{\Gamma}
\newcommand{\labor}{\ell}
\newcommand{\aRat}{a}
\newcommand{\mRat}{m}
\newcommand{\Rport}{\mathbb{R}}
\newcommand{\Risky}{\mathbf{R}}
\newcommand{\risky}{\mathbf{r}}
\newcommand{\vOpt}{\tilde{\mathfrak{v}}}
\newcommand{\vEnd}{\mathfrak{v}}
\newcommand{\vE}{{v}^{e}}
\newcommand{\vOptAlt}{\grave{\tilde{\mathfrak{v}}}}
\newcommand{\q}{\koppa}
\newcommand{\cEndFunc}{\mathfrak{c}}
\newcommand{\cE}{\cRat^{e}}
\newcommand{\xRat}{x}
\newcommand{\aMat}{[\mathrm{a}]}
\newcommand{\mEndFunc}{\mathfrak{m}}
\newcommand{\mE}{\mRat^{e}}
\newcommand{\mMat}{[\mathrm{m}]}
\newcommand{\tShkMat}{[\mathrm{\tShkEmp}]}
\newcommand{\zEndFunc}{\mathfrak{z}}
\newcommand{\lEndFunc}{\mathfrak{l}}
\newcommand{\bEndFunc}{\mathfrak{b}}
\newcommand{\bE}{\bRat^{e}}
\newcommand{\nRat}{n}
\newcommand{\dRat}{d}
\newcommand{\gFunc}{\mathrm{g}}
\newcommand{\xFer}{\chi}
\newcommand{\lRat}{l}
\newcommand{\wFunc}{\mathrm{w}}
\newcommand{\dEndFunc}{\mathfrak{d}}
\newcommand{\nEndFunc}{\mathfrak{n}}
\newcommand{\uFunc}{\mathrm{u}}
\newcommand{\TFunc}{\mathrm{T}}
\newcommand{\UFunc}{\mathrm{U}}
\newcommand{\WFunc}{\mathrm{W}}
\newcommand{\yRat}{y}
\newcommand{\XLev}{X}
\newcommand{\Retire}{\mathbb{R}}
\newcommand{\Work}{\mathbb{W}}
\newcommand{\error}{\epsilon}
\newcommand{\err}{z}
\newcommand{\kapShare}{\alpha}
\newcommand{\kap}{k}
\newcommand{\cTarg}{\check{c}}
\newcommand{\Decision}{\mathbb{D}}
\newcommand{\Prob}{\mathbb{P}}
$$

---

## Linear Interpolation on a Uniform Grid

{{< video videos/LinearInterpolation.mp4 >}}

---

## Linear Interpolation on a Non-linear Grid

{{< video videos/LinearTransform.mp4 >}}

---

## Bilinear Interpolation

{{< video videos/MeshGridTransform.mp4 >}}

---

## Curvilinear (Warped) Grid Interpolation {.smaller}

{{< video videos/CurvilinearTransform.mp4 >}}

See: @White2015

---

## What about Unstructured Grids? {.smaller}

{{< video videos/UnstructuredGrid.mp4 >}}

See: @Ludwig2018

---

# Machine Learning <br> in Economics

---

## Artificial Neural Networks {auto-animate=true}

![Figure 1: ANN (Source: scikit-learn.org)](multilayerperceptron_network.png)

## Artificial Neural Networks {auto-animate=true}

:::: {.columns}

::: {.column width="40%"}
![Figure 1: ANN (Source: scikit-learn.org)](multilayerperceptron_network.png)
:::

::: {.column width="60%"}
- Based on biological neural networks (neurons in a brain)
- Learns function $f(X): R^n \rightarrow R^m$
- Consists of
  - input (features) $X$
  - hidden layers $g(\cdots)$
  - output (target) $y = f(X)$
- Hidden layers can have many nodes
- Neural nets can have many hidden layers (deep learning)
:::

::::

---

## A single neuron, and a bit of math {auto-animate=true}

![Figure 2: Perceptron](perceptron.png){width=50%}

$$\begin{equation}
y = g(w_0 + \sum_{i=1}^n w_i x_i) = g(w_0 + \mathbf{x}' \mathbf{w})
\end{equation}$$

---

## A single neuron, and a bit of math {auto-animate=true}

:::: {.columns}

::: {.column width="30%"}

![Figure 2: Perceptron](perceptron.png)

:::

::: {.column width="70%"}

$$\begin{equation}
y = g(w_0 + \sum_{i=1}^n w_i x_i) = g(w_0 + \mathbf{x}' \mathbf{w})
\end{equation}$$

:::

::::

. . .

:::: {.columns}

::: {.column width="60%"}

- $y$ is the output or target
- $x_i$ are the inputs or features
- $w_0$ is the bias
- $w_i$ are the weights

:::

::: {.column width="40%"}

- $g(\cdot)$ is the activation function (non-linear)
$$\begin{equation}
g(z) = \frac{1}{1 + e^{-z}}
\end{equation}$$
- usually a sigmoid, but there are many others

:::

::::

---

## Training a Neural Network

Mean Squared Error (MSE)

$$\begin{equation}
J(\mathbf{w}) = \frac{1}{2n} \sum_{i=1}^n (y_i - f(\mathbf{x}^{(i)}
; \mathbf{w}))^2
\end{equation}$$

. . .

Objective

$$\begin{equation}
\mathbf{w}^* = \arg \min_{\mathbf{w}} J(\mathbf{w})
\end{equation}$$

. . .

Gradient Descent

$$\begin{equation}
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla J(\mathbf{w}^{(t)})
\end{equation}$$

---

## Stochastic Gradient Descent

$$\begin{equation}
\widetilde{\nabla} J(\mathbf{w}^{(t)}) = \frac{1}{B} \sum_{i=1}^B \nabla_i J(\mathbf{w}^{(t)})
\end{equation}$$

$$\begin{equation}
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \widetilde{\nabla} J(\mathbf{w}^{(t)})
\end{equation}$$

---

# Dynamic Programming

[Preview](https://playground.tensorflow.org/){preview-link="true"}

---

### The Endogenous Grid Method by @Carroll2006-ag

- Simple
  - Inverted Euler equation
- Fast
  - No root-finding or optimization required
- Efficient
  - Finds exact solution at each gridpoint

---

### Limitations of EGM

  - **One-dimensional** problems/subproblems (nested)
    - (GEGM) @Barillas2007
    - (NEGM) @Druedahl2021
  - Can result in **non-rectangular grids**
    - (Curvilinear) @White2015
    - (Triangular) @Ludwig2018
  - **Non-convexities** can be problematic
    - (DCEGM) @Iskhakov2017-af
    - (G2EGM) @Druedahl2017-ac

---

###  EGM<sup>n</sup>

- **Insight**: Problems in which agent makes several **simultaneous choices** can be decomposed into **sequence of problems**
- **Problem**: Rectilinear exogenous grid results in **unstructured** endogenous grid
- **Contribution**: Using machine learning (GPR) to **interpolate** on unstructured grids

---

###  The Sequential Endogenous Grid Method

- **Simple, Fast, Efficient**
  - Inherits properties of EGM
- **Multi-dimensional**
  - Can be used for problems with multiple state variables and controls
- **Cutting-edge**
  - Functional approximation and uncertainty quantification approach using a Gaussian Process Regression

---

### Consumption - Labor - Portfolio Choice

Agent maximizes PDV of lifetime utility as in @Bodie1992

$$\begin{equation}
\VFunc_0(\BLev_0, \tShkEmp_0) = \max_{\CLev_{t}, \Leisure_{t}, \riskyshare_{t}} \Ex_{t} \left[ \sum_{t = 0}^{T} \DiscFac^{t} \utilFunc(\CLev_{t}, \Leisure_{t})  \right]
\end{equation}$$

where

$$\begin{equation}
\begin{split}
    \MLev_{t} & = \BLev_{t} + \PGro_{t} \tShkEmp_{t} (1 - \Leisure_{t}) \\
    \Rport_{t+1} & = \Rfree + (\Risky_{t+1} - \Rfree)
    \riskyshare_{t} \\
    \BLev_{t+1} & = (\MLev_{t} - \CLev_{t}) \Rport_{t+1}
  \end{split}
\end{equation}$$

---

Recursive Bellman equation in normalized form:

$$\begin{equation}
\begin{split}
    \vFunc_{t}(\bRat_{t}, \tShkEmp_{t}) & = \max_{\{\cRat_{t},
      \leisure_{t}, \riskyshare_{t}\}} \utilFunc(\cRat_{t}, \leisure_{t}) +
    \DiscFac \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA}
      \vFunc_{t+1} (\bRat_{t+1},
      \tShkEmp_{t+1}) \right] \\
    \labor_{t} & = 1 - \leisure_{t} \\
    \mRat_{t} & = \bRat_{t} + \tShkEmp_{t} \wage \labor_{t} \\
    \aRat_{t} & = \mRat_{t} - \cRat_{t} \\
    \Rport_{t+1} & = \Rfree + (\Risky_{t+1} - \Rfree)
    \riskyshare_{t} \\
    \bRat_{t+1} & = \aRat_{t} \Rport_{t+1} / \PGro_{t+1}
  \end{split}
\end{equation}$$

where

$$\begin{equation}
  \utilFunc(\CLev, \Leisure) = \util(\CLev) + \h(\Leisure) = \frac{C^{1-\CRRA}}{1-\CRRA} + \labShare^{1-\CRRA} \frac{\Leisure^{1-\leiShare}}{1-\leiShare}
\end{equation}$$

---

### Breaking up the problem into sequences

Starting from the beginning of the period, we can define the labor-leisure problem as

$$\begin{equation}
\begin{split}
    \vFunc_{t}(\bRat_{t}, \tShkEmp_{t}) & = \max_{ \leisure_{t}}
    \h(\leisure_{t}) + \vOpt_{t} (\mRat_{t}) \\
    & \text{s.t.} \\
    0 & \leq \leisure_{t} \leq 1 \\
    \labor_{t} & = 1 - \leisure_{t} \\
    \mRat_{t} & = \bRat_{t} + \tShkEmp_{t} \wage \labor_{t}.
  \end{split}
\end{equation}$$

---

### Breaking up the problem into sequences

The pure consumption-saving problem is then

$$\begin{equation}
\begin{split}
    \vOpt_{t}(\mRat_{t}) & = \max_{\cRat_{t}} \util(\cRat_{t}) + \DiscFac\vEnd_{t}(\aRat_{t}) \\
    & \text{s.t.} \\
    0 & \leq \cRat_{t} \leq \mRat_{t} \\
    \aRat_{t} & = \mRat_{t} - \cRat_{t}.
  \end{split}
\end{equation}$$

---

### Breaking up the problem into sequences

Finally, the risky portfolio problem is

$$\begin{equation}
\begin{split}
    \vEnd_{t}(\aRat_{t}) & = \max_{\riskyshare_{t}}
    \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA}
      \vFunc_{t+1}(\bRat_{t+1},
      \tShkEmp_{t+1}) \right] \\
    & \text{s.t.} \\
    0 & \leq \riskyshare_{t} \leq 1 \\
    \Rport_{t+1} & = \Rfree + (\Risky_{t+1} - \Rfree)
    \riskyshare_{t} \\
    \bRat_{t+1} & = \aRat_{t} \Rport_{t+1} / \PGro_{t+1}.
  \end{split}
\end{equation}$$

---

### Solving Consumption-Saving via EGM

We can condense the consumption-saving problem into a single equation:

$$\begin{equation}
\vOpt_{t}(\mRat_{t}) = \max_{\cRat_{t}} \util(\cRat_{t}) +
  \DiscFac \vEnd_{t}(\mRat_{t}-\cRat_{t})
\end{equation}$$

Interior solution must satisfy the Euler equation:

$$\begin{equation}
\utilFunc'(\cRat_t) = \DiscFac \vEnd_{t}'(\mRat_{t} - \cRat_{t}) = \DiscFac
  \vEnd_{t}'(\aRat_{t})
\end{equation}$$

EGM consists of inverting the Euler equation to find the consumption function:

$$\begin{equation}
\cEndFunc_{t}(\aMat) = \utilFunc'^{-1}\left( \DiscFac \vEnd_{t}'(\aMat)
  \right)
\end{equation}$$

---

Then using budget contraint we obtain endogenous grid:

$$\begin{equation}
  \mEndFunc_{t}(\aMat) = \cEndFunc_{t}(\aMat) + \aMat.
\end{equation}$$

. . .

Using points $[\mEndFunc_t]$ and $[\cEndFunc_t]$ we can build a linear interpolator $\cRat_t(\mRat)$. The constraint is handled by exogenous grid $\aMat \ge \underline{\aRat}$ and we can add an anchor point $\cRat_t(\mRat = 0) = 0$ for the linear interpolator to complete our solution.

---

### Solving Labor-Leisure (EGM, Again)

We can condense the labor-leisure problem into a single equation:

$$\begin{equation}
\vFunc_{t}(\bRat_{t}, \tShkEmp_{t}) = \max_{ \leisure_{t}}
  \h(\leisure_{t}) + \vOpt_{t}(\bRat_{t} +
  \tShkEmp_{t} \wage (1-\leisure_{t}))
\end{equation}$$

. . .

Interior solution must satisfy the first-order condition:

$$\begin{equation}
\h'(\leisure_{t}) = \vOpt_{t}'(\mRat_{t}) \wage \tShkEmp_{t}
\end{equation}$$

. . .

EGM consists of inverting the first-order condition to find leisure function:

$$\begin{equation}
\zEndFunc_{t}(\mMat, \tShkMat) = \h'^{-1}\left(
  \vOpt_{t}'(\mMat) \wage \tShkMat \right)
\end{equation}$$

---

Using market resources condition we obtain endogenous grid:

$$\bEndFunc_{t}(\mMat, \tShkMat) = \mMat -
  \tShkMat\wage(1-\zEndFunc_{t}(\mMat, \tShkMat))$$

. . .

So we construct $\leisure_t([\bEndFunc_t], \tShkMat)$. Actual leisure function is bounded between 0 and 1:

$$\begin{equation}
\hat{\leisure}_{t}(\bRat, \tShkEmp) = \max \left[ \min \left[ \leisure_{t}(\bRat, \tShkEmp), 1 \right], 0 \right]
\end{equation}$$

---

### Pretty Simple, Right?

What is the **problem**?

. . .

<table><tr><td>
  Exogenous Rectangular Grid <br>
  <img src="figures/LaborSeparableRectangularGrid.svg" alt="Exogenous Rectangular Grid">
  </td><td>
  Endogenous Curvilinear Grid <br>
  <img src="figures/LaborSeparableWarpedGrid.svg" alt="Endogenous Curvilinear Grid">
</td></tr></table>

. . .

- **One solution:** Curvilinear Interpolation by @White2015

---

### Warped Grid Interpolation

- Our solution: **Warped Grid Interpolation** (simpler, faster, more details on paper)

![](figures/WarpedInterpolation.svg)

---

### A more complex problem {.smaller}

Consumption - Pension Deposit Problem as in **Druedahl and Jorgensen [2017]**

$$\begin{equation}
\begin{split}
    \vFunc_{t}(\mRat_{t}, \nRat_{t}) & = \max_{\cRat_{t}, \dRat_{t}} \util(\cRat_{t}) + \DiscFac \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA} \vFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1}) \right] \\
    & \text{s.t.} \quad \cRat_{t} \ge 0, \quad \dRat_{t} \ge 0 \\
    \aRat_{t} & = \mRat_{t} - \cRat_{t} - \dRat_{t} \\
    \bRat_{t} & = \nRat_{t} + \dRat_{t} + g(\dRat_{t}) \\
    \mRat_{t+1} & = \aRat_{t} \Rfree / \PGro_{t+1}  + \tShkEmp_{t+1} \\
    \nRat_{t+1} & = \bRat_{t} \Risky_{t+1}  / \PGro_{t+1}
  \end{split}
\end{equation}$$

where

$$\begin{equation}
  \uFunc(\cRat) = \frac{\cRat^{1-\CRRA}}{1-\CRRA} \qquad \text{and} \qquad \gFunc(\dRat) = \xFer \log(1+\dRat).
\end{equation}$$

is a tax-advantaged premium on pension contributions.

---

### G2EGM from Druedahl and Jorgensen [2017]

- If we try to use EGM:
  - 2 first order conditions
  - difficult to handle multiple constraints
  - requires local triangulation interpolation

---

### Breaking up the problem makes it easier to solve

Consider the problem of a consumer who chooses how much to put into a pension account:

$$\begin{equation}
\begin{split}
    \vFunc_{t}(\mRat_{t}, \nRat_{t}) & = \max_{\dRat_{t}} \vOpt_{t}(\lRat_{t}, \bRat_{t}) \\
    & \text{s.t.}  \quad \dRat_{t} \ge 0 \\
    \lRat_{t} & = \mRat_{t} - \dRat_{t} \\
    \bRat_{t} & = \nRat_{t} + \dRat_{t} + g(\dRat_{t})
  \end{split}
\end{equation}$$

. . .

After, the consumer chooses how much to consume out of liquid savings:

$$\begin{equation}
\begin{split}
    \vOpt_{t}(\lRat_{t}, \bRat_{t}) & = \max_{\cRat_{t}} \util(\cRat_{t}) + \DiscFac \wFunc_{t}(\aRat_{t}, \bRat_{t})  \\
    & \text{s.t.} \quad \cRat_{t} \ge 0 \\
    \aRat_{t} & = \lRat_{t} - \cRat_{t}
  \end{split}
\end{equation}$$

---

### Solving the pension problem

The pension problem, more compactly

$$\begin{equation}
\vFunc_{t}(\mRat_{t}, \nRat_{t}) = \max_{\dRat_{t}}
  \vOpt_{t}(\mRat_{t} - \dRat_{t}, \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t}))
\end{equation}$$

. . .

Interior solution must satisfy the first-order condition:

$$\begin{equation}
\gFunc'(\dRat_{t}) = \frac{\vOpt_{t}^{\lRat}(\lRat_{t},
    \bRat_{t})}{\vOpt_{t}^{\bRat}(\lRat_{t}, \bRat_{t})} - 1
\end{equation}$$

. . .

Inverting, we can obtain the optimal choice of $\dRat_{t}$:

$$\begin{equation}
\dEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \gFunc'^{-1}\left(
  \frac{\vOpt_{t}^{\lRat}(\lRat_{t},
    \bRat_{t})}{\vOpt_{t}^{\bRat}(\lRat_{t},
    \bRat_{t})} - 1 \right)
\end{equation}$$

. . .

Using resource constraints we obtain endogenous grids:

$$\begin{equation}
  \nEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \bRat_{t} -
  \dEndFunc_{t}(\lRat_{t}, \bRat_{t}) - \gFunc(\dEndFunc_{t}(\lRat_{t},
    \bRat_{t})) \\
  \mEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \lRat_{t} +
  \dEndFunc_{t}(\lRat_{t}, \bRat_{t})
\end{equation}$$

---

### Unstructured Grids

Problem: **Rectilinear** exogenous grid results in **unstructured** endogenous grid

<table><tr><td>
  Exogenous Rectangular Grid <br>
  <img src="figures/SparsePensionExogenousGrid.svg" alt="Sparse Pension Exogenous Grid">
  </td><td>
  Endogenous Unstructured Grid <br>
  <img src="figures/PensionEndogenousGrid.svg" alt="Unstructured Pension Endogenous Grid">
</td></tr></table>

How do we **interpolate** on this grid?

---

### Gaussian Process Regression

A Gaussian Process is a probability distribution over functions

$$\begin{equation}
\begin{gathered}
    \mathbf{X} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{\Sigma}) \quad \text{s.t.} \quad x_i \sim \mathcal{N}(\mu_i, \sigma_{ii}) \\
    \text{and} \quad  \sigma_{ij} = \Ex[(x_i - \mu_i)(x_j - \mu_j)] \quad \forall i,j \in \{1, \ldots, n\}.
  \end{gathered}
\end{equation}$$

where

$$\begin{equation}
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
\end{equation}$$

A Gaussian Process Regression is used to find the function that best fits a set of data points

$$\begin{equation}
\mathbb{P}(\mathbf{f} | \mathbf{X}) = \mathcal{N}(\mathbf{f} | \mathbf{m}, \mathbf{K})
\end{equation}$$

We use standard covariance function, exploring alternatives is an active area of research

$$\begin{equation}
k(\mathbf{x}_i, \mathbf{x}_j) = \sigma^2_f \exp\left(-\frac{1}{2l^2} (\mathbf{x}_i - \mathbf{x}_j)' (\mathbf{x}_i - \mathbf{x}_j)\right).
\end{equation}$$

---

### An example

Consider the true function $f(x) = x \cos(1.5x)$ sampled at random points

![True Function](figures/GPR_True_Function.svg)

---

### An Example

A random sample of the GP posterior distribution of functions

![Posterior Sample](figures/GPR_Posterior_Sample.svg)

---

### An Example

Gaussian Process Regression finds the function that best fits the data

![Alt text](figures/GaussianProcessRegression.svg)

- **Gaussian Process Regression** gives us
  - **Mean** function of the posterior distribution
  - **Uncertainty quantification** of the mean function
  - Can be useful to predict ex-post where we might need **more points**

---

### Back to the model

Second Stage Pension Endogenous Grid

![](figures/2ndStagePensionEndogenousGrid.svg)

---

### Some Results

<table><tr><td>
  Consumption Function <br>
  <img src="figures/PensionConsumptionFunction.svg" alt="Pension Consumption Function">
  </td><td>
  Deposit Function <br>
  <img src="figures/PensionDepositFunction.svg" alt="Pension Deposit Function">
</td></tr></table>

---

### Conditions for using Sequential EGM

- Model must be
  - concave
  - differentiable
  - continuous
  - separable

Need an **additional** function to exploit **invertibility**

. . .

Examples in this paper:

- Separable utility function
  - $\uFunc(\cRat, \leisure) = \uFunc(\cRat) + \h(\leisure)$
- Continuous and differentiable transition
  - $\bRat_{t}  = \nRat_{t} + \dRat_{t} + g(\dRat_{t})$

---

### Thank you!

<center><img src="public/econ-ark-logo.png" align="center"></center>
<center><img src="public/PoweredByEconARK.svg" align="center"></center>

[`engine: github.com/econ-ark/HARK`](https://github.com/econ-ark/HARK)

[`code: github.com/alanlujan91/SequentialEGM`](https://github.com/alanlujan91/SequentialEGM)

[`website: alanlujan91.github.io/SequentialEGM/egmn`](https://alanlujan91.github.io/SequentialEGM/egmn)
