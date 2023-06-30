# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.12
# ---

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# <h1 style="text-align:center"><strong>EGM$^n$: The Sequential Endogenous Grid Method</strong></h1>
#
# <h2 style="text-align:center">H2: Computational Methods III</h2>
#
# <p style="text-align:center">By</p>
#
# <h2 style="text-align:center"><strong>Alan E. Lujan Solis</strong></h2>
#
# <h2 style="text-align:center">The Ohio State University</h2>
# <h2 style="text-align:center">Econ-ARK</h2>
# <h2 style="text-align:center">July 2023</h2>
#
# $$
# \newcommand{\DiscFac}{\beta}
# \newcommand{\utilFunc}{\mathrm{u}}
# \newcommand{\VFunc}{\mathrm{V}}
# \newcommand{\Leisure}{Z}
# \newcommand{\tShk}{\xi}
# \newcommand{\util}{u}
# \newcommand{\tShkEmp}{\theta}
# \newcommand{\BLev}{B}
# \newcommand{\CLev}{C}
# \newcommand{\Ex}{\mathbb{E}}
# \newcommand{\CRRA}{\rho}
# \newcommand{\labShare}{\nu}
# \newcommand{\leiShare}{\zeta}
# \newcommand{\h}{h}
# \newcommand{\bRat}{b}
# \newcommand{\leisure}{z}
# \newcommand{\cRat}{c}
# \newcommand{\PLev}{P}
# \newcommand{\vFunc}{\mathrm{v}}
# \newcommand{\Rfree}{\mathsf{R}}
# \newcommand{\wage}{\mathsf{w}}
# \newcommand{\riskyshare}{\varsigma}
# \newcommand{\PGro}{\Gamma}
# \newcommand{\labor}{\ell}
# \newcommand{\aRat}{a}
# \newcommand{\mRat}{m}
# \newcommand{\Rport}{\mathbb{R}}
# \newcommand{\Risky}{\mathbf{R}}
# \newcommand{\risky}{\mathbf{r}}
# \newcommand{\vOpt}{\tilde{\mathfrak{v}}}
# \newcommand{\vEnd}{\mathfrak{v}}
# \newcommand{\vE}{{v}^{e}}
# \newcommand{\vOptAlt}{\grave{\tilde{\mathfrak{v}}}}
# \newcommand{\q}{\koppa}
# \newcommand{\cEndFunc}{\mathfrak{c}}
# \newcommand{\cE}{\cRat^{e}}
# \newcommand{\xRat}{x}
# \newcommand{\aMat}{[\mathrm{a}]}
# \newcommand{\mEndFunc}{\mathfrak{m}}
# \newcommand{\mE}{\mRat^{e}}
# \newcommand{\mMat}{[\mathrm{m}]}
# \newcommand{\tShkMat}{[\mathrm{\tShkEmp}]}
# \newcommand{\zEndFunc}{\mathfrak{z}}
# \newcommand{\lEndFunc}{\mathfrak{l}}
# \newcommand{\bEndFunc}{\mathfrak{b}}
# \newcommand{\bE}{\bRat^{e}}
# \newcommand{\nRat}{n}
# \newcommand{\dRat}{d}
# \newcommand{\gFunc}{\mathrm{g}}
# \newcommand{\xFer}{\chi}
# \newcommand{\lRat}{l}
# \newcommand{\wFunc}{\mathrm{w}}
# \newcommand{\dEndFunc}{\mathfrak{d}}
# \newcommand{\nEndFunc}{\mathfrak{n}}
# \newcommand{\uFunc}{\mathrm{u}}
# \newcommand{\TFunc}{\mathrm{T}}
# \newcommand{\UFunc}{\mathrm{U}}
# \newcommand{\WFunc}{\mathrm{W}}
# \newcommand{\yRat}{y}
# \newcommand{\XLev}{X}
# \newcommand{\Retire}{\mathbb{R}}
# \newcommand{\Work}{\mathbb{W}}
# \newcommand{\error}{\epsilon}
# \newcommand{\err}{z}
# \newcommand{\kapShare}{\alpha}
# \newcommand{\kap}{k}
# \newcommand{\cTarg}{\check{c}}
# \newcommand{\Decision}{\mathbb{D}}
# \newcommand{\Prob}{\mathbb{P}}
# $$

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## The Endogenous Grid Method
#
# - Simple
#   - Inverted Euler equation
# - Fast
#   - No root-finding or optimization required
# - Efficient
#   - Finds exact solution at each gridpoint

# %% [markdown] tags=[]
# - Limitations
#   - Only works for one-dimensional problems
#   - Non-convexities can be problematic
#   - Can't be used for problems where Euler equations can't be inverted

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Consumption-Labor-Portfolio Choice Problem
#
# \begin{equation}
# \VFunc_0(\BLev_0, \tShkEmp_0) = \max \Ex_{t} \left[ \sum_{n = 0}^{T-t} \DiscFac^{n} \utilFunc(\CLev_{t+n}, \Leisure_{t+n})  \right]
# \end{equation}
#
# \begin{equation}
# \begin{split}
#     \vFunc_{t}(\bRat_{t}, \tShkEmp_{t}) & = \max_{\{\cRat_{t},
#       \leisure_{t}, \riskyshare_{t}\}} \utilFunc(\cRat_{t}, \leisure_{t}) +
#     \DiscFac \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA}
#       \vFunc_{t+1} (\bRat_{t+1},
#       \tShkEmp_{t+1}) \right] \\
#     & \text{s.t.} \\
#     \labor_{t} & = 1 - \leisure_{t} \\
#     \mRat_{t} & = \bRat_{t} + \tShkEmp_{t} \wage \labor_{t} \\
#     \aRat_{t} & = \mRat_{t} - \cRat_{t} \\
#     \Rport_{t+1} & = \Rfree + (\Risky_{t+1} - \Rfree)
#     \riskyshare_{t} \\
#     \bRat_{t+1} & = \aRat_{t} \Rport_{t+1} / \PGro_{t+1}
#   \end{split}
# \end{equation}

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# Starting from the beginning of the period, we can define the labor-leisure problem as
#
# $$\begin{equation}
# \begin{split}
#     \vFunc_{t}(\bRat_{t}, \tShkEmp_{t}) & = \max_{ \leisure_{t}}
#     \h(\leisure_{t}) + \vOpt_{t} (\mRat_{t}) \\
#     & \text{s.t.} \\
#     0 & \leq \leisure_{t} \leq 1 \\
#     \labor_{t} & = 1 - \leisure_{t} \\
#     \mRat_{t} & = \bRat_{t} + \tShkEmp_{t} \wage \labor_{t}.
#   \end{split}
# \end{equation}$$
#
# The pure consumption-saving problem is then
#
# \begin{equation}
# \begin{split}
#     \vOpt_{t}(\mRat_{t}) & = \max_{\cRat_{t}} \util(\cRat_{t}) + \DiscFac\vEnd_{t}(\aRat_{t}) \\
#     & \text{s.t.} \\
#     0 & \leq \cRat_{t} \leq \mRat_{t} \\
#     \aRat_{t} & = \mRat_{t} - \cRat_{t}.
#   \end{split}
# \end{equation}
#
# Finally, the risky portfolio problem is
#
# \begin{equation}
# \begin{split}
#     \vEnd_{t}(\aRat_{t}) & = \max_{\riskyshare_{t}}
#     \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA}
#       \vFunc_{t+1}(\bRat_{t+1},
#       \tShkEmp_{t+1}) \right] \\
#     & \text{s.t.} \\
#     0 & \leq \riskyshare_{t} \leq 1 \\
#     \Rport_{t+1} & = \Rfree + (\Risky_{t+1} - \Rfree)
#     \riskyshare_{t} \\
#     \bRat_{t+1} & = \aRat_{t} \Rport_{t+1} / \PGro_{t+1}.
#   \end{split}
# \end{equation}

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# \begin{equation}
# \vOpt_{t}(\mRat_{t}) = \max_{\cRat_{t}} \util(\cRat_{t}) +
#   \DiscFac \vEnd_{t}(\mRat_{t}-\cRat_{t})
# \end{equation}
#
# \begin{equation}
# \utilFunc'(\cRat_t) = \DiscFac \vEnd_{t}'(\mRat_{t} - \cRat_{t}) = \DiscFac
#   \vEnd_{t}'(\aRat_{t})
# \end{equation}
#
# \begin{equation}
# \cEndFunc_{t}(\aRat_{t}) = \utilFunc'^{-1}\left( \DiscFac \vEnd_{t}'(\aRat_{t})
#   \right)
# \end{equation}

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# \begin{equation}
# \vFunc_{t}(\bRat_{t}, \tShkEmp_{t}) = \max_{ \leisure_{t}}
#   \h(\leisure_{t}) + \vOpt_{t}(\bRat_{t} +
#   \tShkEmp_{t} \wage (1-\leisure_{t}))
# \end{equation}
#
# \begin{equation}
# \h'(\leisure_{t}) = \vOpt_{t}'(\mRat_{t}) \tShkEmp_{t}
# \end{equation}
#
# \begin{equation}
# \zEndFunc_{t}(\mMat, \tShkMat) = \h'^{-1}\left(
#   \vOpt_{t}'(\mMat) \tShkMat \right)
# \end{equation}
#
# \begin{equation}
# \hat{\zEndFunc}_{t}(\mMat, \tShkMat) = \max \left[ \min \left[ \zEndFunc_{t}(\mMat, \tShkMat), 1 \right], 0 \right]
# \end{equation}

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# \begin{equation}
# \begin{split}
#     \vFunc_{t}(\mRat_{t}, \nRat_{t}) & = \max_{\cRat_{t}, \dRat_{t}} \util(\cRat_{t}) + \DiscFac \Ex_{t} \left[ \PGro_{t+1}^{1-\CRRA} \vFunc_{t+1}(\mRat_{t+1}, \nRat_{t+1}) \right] \\
#     & \text{s.t.} \quad \cRat_{t} \ge 0, \quad \dRat_{t} \ge 0 \\
#     \aRat_{t} & = \mRat_{t} - \cRat_{t} - \dRat_{t} \\
#     \bRat_{t} & = \nRat_{t} + \dRat_{t} + g(\dRat_{t}) \\
#     \mRat_{t+1} & = \aRat_{t} \Rfree / \PGro_{t+1}  + \tShkEmp_{t+1} \\
#     \nRat_{t+1} & = \bRat_{t} \Risky_{t+1}  / \PGro_{t+1}
#   \end{split}
# \end{equation}

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# \begin{equation}
# \begin{split}
#     \vFunc_{t}(\mRat_{t}, \nRat_{t}) & = \max_{\dRat_{t}} \vOpt_{t}(\lRat_{t}, \bRat_{t}) \\
#     & \text{s.t.}  \quad \dRat_{t} \ge 0 \\
#     \lRat_{t} & = \mRat_{t} - \dRat_{t} \\
#     \bRat_{t} & = \nRat_{t} + \dRat_{t} + g(\dRat_{t})
#   \end{split}
# \end{equation}
#
# \begin{equation}
# \begin{split}
#     \vOpt_{t}(\lRat_{t}, \bRat_{t}) & = \max_{\cRat_{t}} \util(\cRat_{t}) + \DiscFac \wFunc_{t}(\aRat_{t}, \bRat_{t})  \\
#     & \text{s.t.} \quad \cRat_{t} \ge 0 \\
#     \aRat_{t} & = \lRat_{t} - \cRat_{t}
#   \end{split}
# \end{equation}

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# \begin{equation}
# \vFunc_{t}(\mRat_{t}, \nRat_{t}) = \max_{\dRat_{t}}
#   \vOpt_{t}(\mRat_{t} - \dRat_{t}, \nRat_{t} + \dRat_{t} + \gFunc(\dRat_{t}))
# \end{equation}
#
# \begin{equation}
# \vOpt_{t}^{\lRat}(\lRat_{t}, \bRat_{t})(-1) +
#   \vOpt_{t}^{\bRat}(\lRat_{t}, \bRat_{t})(1+\gFunc'(\dRat_{t})) = 0.
# \end{equation}
#
# \begin{equation}
# \gFunc'(\dRat_{t}) = \frac{\vOpt_{t}^{\lRat}(\lRat_{t},
#     \bRat_{t})}{\vOpt_{t}^{\bRat}(\lRat_{t}, \bRat_{t})} - 1
# \end{equation}
#
# \begin{equation}
# \dEndFunc_{t}(\lRat_{t}, \bRat_{t}) = \gFunc'^{-1}\left(
#   \frac{\vOpt_{t}^{\lRat}(\lRat_{t},
#     \bRat_{t})}{\vOpt_{t}^{\bRat}(\lRat_{t},
#     \bRat_{t})} - 1 \right)
# \end{equation}

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ![Sparse Pension Exogenous Grid](../../content/figures/SparsePensionExogenousGrid.svg)

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ![Unstructured Pension Endogenous Grid](../../content/figures/PensionEndogenousGrid.svg)
