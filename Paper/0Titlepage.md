---
title: The Sequential Endogenous Grid Method
subject: Economics
# subtitle: Evolve your markdown documents into structured data
short_title: "EGM$^n$"
authors:
  - name: Alan Lujan
    affiliations:
      - The Ohio State University
      - Econ-ARK
    # orcid: 0000-0002-7859-8394
    email: alanlujan91@gmail.com
license: CC-BY-4.0
keywords: Endogenous Grid Method, Gaussian Processes, Machine Learning, Stochastic Dynamic Programming
exports:
    - format: pdf
      template: arxiv_nips
      output: arxiv.zip

---

+++ {"part": "abstract"}

Heterogeneous agent models with multiple decisions are often solved using inefficient grid search methods that require many evaluations and are slow.
This paper provides a novel method for solving such models using an extension of the Endogenous Grid Method (EGM) that uses Gaussian Process Regression (GPR) to interpolate functions on unstructured grids.
First, I propose an intuitive and strategic procedure for decomposing a problem into subproblems which allows the use of efficient solution methods.
Second, using an exogenous grid of post-decision states and solving for an endogenous grid of pre-decision states that obey a first-order condition greatly speeds up the solution process.
Third, since the resulting endogenous grid can often be non-rectangular at best and unstructured at worst, GPR provides an efficient and accurate method for interpolating the value, marginal value, and decision functions.
Applied sequentially to each decision within the problem, the method is able to solve heterogeneous agent models with multiple decisions in a fraction of the time and with less computational resources than are required by standard methods currently used.
Software to reproduce these methods is available under the [`Econ-ARK/HARK`](https://econ-ark.org/) project for the `python` programming language.

+++

+++ {"part": "acknowledgements"}
I would like to thank Christopher D. Carroll and Simon Scheidegger for their helpful comments and suggestions. The remaining errors are my own. All figures and other numerical results were produced using the [`Econ-ARK/HARK`](https://econ-ark.org/) toolkit ({cite:t}`Carroll2018`). Additional libraries used in the production of this paper include but are not limited to: [`scipy`](https://www.scipy.org/) ({cite:t}`Virtanen2020`), [`numpy`](https://www.numpy.org/) ({cite:t}`Harris2020`), [`numba`](https://numba.pydata.org/) ({cite:t}`Lam2015`), [`cupy`](https://cupy.dev/) ({cite:t}`Okuta2017`), [`scikit-learn`](https://scikit-learn.org/) ({cite:t}`Pedregosa2011`), [`pytorch`](https://pytorch.org/) ({cite:t}`Paszke2019`), and [`gpytorch`](https://gpytorch.ai/) ({cite:t}`Gardner2018`)
+++
