---
title: Abstract # a string (max 500 chars) page & project
# description: # a string (max 500 chars) page & project
short_title: Abstract # a string (max 40 chars) page & project
# name:  # a string (max 500 chars) page & project
# tags:  # a list of strings page only
# thumbnail: # a link to a local or remote image page only
# subtitle: # a string (max 500 chars) page only
# date: # a valid date formatted string page can override project
# authors:  # a list of author objects page can override project
# doi:  # a valid DOI, either URL or id page can override project
# arxiv: # a valid arXiv reference, either URL or id page can override project
# open_access: # boolean (true/false) page can override project
# license: # a license object or a string page can override project
# github: # a valid GitHub URL or owner/reponame page can override project
# binder: # any valid URL page can override project
# subject: # a string (max 40 chars) page can override project
# venue: # a venue object page can override project
# biblio: # a biblio object with various fields page can override project
---

+++ {"part": "abstract"}

Heterogeneous agent models with multiple decisions are often solved using inefficient grid search methods that require
many evaluations and are slow.
This paper provides a novel method for solving such models using an extension of the Endogenous Grid Method (EGM) that
uses Gaussian Process Regression (GPR) to interpolate functions on unstructured grids.
First, I propose an intuitive and strategic procedure for decomposing a problem into subproblems which allows the use of
efficient solution methods.
Second, using an exogenous grid of post-decision states and solving for an endogenous grid of pre-decision states that
obey a first-order condition greatly speeds up the solution process.
Third, since the resulting endogenous grid can often be non-rectangular at best and unstructured at worst, GPR provides
an efficient and accurate method for interpolating the value, marginal value, and decision functions.
Applied sequentially to each decision within the problem, the method is able to solve heterogeneous agent models with
multiple decisions in a fraction of the time and with less computational resources than are required by standard methods
currently used.
Software to reproduce these methods is available under the [`Econ-ARK/HARK`](https://econ-ark.org/) project for
the `python` programming language.

+++

+++ {"part": "acknowledgements"}

I would like to thank Christopher D. Carroll and Simon Scheidegger for their helpful comments and suggestions. The
remaining errors are my own. All figures and other numerical results were produced using
the [`Econ-ARK/HARK`](https://econ-ark.org/) toolkit ({cite:t}`Carroll2018`). Additional libraries used in the
production of this paper include but are not limited to: [`scipy`](https://www.scipy.org/) ({cite:
t}`Virtanen2020`), [`numpy`](https://www.numpy.org/) ({cite:t}`Harris2020`), [`numba`](https://numba.pydata.org/) (
{cite:t}`Lam2015`), [`cupy`](https://cupy.dev/) ({cite:t}`Okuta2017`), [`scikit-learn`](https://scikit-learn.org/) (
{cite:t}`Pedregosa2011`), [`pytorch`](https://pytorch.org/) ({cite:t}`Paszke2019`),
and [`gpytorch`](https://gpytorch.ai/) ({cite:t}`Gardner2018`)

+++
