---
title: "EGMⁿ: The Sequential Endogenous Grid Method"
author: "Alan Lujan"
date: 2023/02/15
---

# EGMⁿ: The Sequential Endogenous Grid Method

## Abstract

Heterogeneous agent models with multiple decisions are often solved using inefficient grid search methods that require a large number of points and are time intensive. This paper provides a novel method for solving such models using an extension of the endogenous grid method (EGM) that uses Gaussian Process Regression (GPR) to interpolate functions on unstructured grids. First, separating models into smaller, sequential problems allows the problems to be more tractable and easily analyzed. Second, using an exogenous grid of post-decision states and solving for an endogenous grid of pre-decision states that obey a first order condition greatly speeds up the solution process. Third, since the resulting endogenous grid can often be curvilinear at best and unstructured at worst, GPR provides an efficient and accurate method for interpolating the value, marginal value, and policy functions. Applied sequentially to each decision within the overarching problem, the method is able to solve heterogeneous agent models with multiple decisions in a fraction of the time and with less computational resources than are required by standard grid search methods currently used. This paper also illustrates how this method can be applied to a number of increasingly complex models. Software is provided in the form of a Python module under the `HARK` package.

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alanlujan91/SequentialEGM/workflows/CI/badge.svg
[actions-link]:             https://github.com/alanlujan91/SequentialEGM/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/SequentialEGM
[conda-link]:               https://github.com/conda-forge/SequentialEGM-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/alanlujan91/SequentialEGM/discussions
[pypi-link]:                https://pypi.org/project/SequentialEGM/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/SequentialEGM
[pypi-version]:             https://img.shields.io/pypi/v/SequentialEGM
[rtd-badge]:                https://readthedocs.org/projects/SequentialEGM/badge/?version=latest
[rtd-link]:                 https://SequentialEGM.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
