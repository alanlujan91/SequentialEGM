---
title: "EGMⁿ: The Sequential Endogenous Grid Method"
author: "Alan Lujan"
date: 2023/02/15
---

# EGMⁿ: The Sequential Endogenous Grid Method

## Abstract

Heterogeneous agent models with multiple decisions (consumption, labor, portfolio allocation) face a computational dilemma. Joint optimization over all choices is slow because it requires evaluating high-dimensional optimization problems at every point in a large state space. Imposing strong separability restrictions speeds things up but rules out economically interesting interactions between decisions. We show how to resolve this tension by recognizing that while decisions are economically simultaneous (made under the same information set), they need not be computationally joint. By decomposing the problem into sequential stages, each stage can exploit the efficiency of the Endogenous Grid Method, effectively chaining the speed gains across decisions. The resulting endogenous grids are warped or unstructured, but we develop interpolation methods using both curvilinear techniques and Gaussian Process Regression that preserve accuracy while maintaining speed. The method solves multidecision heterogeneous agent models in a fraction of the time required by standard grid search approaches. Software is available under the Econ-ARK/HARK project.

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
