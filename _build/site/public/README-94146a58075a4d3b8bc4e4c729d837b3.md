---
title: "EGM$^n$: The Sequential Endogenous Grid Method"
author: "Alan Lujan"
date: 2023/02/15
---

# EGM$^n$: The Sequential Endogenous Grid Method

## Abstract

Heterogeneous agent models with multiple decisions are often solved using inefficient grid search methods that require a large number of points and are time intensive. This paper provides a novel method for solving such models using an extension of the endogenous grid method (EGM) that uses Gaussian Process Regression (GPR) to interpolate functions on unstructured grids. First, separating models into smaller, sequential problems allows the problems to be more tractable and easily analyzed. Second, using an exogenous grid of post-decision states and solving for an endogenous grid of pre-decision states that obey a first order condition greatly speeds up the solution process. Third, since the resulting endogenous grid can often be curvilinear at best and unstructured at worst, GPR provides an efficient and accurate method for interpolating the value, marginal value, and policy functions. Applied sequentially to each decision within the overarching problem, the method is able to solve heterogeneous agent models with multiple decisions in a fraction of the time and with less computational resources than are required by standard grid search methods currently used. This paper also illustrates how this method can be applied to a number of increasingly complex models. Software is provided in the form of a Python module under the `HARK` package.

## Replication from a unix (macOS/linux) command line

To reproduce all the computational results of the paper:

```
	/bin/bash reproduce_document.sh
```

To produce pdf version of the paper from a unix (macOS/linux) command line:

```
	/bin/bash reproduce_document.sh
```

To reproduce both computational results and the paper:

```
	/bin/bash reproduce.sh
```
