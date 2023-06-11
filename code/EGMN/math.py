# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: egmn-dev
#     language: python
#     name: python3
# ---

# %%
from sympy import *

# %%
c, rho, gamma = symbols("c rho gamma")
l, z = symbols("l z")

# %%
util = -gamma * z ** (1 - rho) / (1 - rho)
util

# %%
util.diff(z)

# %%
