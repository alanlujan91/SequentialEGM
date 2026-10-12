"""Tests of the portfolio-share extension of the consumption-labor model."""

from __future__ import annotations

import numpy as np
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType
from HARKmodels.ConsLaborPortfolioModel import (
    LaborPortfolioConsumerType,
    LaborPortfolioSolver,
    first_crossing_share,
    portfolio_residuals,
)
from HARKmodels.ConsLaborSeparableModel import LaborSeparableSolver


def solved(cycles: int, **params) -> LaborPortfolioConsumerType:
    """A solved portfolio agent at the default calibration plus ``params``."""
    agent = LaborPortfolioConsumerType(cycles=cycles, **params)
    agent.solve()
    return agent


def test_worthless_leisure_reproduces_harks_portfolio_model() -> None:
    """With leisure nearly worthless, labor is 1 to within 1e-6 and the model is
    HARK's own
    ``PortfolioConsumerType`` with wage income: an independent implementation of
    the same budget and the same first-crossing share rule, given the same
    shock, return and share grids, must deliver the same policies.
    """
    lab = solved(10, LeisureFactor=1e4)
    hark = PortfolioConsumerType(
        cycles=10,
        **{
            k: getattr(lab, k)
            for k in ("CRRA", "DiscFac", "LivPrb", "PermGroFac", "Rfree")
        },
    )
    for name in ("IncShkDstn", "RiskyDstn", "ShockDstn", "ShareGrid", "aXtraGrid"):
        setattr(hark, name, getattr(lab, name))
    hark.solve()
    m = np.linspace(0.2, 30.0, 300)
    sol = lab.solution[0]
    labor = sol.labor_leisure.labor_func(m, np.ones_like(m))
    np.testing.assert_allclose(labor, 1.0, rtol=0, atol=1e-6)
    c = sol.consumption_saving.c_func(m)
    np.testing.assert_allclose(c, hark.solution[0].cFuncAdj(m), rtol=5e-4)
    np.testing.assert_allclose(
        sol.post_decision.share_func(m - c), hark.solution[0].ShareFuncAdj(m), atol=1e-3
    )


def test_first_crossing_share_interpolates_and_selects_corners() -> None:
    """Interior rows return the linear interpolant's first zero; rows of one
    sign return the corner that sign implies.
    """
    grid = np.linspace(0.0, 1.0, 5)
    foc = np.array(
        [
            [1.0, 0.5, -0.5, -1.0, -2.0],  # zero at 0.25 + 0.5 * 0.25
            [1.0, 0.8, 0.6, 0.4, 0.2],  # positive throughout: share 1
            [-0.1, -0.2, -0.3, -0.4, -0.5],  # negative throughout: share 0
            [1.0, -1.0, 1.0, -1.0, -1.0],  # the first crossing is the one chosen
        ]
    )
    np.testing.assert_allclose(
        first_crossing_share(foc, grid), [0.375, 1.0, 0.0, 0.125], rtol=0, atol=1e-15
    )


def test_portfolio_solver_reuses_the_consumption_and_labor_stages() -> None:
    """The portfolio solver overrides only the end-of-period step."""
    for name in (
        "prepare_to_solve",
        "make_consumption_solution",
        "make_labor_leisure_solution",
        "make_consumer_labor_solution",
        "solve",
    ):
        assert getattr(LaborPortfolioSolver, name) is getattr(
            LaborSeparableSolver, name
        )
    assert (
        LaborPortfolioSolver.calc_EndOfPrdvP is not LaborSeparableSolver.calc_EndOfPrdvP
    )


def test_portfolio_euler_and_share_conditions_hold() -> None:
    """The Euler equation with the portfolio return holds to interpolation
    accuracy, the share condition is near 0 at interior shares and positive at
    the share-one corner, and both kinds of share occur on the checked range.
    """
    agent = solved(cycles=20)
    m = np.linspace(0.3, 10.0, 200)
    euler, share_cond = portfolio_residuals(agent, m)
    sol = agent.solution[0]
    share = sol.post_decision.share_func(m - sol.consumption_saving.c_func(m))
    interior = (share > 0.0) & (share < 1.0)
    corner = share == 1.0
    assert interior.any() and corner.any()
    assert np.max(np.abs(euler)) < 1e-3
    assert np.max(np.abs(share_cond[interior])) < 2e-3
    assert np.all(share_cond[corner] > 0.0)


def test_lower_equity_premium_lowers_the_share() -> None:
    """A smaller mean risky return lowers the risky share at every asset level."""
    a = np.geomspace(1e-3, 60.0, 50)
    base = solved(cycles=10).solution[0].post_decision.share_func(a)
    low = solved(cycles=10, RiskyAvg=1.05).solution[0].post_decision.share_func(a)
    assert np.all(low < base)


def test_labor_portfolio_disutility_branch_finite() -> None:
    """The ``Disutility=True`` branch solves with finite, admissible policies."""
    sol = solved(cycles=3, Disutility=True).solution[0]
    bmat, tmat = np.meshgrid(
        np.linspace(0.5, 10.0, 12), np.linspace(0.9, 1.2, 5), indexing="ij"
    )
    labor = sol.labor_leisure.labor_func(bmat, tmat)
    leisure = sol.labor_leisure.leisure_func(bmat, tmat)
    consumption = sol.labor_leisure.c_func(bmat, tmat)
    assert np.all(np.isfinite(labor)) and np.all(np.isfinite(consumption))
    np.testing.assert_allclose(labor + leisure, 1.0, atol=1e-6)
    assert np.all(consumption >= 0.0)
