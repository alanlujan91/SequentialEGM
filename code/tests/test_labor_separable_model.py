"""Gating tests for the labor-separable and disutility paths.

They cover LaborSeparableConsumerType and the Disutility=True branch.
The tests are smoke-style: they construct the agent at the bundled default
calibration, solve, and assert that policy outputs (within the solved grid)
are finite and lie in the admissible economic range. A regression in the FOC
inversion, the labor / leisure conversion, or the disutility branch will trip
an assertion rather than silently producing nonsensical policies.
"""

from __future__ import annotations

import numpy as np
import pytest
from HARKmodels.ConsLaborSeparableModel import (
    LaborSeparableConsumerType,
    consumption_euler_residual,
)


def solved_first_period(disutility: bool):
    """One-cycle default-calibration solve; returns the first-period solution."""
    agent = LaborSeparableConsumerType()
    agent.cycles = 1
    agent.Disutility = disutility
    agent.update()
    agent.solve()
    assert len(agent.solution) >= 1
    return agent.solution[0]


def test_labor_separable_consumer_solves_at_default_calibration() -> None:
    """`LaborSeparableConsumerType` solves and returns admissible policies.

    Guards the shock-broadcasting contract inside ``calc_EndOfPrdvP``:
    HARK's modern ``expected`` hands the full atom array to the integrand at
    once, so the integrand must broadcast the transitory-shock vector against
    the (n_a, n_atoms) ``bnrm`` grid (rather than the per-atom contract that
    the older API used). A regression to the per-atom pattern would raise a
    ``ValueError`` here.
    """
    sol = solved_first_period(disutility=False)
    b_vals = np.array([0.5, 1.0, 2.5, 5.0])
    t_vals = np.array([1.0, 1.0, 1.0, 1.0])

    labor = sol.labor_leisure.labor_func(b_vals, t_vals)
    leisure = sol.labor_leisure.leisure_func(b_vals, t_vals)
    consumption = sol.labor_leisure.c_func(b_vals, t_vals)

    assert np.all(np.isfinite(labor))
    assert np.all(np.isfinite(leisure))
    assert np.all(np.isfinite(consumption))
    assert np.allclose(labor + leisure, 1.0, atol=1e-6)
    assert np.all(consumption >= 0.0)


def test_labor_separable_consumer_disutility_branch_solves() -> None:
    """The `Disutility=True` branch of `LaborSeparableSolver` solves successfully."""
    sol = solved_first_period(disutility=True)
    labor = sol.labor_leisure.labor_func(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
    leisure = sol.labor_leisure.leisure_func(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
    assert np.all(np.isfinite(labor))
    assert np.all(np.isfinite(leisure))
    assert np.allclose(labor + leisure, 1.0, atol=1e-6)


def test_consumption_euler_equation_holds() -> None:
    """The consumption stage satisfies its Euler equation
    ``u'(c_t) = beta s R E[u'(c_{t+1})]``, with next period's consumption taken
    through the full composition (labor policy at ``b' = R a``, then the
    consumption stage), to interpolation accuracy.

    Guards two defects: the labor stage keyed ``c`` on the wrong bank balance
    (rectangular ``b`` values on the endogenous mesh, up to 0.45 off), and the
    end-of-period marginal value omitted ``DiscFac * LivPrb * Rfree``. Either
    one pushes the residual past 1e-2.
    """
    agent = LaborSeparableConsumerType()
    agent.cycles = 20
    agent.update()
    agent.solve()
    residual = consumption_euler_residual(agent, np.linspace(0.3, 10.0, 200))
    assert np.max(np.abs(residual)) < 2e-3


def test_zero_wage_state_is_required() -> None:
    """Without a zero-wage state the recursion returns NaN policies, so
    construction raises.
    """
    with pytest.raises(ValueError, match="zero-wage state"):
        LaborSeparableConsumerType(UnempPrb=0.0)
    with pytest.raises(ValueError, match="zero-wage state"):
        LaborSeparableConsumerType(IncUnemp=0.3)


def test_labor_stage_consumption_composes_the_stages() -> None:
    """At every node ``(b, theta)`` of the labor stage's endogenous mesh, its
    consumption equals the consumption stage at the cash on hand that labor
    choice produces, ``c_cs(b + w theta labor(b, theta))``.

    Guards the defect that keyed consumption at rectangular bank balances,
    ``c_cs(m_i + w theta labor(m_i))``, to the endogenous nodes ``b_ij``.
    """
    agent = LaborSeparableConsumerType()
    agent.cycles = 3
    agent.update()
    agent.solve()
    wage = agent.WageRte[0]
    sol = agent.solution[0]
    grids = sol.labor_leisure.grids
    b, theta = grids["bNrm"][:, 1:], grids["tShk"][:, 1:]
    labor = sol.labor_leisure.labor_func(b, theta)
    composed = sol.consumption_saving.c_func(b + wage * theta * labor)
    np.testing.assert_allclose(sol.labor_leisure.c_func(b, theta), composed, rtol=1e-12)
