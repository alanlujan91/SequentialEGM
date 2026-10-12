"""Unit tests of the EGM$^n$ Phase 3 cell kernels in ``ssj.hh_egm``."""

from __future__ import annotations

import numpy as np
import pytest
from ssj import benchmark
from ssj.hh_egm import (
    PRODUCTION_RULE,
    advance_bracket,
    backward_step,
    cell_choice,
    corner_costs,
    kernel_tuples,
    stage0_columns,
)

RA, RB, CHI0, CHI1, CHI2 = 1.0125, 1.0075, 0.25, 6.5, 2.0
PARAMS = (0.976, 0.5, RA, RB, CHI0, CHI1, CHI2)
A_GRID = np.array([0.0, 1.0, 10.0])
B_GRID = np.array([0.0, 5.0])
GRIDS = (A_GRID, B_GRID, np.array([0.0, 1.0, 10.0]), np.array([0.1]))


def foc_slope(a_e, ag):
    """``1 + Psi_1(a', a)``: the liquid resources fall in ``a'`` iff it is positive."""
    u = (a_e - RA * ag) / (RA * ag + CHI0)
    return 1.0 + CHI1 * np.sign(u) * abs(u) ** (CHI2 - 1.0)


def choice_at(ib, ia, apv):
    """``cell_choice`` at ``(0, ib, ia)`` with the regridded choice ``apv`` there."""
    a_pol = np.full((1, B_GRID.size, A_GRID.size), 0.5)
    a_pol[0, ib, ia] = apv
    psi_a0 = corner_costs(A_GRID, RA, CHI0, CHI1, CHI2)
    return cell_choice(0, ib, ia, a_pol, psi_a0, GRIDS, PARAMS)


def test_infeasible_corner_delivers_the_kkt_floor() -> None:
    """A negative regridded ``a'`` at a cell whose corner resources are negative
    (``b = 0``, ``a = 10``) is delivered at ``1 + Psi_1 = 0``, a positive ``a'``
    with positive liquid resources, never at the negative regridded value.
    """
    psi_a0 = corner_costs(A_GRID, RA, CHI0, CHI1, CHI2)
    assert GRIDS[3][0] + RA * A_GRID[2] - psi_a0[2] < 0.0
    corner, a_e, ell, *_ = choice_at(0, 2, -0.5)
    assert not corner
    assert a_e > 0.0
    assert foc_slope(a_e, A_GRID[2]) == pytest.approx(0.0, abs=1e-12)
    assert ell > 0.0


@pytest.mark.parametrize("apv", [-0.5, 0.4])
def test_dominated_corner_delivers_the_kkt_floor(apv) -> None:
    """At ``b = 5``, ``a = 1`` the corner resources are positive but ``1 + Psi_1``
    vanishes at a positive ``a'``, so any regridded ``a'`` below that point, a
    negative one included, is delivered there: it leaves more liquid resources
    than the ``a' = 0`` corner and more illiquid wealth too.
    """
    psi_a0 = corner_costs(A_GRID, RA, CHI0, CHI1, CHI2)
    zc = GRIDS[3][0] + RB * B_GRID[1] + RA * A_GRID[1] - psi_a0[1]
    assert zc > 0.0
    corner, a_e, ell, *_ = choice_at(1, 1, apv)
    assert not corner
    assert a_e > max(apv, 0.0)
    assert foc_slope(a_e, A_GRID[1]) == pytest.approx(0.0, abs=1e-12)
    assert ell > zc


def test_feasible_corner_delivers_zero() -> None:
    """At ``a = 0`` the KKT floor is zero, so a negative ``a'`` is the corner."""
    corner, a_e, ell, *_ = choice_at(0, 0, -0.5)
    assert corner
    assert a_e == 0.0
    assert ell > 0.0


def test_flat_stage1_step_takes_the_coincident_node() -> None:
    """A flat first step joins two copies of the constraint point, so the weight
    on the first node is exact at 1.
    """
    assert advance_bracket(np.array([0.0, 0.0, 1.0]), 0.0, 0) == (0, 1.0)


def test_falling_stage1_step_weight_is_nan() -> None:
    """A falling last step has no Stage 1 inversion: the weight is NaN."""
    ip, pi = advance_bracket(np.array([0.0, 2.0, 1.0]), 3.0, 0)
    assert ip == 1
    assert np.isnan(pi)


def test_falling_stage1_step_stops_the_solve_unconverged() -> None:
    """A continuation whose ``Vb_p`` jumps up at the top ``b'`` node breaks Stage 1
    monotonicity: ``c = 100`` at the second-to-last ``b'`` node and ``c ~ 0`` at
    the last put ``l = b' + c`` at about 123 and 50, a falling last step inside
    ``l_grid``. Past the fall the Stage 1 policy is NaN. Within two steps the
    marginals turn NaN too, and the iteration stops unconverged in Python.
    """
    cal = benchmark.calibration(8, 6)
    args = benchmark.backward_args(cal)
    assert args[2][-1] > cal["b"][-2] + 100.0

    def corrupt(Vb_p):
        bad = Vb_p.copy()
        bad[:, -2, :] = 1e-4 / cal["beta"]
        bad[:, -1, :] = 1e6
        return bad

    Va, Vb = benchmark.initial_marginals(cal)
    grids, params = kernel_tuples(*args[:4], *args[5:])
    c_s1 = stage0_columns(
        benchmark.expect(cal["Pi"], Va),
        corrupt(benchmark.expect(cal["Pi"], Vb)),
        grids,
        params,
        PRODUCTION_RULE,
    )[0]
    assert np.isnan(c_s1).any()

    def step(Va_p, Vb_p):
        return backward_step(Va_p, corrupt(Vb_p), *args, 1)[:5]

    out = benchmark.iterate(step, cal)
    assert not np.all(np.isfinite(out["Vb"]))
    assert not out["converged"]
    assert out["iter"] <= 2
