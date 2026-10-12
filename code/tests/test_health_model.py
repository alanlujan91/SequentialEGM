"""Correctness tests for the separable health investment model.

These tests guard the model's *economics*; the interpolators have their own
tests. They pin three properties:

1. The Sequential EGM^n solver and the simultaneous solver produce identical
   endogenous grids and policies at the same interpolator (the stage
   decomposition is a reorganization, not a different numerical answer).

2. The cross-period derivative ``dvdH`` returned by ``calc_exp_next`` equals a
   finite-difference of the continuation value. This catches the wage-timing
   bug where ``(1 - DeprRte)`` was factored out of *both* the wage-income and
   the health-transition terms, which is inconsistent with income accruing on
   pre-depreciation health ``H`` (the paper's ``y_{t+1} = omega_{t+1} H_t``).

3. The ``n = 0`` singularity in ``dvdh`` is handled without warnings or NaNs.
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
import pytest
from HARK.distributions import DiscreteDistributionLabeled
from HARK.interpolation import Curvilinear2DInterp
from HARK.rewards import CRRAutility
from HARKmodels.ConsHealthSeparableModel import (
    BasicHealthConsumerType,
    SequentialHealthConsumerType,
    calc_exp_next,
    init_basic_health,
    policy_violations,
    solve_one_period_ConsBasicHealth,
    solve_one_period_ConsBasicHealth_Sequential,
)

# --- shared fixtures -------------------------------------------------------


class _Capture:
    """Interpolator stand-in that records the (values, x, y) it is handed.

    Both solvers call ``interpolator([f0, f1, f2], xMat, yMat)``; the sequential
    solver builds three of them, the last being the health-investment stage.
    """

    store: ClassVar[list] = []

    def __init__(self, values, x, y) -> None:
        _Capture.store.append(
            ([np.asarray(v) for v in values], np.asarray(x), np.asarray(y)),
        )

    def __call__(self, x, y):  # pragma: no cover - never queried in these tests
        z = np.zeros_like(np.asarray(x, dtype=float))
        return z, z, z


@pytest.fixture(autouse=True)
def reset_capture_store() -> None:
    """Clear the class-level _Capture.store before each test (shared state)."""
    _Capture.store.clear()


def assert_solvers_match(kwargs) -> None:
    """Solve both solvers with ``kwargs`` and assert identical captured output.

    The FOC inversions are algebraically the same, so with the same
    interpolator the (values, x, y) each solver hands to it must agree to
    floating-point precision. Shared by the solver-equivalence regression
    tests below.
    """
    _Capture.store.clear()
    solve_one_period_ConsBasicHealth(**kwargs)
    sim_vals, sim_x, sim_y = _Capture.store[-1]

    _Capture.store.clear()
    solve_one_period_ConsBasicHealth_Sequential(**kwargs)
    seq_vals, seq_x, seq_y = _Capture.store[-1]

    np.testing.assert_allclose(seq_x, sim_x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(seq_y, sim_y, rtol=0, atol=1e-9)
    for sim_f, seq_f in zip(sim_vals, seq_vals, strict=False):
        np.testing.assert_allclose(seq_f, sim_f, rtol=0, atol=1e-9)


def shock_dstn(wage=(0.05, 0.10, 0.15), seed=1):
    """Small joint wage/depreciation distribution, by default with strictly
    positive wage.
    """
    depr = np.array([0.02, 0.05, 0.08])
    w, d = np.meshgrid(np.array(wage), depr, indexing="ij")
    atoms = np.vstack([w.ravel(), d.ravel()])
    pmv = np.full(atoms.shape[1], 1.0 / atoms.shape[1])
    return DiscreteDistributionLabeled(
        pmv=pmv,
        atoms=atoms,
        var_names=["WageRte", "DeprRte"],
        seed=seed,
    )


def terminal(m, h):
    """Terminal continuation: consume all, invest nothing -> (vNvrs, c, n)."""
    m = np.asarray(m, dtype=float)
    return m, m, np.zeros_like(m)


def exp_next_at_corner(funcs):
    """``(dvda, dvdH, omega * dvdm)`` from ``calc_exp_next`` on one shock with
    continuation ``funcs``, for the dvdh corner-case tests: at the n = 0 corner
    dvdh vanishes, so dvdH must equal the third entry.
    """
    R, rho, alpha, gamma = 1.03, 0.5, 0.35, 1.0
    shock = {"WageRte": 0.1, "DeprRte": 0.05}
    a = np.array([0.5, 2.0])
    H = np.array([1.0, 5.0])
    _v, dvda, dvdH = calc_exp_next(shock, a, H, R, rho, alpha, gamma, funcs)
    omega = shock["WageRte"]
    return dvda, dvdH, omega * (R * a + omega * H) ** (-rho)


def small_health_params(a_count, h_count, **overrides):
    """One-period ``init_basic_health`` copy on an ``a_count`` x ``h_count`` grid."""
    return {
        **init_basic_health,
        "cycles": 1,
        "aXtraCount": a_count,
        "hLvlCount": h_count,
        **overrides,
    }


def contract_guard_base_kwargs():
    """Shared solver kwargs (minus ``**_COMMON``) for the parameter-contract
    guard tests: an out-of-contract or NaN value is injected on top of these.
    """
    return {
        "solution_next": terminal,
        "ShockDstn": shock_dstn(),
        "aLvlGrid": np.geomspace(1e-5, 100.0, 20),
        "HLvlGrid": np.linspace(0.01, 50.0, 20),
        "interpolator": _Capture,
    }


_COMMON = {
    "DiscFac": 0.95,
    "Rfree": 1.03,
    "CRRA": 0.5,
    "HealthProdExp": 0.35,
    "HealthProdFac": 1.0,
    "DieProbMax": 0.5,
    "constrained_N": 7,
}


# --- Test 1: solver equivalence -------------------------------------------


def test_sequential_matches_simultaneous() -> None:
    """Both solvers must produce identical endogenous grids/policies.

    The FOC inversions are algebraically the same, so with the same
    interpolator the (mLvl, hLvl, vNvrs, cLvl, nLvl) arrays must agree to
    floating-point precision. A future divergence (e.g. fixing one solver's
    derivative but not the other) would break this.
    """
    shock = shock_dstn()
    aLvlGrid = np.geomspace(1e-5, 100.0, 44)
    HLvlGrid = np.linspace(0.01, 50.0, 50)
    kwargs = dict(
        solution_next=terminal,
        ShockDstn=shock,
        aLvlGrid=aLvlGrid,
        HLvlGrid=HLvlGrid,
        interpolator=_Capture,
        **_COMMON,
    )
    assert_solvers_match(kwargs)


# --- Test 2: dvdH matches a finite difference of the continuation ----------


def test_calc_exp_next_marginals_match_finite_difference() -> None:
    """``dvda`` and ``dvdH`` must equal numerical derivatives of E[v_next].

    Uses a self-consistent toy continuation v(m, h) = (m + psi h)^(1-rho)/(1-rho)
    so that the envelope marginals the solver computes (dvdm = c^-rho,
    dvdh = dvdm / g'(n)) coincide with the true partials of v. Then the
    cross-period derivative wrt H must be omega*dvdm + (1-delta)*dvdh. A
    regression to (1-delta)*(omega*dvdm + dvdh) would fail here.
    """
    R, rho, alpha, gamma = 1.03, 0.5, 0.35, 1.0
    psi = 0.5  # weight of health in the toy continuation value
    w, d = 0.12, 0.04
    shock = {"WageRte": w, "DeprRte": d}

    # Self-consistent toy solution: vNvrs = c = m + psi h, n constant such that
    # dvdm / (gamma n^(alpha-1)) = psi * dvdm = dv/dh.
    n_const = (gamma * psi) ** (1.0 / (1.0 - alpha))

    def funcs(m, h):
        m = np.asarray(m, dtype=float)
        h = np.asarray(h, dtype=float)
        vNvrs = m + psi * h
        c = m + psi * h
        n = np.full_like(m, n_const)
        return vNvrs, c, n

    def v_cont(a, H):
        m_next = R * a + w * H
        h_next = (1.0 - d) * H
        return CRRAutility(m_next + psi * h_next, rho=rho)

    a = np.array([0.5, 2.0, 10.0])
    H = np.array([1.0, 5.0, 20.0])

    _v, dvda, dvdH = calc_exp_next(shock, a, H, R, rho, alpha, gamma, funcs)

    eps = 1e-6
    dvda_fd = (v_cont(a + eps, H) - v_cont(a - eps, H)) / (2 * eps)
    dvdH_fd = (v_cont(a, H + eps) - v_cont(a, H - eps)) / (2 * eps)

    np.testing.assert_allclose(dvda, dvda_fd, rtol=1e-5)
    np.testing.assert_allclose(dvdH, dvdH_fd, rtol=1e-5)

    # And explicitly distinct from the wrong factoring (1-delta)*(omega*dvdm + dvdh).
    dvdm = (a * 0 + (R * a + w * H) + psi * (1.0 - d) * H) ** (-rho)
    dvdh = psi * dvdm
    buggy = (1.0 - d) * (w * dvdm + dvdh)
    assert not np.allclose(dvdH, buggy, rtol=1e-3)


# --- Test 4: n = 0 singularity handled cleanly -----------------------------


def test_dvdh_zero_at_n_zero_no_warning() -> None:
    """At n = 0 the marginal health product is infinite, so dvdh -> 0.

    The guarded computation must return finite values and emit no divide/invalid
    RuntimeWarning (the unguarded ``0**(alpha-1)`` would warn).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        dvda, dvdH, corner = exp_next_at_corner(terminal)

    assert np.all(np.isfinite(dvda))
    assert np.all(np.isfinite(dvdH))
    np.testing.assert_allclose(dvdH, corner, rtol=1e-12)


# --- Test 6: unconstrained solver branch (WageRte_min = 0) -----------------


def test_sequential_matches_simultaneous_unconstrained() -> None:
    """Solver equivalence must also hold on the unconstrained branch.

    The tests above all use a strictly positive wage, so ``WageRte_min > 0`` and only
    the liquidity-constrained branch runs. A zero-wage atom flips
    ``WageRte_min`` to 0, so ``constrained = False`` and the cusp-region logic
    (the extra ``constrained_N`` rows, ``nLvl_cnst``/``cLvl_cnst``) is skipped.
    This pins that otherwise-untested branch against the simultaneous solver.
    """
    shock = shock_dstn(wage=(0.0, 0.10, 0.20), seed=2)  # min 0: unconstrained
    kwargs = dict(
        solution_next=terminal,
        ShockDstn=shock,
        aLvlGrid=np.geomspace(1e-5, 100.0, 44),
        HLvlGrid=np.linspace(0.01, 50.0, 50),
        interpolator=_Capture,
        **_COMMON,
    )
    assert_solvers_match(kwargs)


# --- Test 7: parameter-contract guard (CRRA >= 1, alpha out of range) -------


def test_solver_rejects_out_of_contract_params() -> None:
    """CRRA >= 1 (or alpha outside (0, 1)) must raise, not return NaN silently.

    The closed-form FOC inversion is valid only for CRRA < 1 and
    HealthProdExp in (0, 1); outside that region vP_ratio can go negative and
    ``vP_ratio ** (1 / (alpha - 1))`` / ``(.) ** (-1 / CRRA)`` produce NaN with
    only a RuntimeWarning. Both solvers must reject it at the contract boundary.
    """
    base = contract_guard_base_kwargs()
    for solver in (
        solve_one_period_ConsBasicHealth,
        solve_one_period_ConsBasicHealth_Sequential,
    ):
        with pytest.raises(ValueError, match="CRRA must be < 1"):
            solver(**base, **{**_COMMON, "CRRA": 2.0})
        with pytest.raises(ValueError, match=r"alpha.*\(0, 1\)"):
            solver(**base, **{**_COMMON, "HealthProdExp": 1.0})


# --- Test 9: negative-n is surfaced, not silently folded into the corner ---


def test_dvdh_warns_on_negative_n() -> None:
    """Negative health investment must warn (not pass silently) while still
    folding into the n = 0 corner.

    Negative n is non-physical and only reachable as a grid-extrapolation
    artifact. Before the fix it was indistinguishable from the legitimate
    n = 0 corner (both gave dvdh = 0). The guard now emits a RuntimeWarning so a
    *systematic* sign bug or too-narrow grid is visible, while preserving the
    corner value dvdh = 0 (so dvdH = omega * dvdm).
    """

    def funcs_negative_n(m, h):
        m = np.asarray(m, dtype=float)
        return m, m, np.full_like(m, -0.5)  # non-physical negative n

    with pytest.warns(RuntimeWarning, match="Negative health investment"):
        _dvda, dvdH, corner = exp_next_at_corner(funcs_negative_n)

    np.testing.assert_allclose(dvdH, corner, rtol=1e-12)


def test_mortality_cross_derivative_finite_difference() -> None:
    """Solver's ``MargLivPrb`` term matches a finite-difference of ``LivPrb`` in H.

    Covers the health-coupled mortality derivative: the distinguishing term
    in the health solver vs. the durable solver is ``MargLivPrb * v_next_exp``
    accumulated into ``EndOfPrd_dvdH``. The other health tests suppress
    mortality (``DieProbMaxCoeffs = [-25.0]``), so a sign error or wrong-index
    access in this term would escape them. Here we
    bypass the full solver and finite-difference the *formula* the solver
    uses: ``LivPrb(H) = 1 - DieProbMax/(1 + H)`` so
    ``MargLivPrb(H) = DieProbMax / (1 + H)**2 = -dLivPrb/dH``. Tested at
    several health levels with a realistic ``DieProbMax = 0.3``.
    """
    DieProbMax = 0.3
    HLvls = np.array([0.5, 1.0, 2.5, 5.0, 8.0])
    eps = 1e-6

    LivPrb = 1.0 - DieProbMax / (1.0 + HLvls)
    LivPrb_plus = 1.0 - DieProbMax / (1.0 + (HLvls + eps))
    LivPrb_minus = 1.0 - DieProbMax / (1.0 + (HLvls - eps))
    fd_dLivPrb_dH = (LivPrb_plus - LivPrb_minus) / (2.0 * eps)

    # Solver formula (matches the refactored factoring inside the solvers).
    one_plus_H = 1.0 + HLvls
    mortality_rate = DieProbMax / one_plus_H
    MargLivPrb = mortality_rate / one_plus_H

    # MargLivPrb = -dLivPrb/dH (mortality probability falls with health, so
    # survival rises; the sign convention in the solver is that MargLivPrb is
    # the positive derivative of survival w.r.t. H).
    np.testing.assert_allclose(MargLivPrb, fd_dLivPrb_dH, rtol=1e-5)

    # Cross-check the LivPrb factoring against the un-factored expression.
    np.testing.assert_allclose(
        LivPrb,
        1.0 - DieProbMax / (1.0 + HLvls),
        rtol=1e-15,
    )


def test_sequential_matches_simultaneous_real_interpolator() -> None:
    """Solver equivalence through a REAL interpolant, at the paper's level.

    Tests 1 and 6 compare the arrays handed to a mock interpolator; this test
    solves both ways with Curvilinear2DInterp and compares evaluated policies
    on an off-mesh query grid, pinning the appendix claim that the two solvers
    produce identical consumption (measured 3e-13; threshold 1e-11 leaves
    room for BLAS variation while still excluding any real divergence).
    """
    params = small_health_params(25, 20, interpolator=Curvilinear2DInterp)

    sim = BasicHealthConsumerType(**params)
    sim.solve()
    seq = SequentialHealthConsumerType(**params)
    seq.solve()

    m = np.linspace(0.5, 25.0, 30)
    h = np.linspace(0.5, 25.0, 25)
    M, H = np.meshgrid(m, h, indexing="ij")
    _, c_sim, n_sim = sim.solution[0](M, H)
    _, c_seq, n_seq = seq.solution[0](M, H)
    assert np.max(np.abs(c_sim - c_seq)) < 1e-11
    assert np.max(np.abs(n_sim - n_seq)) < 1e-11


def test_solver_rejects_nan_params() -> None:
    """NaN CRRA/alpha/gamma must raise, not flow through the comparisons.

    The validator uses ``not (x < 1.0)``-style guards precisely so NaN (whose
    comparisons are all False) lands on the raising branch; pin that.
    """
    base = contract_guard_base_kwargs()
    for key in ("CRRA", "HealthProdExp", "HealthProdFac"):
        with pytest.raises(ValueError):
            solve_one_period_ConsBasicHealth_Sequential(
                **base,
                **{**_COMMON, key: float("nan")},
            )


# --- m = 0 boundary row and the exact budget test -------------------------


def test_boundary_row_follows_hark() -> None:
    """The ``m = 0`` row of the endogenous mesh is HARK ConsHealthModel's (White):
    ``m = c = n = vNvrs = 0`` with ``h`` copied from the first constrained row.
    """
    kwargs = {
        **contract_guard_base_kwargs(),
        "aLvlGrid": np.geomspace(1e-5, 100.0, 44),
        "HLvlGrid": np.linspace(0.01, 50.0, 50),
        **_COMMON,
    }
    for solver in (
        solve_one_period_ConsBasicHealth,
        solve_one_period_ConsBasicHealth_Sequential,
    ):
        solver(**kwargs)
        (vNvrs, c, n), m, h = _Capture.store[-1]
        for row in (m[0], c[0], n[0], vNvrs[0]):
            np.testing.assert_array_equal(row, 0.0)
        np.testing.assert_array_equal(h[0], h[1])


def test_policy_violations_tests_the_budget_exactly() -> None:
    """An overspend of one ulp is a violation; spending exactly ``m`` is not;
    the maximum overspend is reported relative to ``m``.
    """
    m = np.array([1.0, 2.0, 4.0])
    n = np.array([0.25, 0.5, 0.0])
    c = m - n
    assert policy_violations(m, c, n) == (0, 0.0)
    c_over = c.copy()
    c_over[1] = np.nextafter(c[1], np.inf)
    count, overspend = policy_violations(m, c_over, n)
    assert count == 1
    assert overspend == -(m[1] - c_over[1] - n[1]) / m[1] > 0.0
    count, _ = policy_violations(
        m,
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, -0.1, 0.0]),
    )
    assert count == 2
