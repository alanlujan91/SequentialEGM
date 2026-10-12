"""Health investment model with sequential EGM solver and multiple interpolators.

This module extends HARK's ConsHealthModel with:

1. **Two solvers:**
   - `solve_one_period_ConsBasicHealth`: Simultaneous solver (from HARK)
   - `solve_one_period_ConsBasicHealth_Sequential`: Sequential EGM^n solver

2. **Three interpolators** (all with identical API):
   - `Curvilinear2DInterp`: HARK's sector-walking implementation
   - `Delaunay2DInterp`: Scipy's LinearNDInterpolator (triangulation)
   - `Engine2DInterp`: ENGINE (ENdogenous Grid INterpolation and Extrapolation)

Both solvers accept an `interpolator` parameter to select the interpolation method.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from HARK.Calibration.Income.IncomeProcesses import construct_lognormal_wage_dstn
from HARK.ConsumptionSaving.ConsHealthModel import (
    combine_indep_wage_and_depr_dstns,
    eval_health_prod,
    eval_marg_health_prod,
    make_logistic_polynomial_die_prob,
    make_uniform_depreciation_dstn,
)
from HARK.core import AgentType
from HARK.distributions import expected
from HARK.interpolation import Curvilinear2DInterp
from HARK.metric import MetricObject
from HARK.rewards import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutilityP,
    CRRAutilityP_inv,
)
from HARK.utilities import NullFunc, make_assets_grid

if TYPE_CHECKING:
    from collections.abc import Callable

###############################################################################
# Stage solution classes for Sequential EGM
###############################################################################


@dataclass
class InterpStage(MetricObject):
    """One stage of the sequential solution: an interpolator over the stage's grid
    returning the stage's outputs (listed per field of ``HealthSequentialSolution``).
    """

    distance_criteria: ClassVar[list[str]] = ["interp"]
    interp: Callable[..., Any] = field(default_factory=NullFunc)


@dataclass
class HealthSequentialSolution(MetricObject):
    """Complete solution for one period of the health model using Sequential EGM.

    Contains all intermediate stage solutions. Callable: (m, h) -> (vNvrs, c, n).

    Backward induction only consults ``health_investment`` (via ``__call__``);
    the ``post_decision`` and ``consumption`` stages are retained as diagnostic
    by-products and are not read during the recursion.
    """

    distance_criteria: ClassVar[list[str]] = ["health_investment"]
    # (a, H) -> (w, w_a, w_H)
    post_decision: InterpStage = field(default_factory=InterpStage)
    # (a, H) -> (c, l, w_a, w_H)
    consumption: InterpStage = field(default_factory=InterpStage)
    # (m, h) -> (vNvrs, c, n)
    health_investment: InterpStage = field(default_factory=InterpStage)

    def __call__(self, m, h):
        """Evaluate solution at (m, h) -> (vNvrs, c, n)."""
        return self.health_investment.interp(m, h)


###############################################################################
# Utility functions
###############################################################################


class TerminalInterp:
    """Constant-function "interpolator" for the terminal period: wraps one-argument
    functions ``[f_0, f_1, ...]`` so ``interp(x, y)`` returns ``(f_0(x), f_1(x),
    ...)``, ignoring ``y`` as the stage interpolators' ``(m, h)`` signature allows.
    """

    def __init__(self, output_funcs) -> None:
        self.output_funcs = output_funcs

    def __call__(self, x, y):
        x = np.asarray(x)
        return tuple(f(x) for f in self.output_funcs)


def dvdh_from_envelope(dvdm_next, n_next, gamma, alpha):
    """Envelope marginal ``dv/dh = u'(c) / g'(n)``, exactly 0 at the ``n = 0`` corner.

    With ``g'(n) = gamma * n**(alpha - 1)`` and ``alpha < 1`` the marginal product
    diverges as ``n -> 0``, so the envelope derivative goes to zero there; the
    safe-where pattern returns that limit without ``inf`` arithmetic. Negative
    ``n`` (only a grid-extrapolation artifact) folds into the corner with a
    warning.
    """
    if np.any(np.asarray(n_next) < 0.0):
        warnings.warn(
            "Negative health investment n encountered (grid-extrapolation "
            "artifact); folding into the health investment n = 0 corner. Consider "
            "widening the grid if this recurs.",
            RuntimeWarning,
            stacklevel=2,
        )
    safe_n = np.where(n_next > 0.0, n_next, 1.0)
    return np.where(
        n_next > 0.0,
        dvdm_next / eval_marg_health_prod(safe_n, alpha, gamma),
        0.0,
    )


def validate_health_contract(CRRA, HealthProdExp, HealthProdFac=1.0) -> None:
    """Enforce the solver's documented parameter contract, raising on a violation.

    The closed-form FOC inversions assume a CRRA coefficient strictly below 1
    and a health-production exponent ``alpha`` in ``(0, 1)``. Outside that
    region the marginal-value ratio ``vP_ratio`` can turn negative, so
    ``vP_ratio ** (1 / (alpha - 1))`` (and ``EndOfPrd_dvda ** (-1 / CRRA)``)
    return NaN; numpy emits only an "invalid value encountered in power"
    ``RuntimeWarning`` and the solver hands back a quietly wrong solution.
    Raising here turns that silent NaN into an error naming the parameter, matching the
    "CRRA must be < 1" restriction already stated in both solver docstrings.
    """
    if not CRRA < 1.0:
        msg = (
            f"CRRA must be < 1 for this health model (got {CRRA}); CRRA >= 1 "
            "drives the marginal-value ratio negative and the FOC inversion "
            "returns NaN silently."
        )
        raise ValueError(
            msg,
        )
    if not 0.0 < HealthProdExp < 1.0:
        msg = (
            f"HealthProdExp (alpha) must lie in (0, 1) (got {HealthProdExp}); "
            "the health production g(n) = (gamma / alpha) n**alpha needs "
            "0 < alpha < 1 for a concave, invertible first-order condition."
        )
        raise ValueError(
            msg,
        )
    if not HealthProdFac > 0.0:
        msg = (
            f"HealthProdFac (gamma) must be positive and finite "
            f"(got {HealthProdFac}); gamma <= 0 or NaN makes the marginal "
            "product non-invertible."
        )
        raise ValueError(
            msg,
        )


def inv_marg_health_prod(gP, alpha, gamma):
    """Investment ``n`` with ``eval_marg_health_prod(n) = gP``, i.e. ``g'(n) = gP``."""
    return (gP / gamma) ** (1.0 / (alpha - 1.0))


def warn_bad_vp_ratio(vP_ratio, solver_name) -> None:
    """Surface non-finite or non-positive FOC ratios loudly in BOTH solvers.

    ``vP_ratio = dvda / dvdH`` must be positive and finite for the closed-form
    inversion ``n = (vP_ratio / gamma)**(1/(alpha-1))``; zero/negative or
    non-finite values produce NaN policies with only a generic numpy warning.
    In-contract (rho < 1, alpha in (0,1)) the ratio is positive by
    construction, so any hit here is a grid or calibration problem.
    """
    bad = ~(np.isfinite(vP_ratio) & (vP_ratio > 0.0))
    if np.any(bad):
        warnings.warn(
            f"{solver_name}: non-finite or non-positive vP_ratio at "
            f"{int(bad.sum())} grid nodes; the FOC inversion returns NaN "
            "there. Check DieProbMax and grid extent at small H.",
            RuntimeWarning,
            stacklevel=3,
        )


def policy_violations(m, c, n):
    """``(count, max_overspend)`` of the policy ``(c, n)`` at market resources ``m``.

    ``count`` tallies the infeasible outputs: ``c <= 0``, ``n < 0``, a
    non-finite ``c`` or ``n``, and an overspent budget ``a = m - c - n < 0``
    (``n`` projected onto ``n >= 0``), tested exactly. ``max_overspend`` is
    the largest overspend relative to market resources, ``max(-a / m, 0)``, over
    the finite entries.
    """
    a = m - c - np.maximum(n, 0.0)
    count = np.sum((c <= 0.0) | (n < 0.0) | ~np.isfinite(c + n)) + np.sum(a < 0.0)
    overspend = np.where(np.isfinite(a), -a / m, 0.0)
    return int(count), float(max(np.max(overspend), 0.0))


def value_nvrs(c, w, rho):
    """Inverse-utility value ``u^{-1}(u(c) + w)`` of consuming ``c`` against the
    post-decision value ``w``.
    """
    return CRRAutility_inv(CRRAutility(c, rho=rho) + w, rho=rho)


def post_decision_mesh(aLvlGrid, HLvlGrid, ShockDstn):
    """``(constrained, aLvl, HLvl)``: the ``(a, H)`` mesh, with ``a = 0`` prepended
    when every wage draw is positive (the agent can then end at the constraint).
    """
    constrained = np.min(ShockDstn.atoms[0]) > 0.0
    a_grid = np.insert(aLvlGrid, 0, 0.0) if constrained else aLvlGrid
    return constrained, *np.meshgrid(a_grid, HLvlGrid, indexing="ij")


def continuation_terms(solution_next, ShockDstn, aLvl, HLvl, params, DieProbMax):
    """``(v, dv/da, dv/dH, LivPrb, dLivPrb/dH)`` on the ``(a, H)`` mesh: next
    period's value and marginals integrated over ``ShockDstn`` by
    ``calc_exp_next``, with ``params = (Rfree, CRRA, alpha, gamma)``, and the
    survival probability ``1 - D / (1 + H)`` with its derivative in ``H``.
    """
    v, dvda, dvdH = expected(
        calc_exp_next,
        ShockDstn,
        args=(aLvl, HLvl, *params, solution_next),
    )
    # Factor (1 + H) once: it is also the denominator of the derivative.
    one_plus_H = 1.0 + HLvl
    mortality_rate = DieProbMax / one_plus_H
    return v, dvda, dvdH, 1.0 - mortality_rate, mortality_rate / one_plus_H


def post_decision_marginals(
    solution_next,
    ShockDstn,
    aLvl,
    HLvl,
    params,
    DieProbMax,
    DiscFac,
):
    """``(w, w_a, w_H)``: the sequential solver's post-decision value
    ``w = beta S(H) E[v]`` and its marginals on the ``(a, H)`` mesh, from
    ``continuation_terms``. The simultaneous solver keeps HARK's own grouping
    ``DiscFac * (LivPrb * x)``, which rounds differently.
    """
    v, dvda, dvdH, Surv, MargSurv = continuation_terms(
        solution_next,
        ShockDstn,
        aLvl,
        HLvl,
        params,
        DieProbMax,
    )
    w = DiscFac * Surv * v
    w_a = DiscFac * Surv * dvda
    w_H = DiscFac * (Surv * dvdH + MargSurv * v)
    return w, w_a, w_H


def invert_sequential_stages(w_a, w_H, aMat, HMat, rho, alpha, gamma):
    """``(c, l, n, m, h)`` on the ``(a, H)`` mesh from the post-decision
    marginals: consumption EGM ``c = w_a^{-1/rho}``, ``l = a + c``, then the
    health-investment FOC ``g'(n) = w_a / w_H`` and the pre-decision states.
    """
    cLvl = CRRAutilityP_inv(w_a, rho)
    lLvl = aMat + cLvl
    vP_ratio = w_a / w_H
    warn_bad_vp_ratio(vP_ratio, "Sequential health solver")
    nLvl = inv_marg_health_prod(vP_ratio, alpha, gamma)
    mLvl = lLvl + nLvl
    hLvl = HMat - eval_health_prod(nLvl, alpha, gamma)
    return cLvl, lLvl, nLvl, mLvl, hLvl


def stack_rows(top, arrays):
    """Each of ``top`` stacked above its counterpart in ``arrays``."""
    return tuple(np.concatenate([t, x]) for t, x in zip(top, arrays, strict=True))


def endogenous_mesh(arrays, constrained, w_H_row, w_row, HLvlGrid, N, params):
    """The final ``(m, h, c, n, vNvrs)`` mesh both solvers interpolate.

    The interior ``arrays`` are preceded, when ``constrained``, by ``N``
    liquidity-constrained rows at ``a = 0`` (investment on fractions of the
    cusp value ``n[0, :]``, consumption from the intratemporal FOC
    ``u'(c) = g'(n) w_H`` with ``w_H_row = w_H[0, :]``, value from
    ``w_row = w[0, :]``) and then by the ``m = 0`` boundary row;
    ``params = (rho, alpha, gamma)``.
    """
    rho, alpha, gamma = params
    if constrained:
        nLvl = np.linspace(0.01, 0.99, num=N)[:, None] * arrays[3][0, :]
        marg_health = eval_marg_health_prod(nLvl, alpha, gamma)
        cLvl = CRRAutilityP_inv(w_H_row * marg_health, rho)
        hLvl = HLvlGrid - eval_health_prod(nLvl, alpha, gamma)
        rows = (cLvl + nLvl, hLvl, cLvl, nLvl, value_nvrs(cLvl, w_row, rho))
        arrays = stack_rows(rows, arrays)
    # White's (HARK ConsHealthModel) m = 0 row: h from the first row, vNvrs 0.
    zeros = np.zeros((1, HLvlGrid.size))
    return stack_rows((zeros, arrays[1][:1], zeros, zeros, zeros), arrays)


def calc_exp_next(shock, a, H, R, rho, alpha, gamma, funcs):
    """Compute expectations over next period's (marginal) value.

    Adapted from HARK ConsHealthModel. Labor income accrues on post-investment
    health ``H`` (pre-depreciation), matching the paper's law of motion
    ``y_{t+1} = omega_{t+1} * H_t``, ``h_{t+1} = (1 - delta_{t+1}) * H_t``, and
    ``m_{t+1} = R * a_t + y_{t+1}``. Because only the health-transition term
    carries the depreciation factor, the cross-period derivative ``dvdH`` must
    apply ``(1 - DeprRte)`` to the ``dvdh`` term only, not to the wage-income
    term. (HARK's upstream copy factors ``(1 - DeprRte)`` out of both terms,
    which is inconsistent with income on pre-depreciation ``H``.)
    """
    m_next = R * a + shock["WageRte"] * H
    h_next = (1.0 - shock["DeprRte"]) * H
    vNvrs_next, c_next, n_next = funcs(m_next, h_next)
    dvdm_next = CRRAutilityP(c_next, rho)
    dvdh_next = dvdh_from_envelope(dvdm_next, n_next, gamma, alpha)
    v_next = CRRAutility(vNvrs_next, rho=rho)
    dvda = R * dvdm_next
    # d(m_next)/dH = omega (no depreciation factor); d(h_next)/dH = 1 - delta.
    dvdH = shock["WageRte"] * dvdm_next + (1.0 - shock["DeprRte"]) * dvdh_next
    return v_next, dvda, dvdH


###############################################################################
# SOLVER 1: Simultaneous (from HARK ConsHealthModel, with interpolator param)
###############################################################################


def solve_one_period_ConsBasicHealth(
    solution_next,
    DiscFac,
    Rfree,
    CRRA,
    HealthProdExp,
    HealthProdFac,
    DieProbMax,
    ShockDstn,
    aLvlGrid,
    HLvlGrid,
    constrained_N,
    interpolator=Curvilinear2DInterp,
):
    """Solve one period of the health investment model (simultaneous).

    This is HARK's original solver with an added `interpolator` parameter.

    Parameters
    ----------
    solution_next : callable
        Solution to next period's problem
    DiscFac : float
        Discount factor
    Rfree : float
        Risk-free return
    CRRA : float
        Coefficient of relative risk aversion (must be < 1)
    HealthProdExp : float
        Health production exponent alpha
    HealthProdFac : float
        Health production factor gamma
    DieProbMax : float
        Maximum death probability when H=0
    ShockDstn : DiscreteDistribution
        Joint distribution of shocks
    aLvlGrid : np.array
        End-of-period assets grid
    HLvlGrid : np.array
        Post-investment health grid
    constrained_N : int
        Number of nodes for liquidity-constrained region
    interpolator : class
        Interpolation class (Curvilinear2DInterp, Delaunay2DInterp, or Engine2DInterp)

    Returns
    -------
    solution : interpolator instance
        Solution as callable: (m, h) -> (vNvrs, c, n)

    Notes
    -----
    ``calc_exp_next`` returns ``(v, dvda, dvdH)``; ``expected()`` integrates
    each over ``ShockDstn`` into ``v_next_exp``, ``dvdm_next_exp``, and
    ``dvdh_next_exp`` below (the HARK-inherited ``dvdm``/``dvdh`` names refer
    to the same quantities as ``dvda``/``dvdH`` in the return). These feed the
    end-of-period values asymmetrically: ``dvda`` maps cleanly
    (``EndOfPrd_dvda = DiscFac * LivPrb * dvdm_next_exp``), but ``dvdH`` picks
    up an extra ``MargLivPrb * v_next_exp`` term, because survival probability
    depends on health H, not on assets a:
    ``d/dH[LivPrb * v] = LivPrb * dvdH + MargLivPrb * v``.

    """
    validate_health_contract(CRRA, HealthProdExp, HealthProdFac)
    constrained, aLvl, HLvl = post_decision_mesh(aLvlGrid, HLvlGrid, ShockDstn)

    # Expected continuation values; see the Notes section above for the
    # asymmetric dvda/dvdH envelope this feeds.
    params = (Rfree, CRRA, HealthProdExp, HealthProdFac)
    v_next_exp, dvdm_next_exp, dvdh_next_exp, LivPrb, MargLivPrb = continuation_terms(
        solution_next,
        ShockDstn,
        aLvl,
        HLvl,
        params,
        DieProbMax,
    )

    # End-of-period expectations
    EndOfPrd_v = DiscFac * (LivPrb * v_next_exp)
    EndOfPrd_dvda = DiscFac * (LivPrb * dvdm_next_exp)
    EndOfPrd_dvdH = DiscFac * (LivPrb * dvdh_next_exp + MargLivPrb * v_next_exp)
    vP_ratio = EndOfPrd_dvda / EndOfPrd_dvdH

    warn_bad_vp_ratio(vP_ratio, "Simultaneous health solver")

    # Invert FOCs, then the transitions
    cLvl = CRRAutilityP_inv(EndOfPrd_dvda, CRRA)
    nLvl = inv_marg_health_prod(vP_ratio, HealthProdExp, HealthProdFac)
    mLvl = aLvl + cLvl + nLvl
    hLvl = HLvl - eval_health_prod(nLvl, HealthProdExp, HealthProdFac)
    vNvrs = value_nvrs(cLvl, EndOfPrd_v, CRRA)
    mLvl, hLvl, cLvl, nLvl, vNvrs = endogenous_mesh(
        (mLvl, hLvl, cLvl, nLvl, vNvrs),
        constrained,
        EndOfPrd_dvdH[0, :],
        EndOfPrd_v[0, :],
        HLvlGrid,
        constrained_N,
        (CRRA, HealthProdExp, HealthProdFac),
    )
    return interpolator([vNvrs, cLvl, nLvl], mLvl, hLvl)


###############################################################################
# SOLVER 2: Sequential EGM^n (with interpolator param)
###############################################################################


def solve_one_period_ConsBasicHealth_Sequential(
    solution_next,
    DiscFac,
    Rfree,
    CRRA,
    HealthProdExp,
    HealthProdFac,
    DieProbMax,
    ShockDstn,
    aLvlGrid,
    HLvlGrid,
    constrained_N,
    interpolator=Curvilinear2DInterp,
):
    """Solve one period using Sequential EGM (EGM^n) decomposition.

    Decomposes the problem into three sequential stages:
    1. Post-decision stage: w(a, H) = beta * S(H) * E[v_{t+1}]
    2. Consumption stage: EGM for c given (a, H)
    3. Health investment stage: EGM for n, mapping (l, H) to the
       endogenous (m, h) grid that the returned solution is keyed on

    Parameters are the same as ``solve_one_period_ConsBasicHealth``, except
    ``solution_next`` may also be a ``HealthSequentialSolution``.

    Returns
    -------
    solution : HealthSequentialSolution
        Complete solution with all intermediate stages.
        Callable as: (m, h) -> (vNvrs, c, n)

    """
    alpha = HealthProdExp
    gamma = HealthProdFac
    rho = CRRA
    validate_health_contract(CRRA, HealthProdExp, HealthProdFac)

    # =========================================================================
    # Stage 1: Post-decision value function w(a, H)
    # =========================================================================
    constrained, aMat, HMat = post_decision_mesh(aLvlGrid, HLvlGrid, ShockDstn)
    w, w_a, w_H = post_decision_marginals(
        solution_next,
        ShockDstn,
        aMat,
        HMat,
        (Rfree, rho, alpha, gamma),
        DieProbMax,
        DiscFac,
    )

    # Create post-decision stage (over exogenous (a, H) grid)
    post_decision_stage = InterpStage(interp=interpolator([w, w_a, w_H], aMat, HMat))

    # =========================================================================
    # Stages 2 and 3: consumption EGM, then health-investment EGM
    # =========================================================================
    cLvl, lLvl, nLvl, mLvl, hLvl = invert_sequential_stages(
        w_a,
        w_H,
        aMat,
        HMat,
        rho,
        alpha,
        gamma,
    )
    # Consumption stage over the exogenous (a, H) grid; marginals pass through.
    consumption_stage = InterpStage(
        interp=interpolator([cLvl, lLvl, w_a, w_H], aMat, HMat),
    )

    # Value at decision time
    vNvrs = value_nvrs(cLvl, w, rho)
    mLvl, hLvl, cLvl, nLvl, vNvrs = endogenous_mesh(
        (mLvl, hLvl, cLvl, nLvl, vNvrs),
        constrained,
        w_H[0, :],
        w[0, :],
        HLvlGrid,
        constrained_N,
        (rho, alpha, gamma),
    )

    # Create final stage (over endogenous (m, h) grid)
    health_investment_stage = InterpStage(
        interp=interpolator([vNvrs, cLvl, nLvl], mLvl, hLvl),
    )

    # Return complete solution with all stages
    return HealthSequentialSolution(
        post_decision=post_decision_stage,
        consumption=consumption_stage,
        health_investment=health_investment_stage,
    )


###############################################################################
# Constructors and default parameters (from HARK ConsHealthModel)
###############################################################################


def make_solution_terminal_ConsBasicHealth():
    """Terminal period solution: consume all, invest nothing.

    Returns a simple callable for the simultaneous solver.
    """
    return lambda mLvl, hLvl: (mLvl, mLvl, np.zeros_like(mLvl))


def make_solution_terminal_ConsBasicHealth_Sequential():
    """Terminal period solution for the Sequential EGM solver: consume all wealth
    (c = m), invest nothing (n = 0), so v = u(m).
    """

    def zeros(x):
        return np.zeros_like(x)

    def identity(x):
        return x

    return HealthSequentialSolution(
        post_decision=InterpStage(interp=TerminalInterp([zeros, zeros, zeros])),
        consumption=InterpStage(
            interp=TerminalInterp([identity, identity, zeros, zeros]),
        ),
        health_investment=InterpStage(
            interp=TerminalInterp([identity, identity, zeros]),
        ),
    )


def make_health_grid(hLvlMin, hLvlMax, hLvlCount):
    """Make uniform grid of health capital levels."""
    return np.linspace(hLvlMin, hLvlMax, hLvlCount)


# Default constructors for simultaneous solver
basic_health_constructors = {
    "WageRteDstn": construct_lognormal_wage_dstn,
    "DeprRteDstn": make_uniform_depreciation_dstn,
    "ShockDstn": combine_indep_wage_and_depr_dstns,
    "aLvlGrid": make_assets_grid,
    "HLvlGrid": make_health_grid,
    "DieProbMax": make_logistic_polynomial_die_prob,
    "solution_terminal": make_solution_terminal_ConsBasicHealth,
}

# Default constructors for sequential EGM solver
sequential_health_constructors = {
    **basic_health_constructors,
    "solution_terminal": make_solution_terminal_ConsBasicHealth_Sequential,
}

AXTRA_EXTRA = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


# The paper's health timing configuration: HARK's default (aXtraCount,
# hLvlCount) grid solved over ten cycles.
PAPER_TIMING_GRID = (44, 50)
PAPER_TIMING_CYCLES = 10

# The health summary figure's sweeps through that point: the asset axis scaled
# at its health density (plus one small grid), and up to 15 cycles, where every
# mesh keeps ENGINE's def-ipm precondition (generate_health_model_summary).
SUMMARY_CYCLES = (2, 5, PAPER_TIMING_CYCLES, 15)
SUMMARY_GRIDS = (
    (20, 15),
    *((k * PAPER_TIMING_GRID[0], PAPER_TIMING_GRID[1]) for k in (1, 2, 3, 4)),
)


def paper_timing_params():
    """Grid and horizon overrides for the paper's health timing runs."""
    a_count, h_count = PAPER_TIMING_GRID
    return {"aXtraCount": a_count, "hLvlCount": h_count, "cycles": PAPER_TIMING_CYCLES}


def health_grid_params(a_max, a_count, h_max, h_count, n_extra=6):
    """One-period grid overrides for the health benchmarks: ``aXtra`` up to
    ``a_max`` plus the first ``n_extra`` points of ``AXTRA_EXTRA``, ``hLvl`` from
    ``init_basic_health``'s ``hLvlMin`` to ``h_max``.
    """
    return {
        "aXtraMin": 1e-5,
        "aXtraMax": a_max,
        "aXtraCount": a_count,
        "aXtraNestFac": 1,
        "aXtraExtra": list(AXTRA_EXTRA[:n_extra]),
        "hLvlMin": init_basic_health["hLvlMin"],
        "hLvlMax": h_max,
        "hLvlCount": h_count,
        "cycles": 1,
    }


# Shared default parameters (both solvers use the same economic calibration)
_shared_health_params = {
    "DiscFac": 0.95,
    "Rfree": [1.03],
    "CRRA": 0.5,
    "HealthProdExp": 0.35,
    "HealthProdFac": 1.0,
    "T_cycle": 1,
    "cycles": 1,
    "DeprRteMean": [0.05],
    "DeprRteSpread": [0.05],
    "DeprRteCount": 7,
    "WageRteMean": [0.1],
    "WageRteStd": [0.1],
    "WageRteCount": 7,
    "UnempPrb": 0.07,
    "IncUnemp": 0.3,
    "aXtraMin": 1e-5,
    "aXtraMax": 100.0,
    "aXtraCount": 44,
    "aXtraNestFac": 1,
    "aXtraExtra": list(AXTRA_EXTRA),
    # Departs from HARK ConsHealthModel's 0.0: at H = 0, a = 0 the inversion is w_a/w_H = inf/inf,
    # which ENGINE rejects and curvilinear/Delaunay carry as NaN nodes. The lowest-node
    # preimages (h < 0) are the exact EGM image at either value.
    "hLvlMin": 0.01,
    "hLvlMax": 50.0,
    "hLvlCount": 50,
    "DieProbMaxCoeffs": [0.0],
    "constrained_N": 7,
    "interpolator": Curvilinear2DInterp,
}

# deepcopy prevents shared mutable aliasing of list-valued parameters
# (Rfree, aXtraExtra, etc.) between the two solver configurations.
init_basic_health = {
    "constructors": basic_health_constructors,
    **copy.deepcopy(_shared_health_params),
}
init_sequential_health = {
    "constructors": sequential_health_constructors,
    **copy.deepcopy(_shared_health_params),
}


###############################################################################
# Agent Type (from HARK ConsHealthModel)
###############################################################################


class BasicHealthConsumerType(AgentType):
    """Health investment consumer from HARK with configurable solver/interpolator.

    By default uses the simultaneous solver with Curvilinear2DInterp.
    Set `interpolator` parameter to Delaunay2DInterp or Engine2DInterp for alternatives.
    """

    default_: ClassVar[dict] = {
        "params": init_basic_health,
        "solver": solve_one_period_ConsBasicHealth,
    }
    time_vary_: ClassVar[list[str]] = ["Rfree", "DieProbMax", "ShockDstn"]
    time_inv_: ClassVar[list[str]] = [
        "DiscFac",
        "CRRA",
        "HealthProdExp",
        "HealthProdFac",
        "aLvlGrid",
        "HLvlGrid",
        "constrained_N",
        "interpolator",
    ]
    distributions: ClassVar[list[str]] = ["ShockDstn"]


class SequentialHealthConsumerType(BasicHealthConsumerType):
    """Health consumer using Sequential EGM^n solver.

    Returns ``HealthSequentialSolution`` objects whose stage interpolators are
    ``post_decision`` (``(a, H) -> (w, w_a, w_H)``), ``consumption``
    (``(a, H) -> (c, l, w_a, w_H)``) and ``health_investment``
    (``(m, h) -> (vNvrs, c, n)``). The solution is callable:
    ``solution[t](m, h) -> (vNvrs, c, n)``.
    """

    default_: ClassVar[dict] = {
        "params": init_sequential_health,
        "solver": solve_one_period_ConsBasicHealth_Sequential,
    }
