"""Risky portfolio share appended to the consumption-labor model of Section 2.

The share stage has no closed-form inversion, so it takes the fallback of
Section 5: its first-order condition is evaluated on a grid of shares and the
first nonnegative-to-nonpositive crossing is linearly interpolated, with the
corner shares 0 and 1 where no crossing exists.

``LaborPortfolioSolver`` subclasses ``LaborSeparableSolver`` and overrides only
the end-of-period step, ``calc_EndOfPrdvP``. That step now integrates over the
risky return and chooses the share, and it hands the consumption stage the same
inverse marginal value on the same asset grid as the two-stage model does. The
consumption and labor-leisure stages, the terminal solution and the solution
container are the two-stage model's, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from HARK.Calibration.Assets.AssetProcesses import (
    combine_IncShkDstn_and_RiskyDstn,
    make_lognormal_RiskyDstn,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import init_portfolio
from HARK.ConsumptionSaving.ConsRiskyAssetModel import make_simple_ShareGrid
from HARK.core import make_one_period_oo_solver
from HARK.distributions import DiscreteDistribution
from HARK.interpolation import LinearInterp, MargValueFuncCRRA
from HARK.utilities import NullFunc

from HARKmodels.ConsLaborSeparableModel import (
    LaborSeparableConsumerType,
    LaborSeparableSolver,
    PostDecisionStage,
    calc_expectation,
    init_labor_separable,
    next_marginal_utility,
)


@dataclass
class PortfolioStage(PostDecisionStage):
    """Share stage at end-of-period assets ``a``: the risky share ``share_func``
    and, in the inherited fields, the marginal value at that share, which the
    consumption stage reads exactly as it reads a post-decision stage.
    ``share_foc`` holds the share condition on the ``(a, share)`` grid.
    """

    share_func: LinearInterp = field(default_factory=NullFunc)
    share_foc: np.ndarray = field(default_factory=lambda: np.array([]))


def first_crossing_share(foc, share_grid):
    """Risky share for each row of ``foc``, the share condition on ``share_grid``.

    The share is the linear interpolant's zero at the first nonnegative-to-
    nonpositive crossing along the row. A row with no crossing is a corner:
    share 1 where the condition is positive at share 1, share 0 otherwise.
    """
    crossing = (foc[:, :-1] >= 0.0) & (foc[:, 1:] <= 0.0)
    j = np.argmax(crossing, axis=1)
    rows = np.arange(foc.shape[0])
    lo, hi = foc[rows, j], foc[rows, j + 1]
    gap = lo - hi
    weight = np.divide(lo, gap, out=np.zeros_like(lo), where=gap > 0.0)
    share = share_grid[j] + weight * (share_grid[j + 1] - share_grid[j])
    corner = np.where(foc[:, -1] > 0.0, 1.0, 0.0)
    return np.where(crossing.any(axis=1), share, corner)


@dataclass
class LaborPortfolioSolver(LaborSeparableSolver):
    """``LaborSeparableSolver`` with the risky share chosen after consumption."""

    ShockDstn: DiscreteDistribution
    ShareGrid: np.ndarray

    def end_of_period_partials(self):
        """``(v_a, v_s / a)`` on the ``aXtraGrid x ShareGrid`` grid: the marginal
        value of assets and the share condition, each a discounted expectation
        over permanent, transitory and risky-return draws.
        """
        a_mat, s_mat = np.meshgrid(self.aXtraGrid, self.ShareGrid, indexing="ij")

        def partials(shock, anrm, share):
            excess = shock[2] - self.Rfree
            rport = self.Rfree + excess * share
            p_shk = self.PermGroFac * shock[0]
            bnrm = anrm * rport / p_shk
            tshk = np.broadcast_to(shock[1], bnrm.shape)
            vp_next = p_shk**-self.CRRA * self.vp_func_next(bnrm, tshk)
            return rport * vp_next, excess * vp_next

        expectation = calc_expectation(self.ShockDstn, partials, a_mat, s_mat)
        discount = self.DiscFac * self.LivPrb
        return discount * expectation[0], discount * expectation[1]

    def calc_EndOfPrdvP(self) -> None:
        """Share stage: choose the share on each asset node, then store the
        inverse marginal value at that share on ``aGrid`` (with ``a = 0``) where
        the consumption stage reads it.
        """
        dvda, foc = self.end_of_period_partials()
        share = first_crossing_share(foc, self.ShareGrid)
        nvrs = self.u_func.derinv(dvda)
        nvrs_at_share = np.array(
            [
                np.interp(s, self.ShareGrid, row)
                for s, row in zip(share, nvrs, strict=True)
            ]
        )
        # At a = 0 the inverse marginal value is exactly 0, as in the two-stage
        # model; the share there is undefined and continues the first node's.
        vp = np.append(0.0, nvrs_at_share)
        self.post_decision_stage = PortfolioStage(
            vp_func=MargValueFuncCRRA(LinearInterp(self.aGrid, vp), self.CRRA),
            vp=vp,
            share_func=LinearInterp(self.aGrid, np.append(share[0], share)),
            share_foc=foc,
        )


init_labor_portfolio = init_labor_separable.copy()
for _key in ("RiskyAvg", "RiskyStd", "RiskyCount", "ShareCount"):
    init_labor_portfolio[_key] = init_portfolio[_key]
init_labor_portfolio["constructors"] = {
    **init_labor_separable["constructors"],
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShockDstn": combine_IncShkDstn_and_RiskyDstn,
    "ShareGrid": make_simple_ShareGrid,
}


class LaborPortfolioConsumerType(LaborSeparableConsumerType):
    """The consumption-labor agent with a risky asset and a portfolio share."""

    time_vary_: ClassVar[list] = [*LaborSeparableConsumerType.time_vary_, "ShockDstn"]
    time_inv_: ClassVar[list] = [*LaborSeparableConsumerType.time_inv_, "ShareGrid"]

    default_: ClassVar[dict] = {
        "params": init_labor_portfolio,
        "solver": make_one_period_oo_solver(LaborPortfolioSolver),
    }


def portfolio_residuals(agent, m, t=0):
    """``(euler, share)`` residuals at cash on hand ``m`` in period ``t``, each
    an expectation over the shock draws at the share the solver chose.

    ``euler`` is ``c_euler / c - 1``, where ``c_euler`` inverts
    ``u'(c) = β s E[R^p (Γψ)^{-ρ} u'(c_{t+1})]``. ``share`` is the share
    condition ``E[(R_risky - R) (Γψ)^{-ρ} u'(c_{t+1})]`` relative to
    ``E[R^p (Γψ)^{-ρ} u'(c_{t+1})]``; it is 0 at an interior share and
    keeps the corner's sign at a corner.
    """
    k = t % agent.T_cycle
    sol = agent.solution[t]
    c = sol.consumption_saving.c_func(m)
    a = m - c
    share = sol.post_decision.share_func(a)
    dstn = agent.ShockDstn[k]
    excess = dstn.atoms[2] - agent.Rfree[k]
    rport = agent.Rfree[k] + excess * share[:, None]
    marg = next_marginal_utility(agent, t, a[:, None], rport, dstn.atoms)
    wealth_term = (rport * marg) @ dstn.pmv
    rhs = agent.DiscFac * agent.LivPrb[k] * wealth_term
    return rhs ** (-1.0 / agent.CRRA) / c - 1.0, (
        (excess * marg) @ dstn.pmv
    ) / wealth_term
