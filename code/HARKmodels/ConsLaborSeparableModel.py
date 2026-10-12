from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from engine.utilities import interp_on_interp
from HARK.ConsumptionSaving.ConsLaborModel import (
    LaborIntMargConsumerType,
    init_labor_intensive,
)
from HARK.core import make_one_period_oo_solver
from HARK.distributions import DiscreteDistribution, expected
from HARK.interpolation import (
    LinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA, UtilityFuncStoneGeary
from HARK.utilities import NullFunc

if TYPE_CHECKING:
    from HARK.econforgeinterp import LinearFast


def calc_expectation(dstn, func, *args):
    """Compatibility shim for the renamed HARK API (``calc_expectation`` -> ``expected``).

    Older HARK exposed ``calc_expectation(dstn, func, *args)``; current HARK
    requires keyword arguments ``expected(func=..., dstn=..., args=(...))``.

    Current HARK also returns a ``DiscreteDistributionLabeled`` from
    ``combine_IncShkDstn_and_RiskyDstn`` (used for ``ShockDstn``). When
    ``expected`` dispatches on a labeled distribution it passes the shock draws
    to ``func`` keyed by variable *name* (``shocks["Risky"]``), whereas this
    model indexes shocks by integer position (``shocks[self.RiskyShkIdx]``). To
    keep a single integer-indexed convention for every call site, rebuild the
    distribution as a plain ``DiscreteDistribution`` from its atoms and pmv so
    ``func`` always receives an integer-indexable atom array. This is a no-op
    for distributions that are already plain (e.g. ``IncShkDstn``).
    """
    plain = DiscreteDistribution(pmv=dstn.pmv, atoms=dstn.atoms)
    # Pass func/dstn positionally: the keyword for the distribution has been
    # spelled both ``dstn`` and ``dist`` across HARK versions, but the first two
    # positional parameters are stable.
    return expected(func, plain, args=args)


def labor_and_leisure(x, disutility):
    """``(leisure, labor)`` in ``[0, 1]`` from the inverted labor FOC, whose root
    ``x`` is labor under ``disutility`` and leisure otherwise.
    """
    leisure = np.clip(1.0 - x if disutility else x, 0.0, 1.0)
    return leisure, 1.0 - leisure


def clipped_policies(leisure, labor, grids):
    """Leisure and labor policies over the curvilinear ``grids = [b, theta]``
    mesh, clipped to ``[0, 1]`` at query time.
    """
    leisure_interp = interp_on_interp(leisure, grids)
    labor_interp = interp_on_interp(labor, grids)

    def leisure_func(b, t):
        return np.clip(leisure_interp(b, t), 0.0, 1.0)

    def labor_func(b, t):
        return np.clip(labor_interp(b, t), 0.0, 1.0)

    return leisure_func, labor_func


def labor_utility_funcs(
    CRRA,
    Disutility,
    LaborCRRA,
    LaborFactor,
    LeisureCRRA,
    LeisureFactor,
):
    """Consumption utility and the labor disutility or leisure utility. The
    paper's leisure utility is h(z) = α^{1-ρ} z^{1-ξ}/(1-ξ) with α =
    LeisureFactor, ρ = CRRA, ξ = LeisureCRRA: the factor passed on is α^{1-ρ},
    which keeps leisure utility consistent with balanced growth.
    """
    u = UtilityFuncCRRA(CRRA)
    if Disutility:
        return u, StoneGearyUtility(-LaborCRRA, factor=LaborFactor)
    return u, StoneGearyUtility(LeisureCRRA, factor=LeisureFactor ** (1.0 - CRRA))


@dataclass
class ValueStage(MetricObject):
    """A stage's value and marginal value; convergence is measured on ``vp_func``."""

    distance_criteria: ClassVar[list[str]] = ["vp_func"]
    v_func: ValueFuncCRRA = field(default_factory=NullFunc)
    vp_func: MargValueFuncCRRA = field(default_factory=NullFunc)


@dataclass
class PostDecisionStage(ValueStage):
    """Post-decision stage at liquid-asset state ``a``: value and marginal value."""

    vp: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ConsumptionSavingStage(ValueStage):
    """Consumption-savings stage: policy ``c(m)`` plus value and marginal of ``m``."""

    c_func: LinearFast = field(default_factory=NullFunc)  # policy: consumption
    mNrmMin: float = 0.0


@dataclass
class LaborLeisureStage(ValueStage):
    """Labor / leisure stage: policies in ``(labor, leisure, c)`` plus value of
    ``(b, theta)``, and the endogenous mesh ``grids`` the EGM inversion produced.
    """

    labor_func: LinearFast = field(default_factory=NullFunc)
    leisure_func: LinearFast = field(default_factory=NullFunc)
    c_func: LinearFast = field(default_factory=NullFunc)
    bNrmMin: float = 0.0
    grids: dict = field(default_factory=dict)


@dataclass
class LaborSeparableSolution(MetricObject):
    distance_criteria: ClassVar[list[str]] = [
        "consumption_saving",
        "labor_leisure",
        "post_decision",
    ]
    labor_leisure: LaborLeisureStage = field(default_factory=LaborLeisureStage)
    consumption_saving: ConsumptionSavingStage = field(
        default_factory=ConsumptionSavingStage,
    )
    post_decision: PostDecisionStage = field(default_factory=PostDecisionStage)


class StoneGearyUtility(UtilityFuncStoneGeary):
    """Leisure utility (positive CRRA) or labor disutility (negated CRRA)."""

    def derinv(self, u, order=(1, 0)):
        """Inverse of the marginal utility, ``(u')^{-1}(u)``.

        Current HARK defines ``derinv`` on ``UtilityFuncCRRA`` but not on
        ``UtilityFuncStoneGeary``, where it is the short alias
        ``inverse(u, order=(1, 0))``. The EGM solver uses it to recover the
        labor-leisure choice from the first-order-condition right-hand side,
        so we mirror the CRRA alias exactly.
        """
        return self.inverse(u, order)


def require_zero_wage_state(TranShkGrid) -> None:
    """Raise unless every period's lowest transitory wage shock is exactly 0.

    The solver relies on that zero-wage state (``UnempPrb > 0`` with
    ``IncUnemp = 0``): it makes ``a = 0`` the borrowing limit and ``theta = 0``
    the full-leisure column of the labor-leisure stage. Without it the
    backward recursion returns NaN policies.
    """
    lowest = [float(grid[0]) for grid in TranShkGrid]
    if any(x != 0.0 for x in lowest):
        msg = (
            "LaborSeparableConsumerType needs a zero-wage state in every period "
            f"(UnempPrb > 0 and IncUnemp = 0); the lowest wage shocks are {lowest}."
        )
        raise ValueError(msg)


def labor_leisure_on_rectangle(
    cs_stage,
    mgrid,
    tshkgrid,
    wage,
    n_func,
    disutility,
    CRRA,
):
    """EGM for the labor-leisure stage on the exogenous rectangle
    ``mgrid x tshkgrid`` of cash on hand ``m`` and wage shock ``theta``.

    Inverts the labor FOC against the consumption stage ``cs_stage``, maps
    node ``(m_i, theta_j)`` to the bank balance ``b_ij = m_i - w theta_j
    labor_ij`` and keys the policies on that endogenous mesh, consumption at
    ``b_ij`` being ``c(m_i)``. Two edges are exact limits: ``theta = 0``
    (``tshkgrid[0]``) earns nothing, so leisure is full; ``m = 0``
    (``mgrid[0]``) leaves ``c = 0`` and ``u'(c) = inf``, so labor is full.
    """
    mnrmat, tshkmat = np.meshgrid(mgrid, tshkgrid, indexing="ij")
    x = n_func.derinv(cs_stage.vp_func(mnrmat[1:, 1:]) * wage * tshkmat[1:, 1:])
    if not np.all(np.isfinite(x)):
        msg = (
            f"labor FOC inversion is non-finite at {int(np.sum(~np.isfinite(x)))} "
            "nodes with m > 0 and theta > 0"
        )
        raise FloatingPointError(msg)
    leisure = np.ones(mnrmat.shape)
    leisure[0, 1:] = 0.0
    leisure[1:, 1:] = labor_and_leisure(x, disutility)[0]
    labor = 1.0 - leisure
    bnrmat = mnrmat - tshkmat * wage * labor
    leisure_func, labor_func = clipped_policies(leisure, labor, [bnrmat, tshkmat])
    c_func = interp_on_interp(cs_stage.c_func(mnrmat), [bnrmat, tshkmat])
    return LaborLeisureStage(
        labor_func=labor_func,
        leisure_func=leisure_func,
        c_func=c_func,
        vp_func=MargValueFuncCRRA(c_func, CRRA),
        grids={
            "mNrm": mnrmat,
            "bNrm": bnrmat,
            "tShk": tshkmat,
            "labor": labor,
            "leisure": leisure,
        },
    )


@dataclass
class LaborSolverInputs:
    """Period inputs shared by the labor solvers (HARK passes them by keyword)."""

    TranShkDstn: DiscreteDistribution
    LivPrb: float
    DiscFac: float
    CRRA: float
    Disutility: bool
    LaborFactor: float
    LaborCRRA: float
    LeisureFactor: float
    LeisureCRRA: float
    WageRte: float
    Rfree: float
    PermGroFac: float
    BoroCnstArt: float
    aXtraGrid: np.ndarray
    TranShkGrid: np.ndarray

    def utility_funcs(self):
        """``labor_utility_funcs`` at this period's preference parameters."""
        return labor_utility_funcs(
            self.CRRA,
            self.Disutility,
            self.LaborCRRA,
            self.LaborFactor,
            self.LeisureCRRA,
            self.LeisureFactor,
        )


@dataclass
class LaborSeparableSolver(LaborSolverInputs):
    solution_next: LaborSeparableSolution
    IncShkDstn: DiscreteDistribution
    vFuncBool: bool
    CubicBool: bool

    def __post_init__(self):
        self.u_func, self.n_func = self.utility_funcs()

    def prepare_to_solve(self) -> None:
        self.vp_func_next = self.solution_next.labor_leisure.vp_func
        # The zero-wage state (``require_zero_wage_state``) makes ``a = 0`` the
        # borrowing limit, so the asset grid starts there.
        self.aGrid = np.append(0.0, self.aXtraGrid)

    def calc_EndOfPrdvP(self) -> None:
        def dvda_func(shock, anrm):
            """Evaluate marginal value at one atom, batched over the whole array.

            HARK's modern ``expected`` calls this once with the full atom
            array, not per-atom: ``shock`` has shape ``(2, n_atoms)`` and
            ``anrm`` broadcasts to shape ``(n_a, 1)``, so ``shock[1]`` must be
            broadcast to match ``bnrm``'s 2D shape for the interpolator call.
            """
            p_shk = self.PermGroFac * shock[0]
            bnrm = anrm * self.Rfree / p_shk
            tshk = np.broadcast_to(shock[1], bnrm.shape)
            return p_shk**-self.CRRA * self.vp_func_next(bnrm, tshk)

        EndOfPrdvP_vals = (
            self.DiscFac
            * self.LivPrb
            * self.Rfree
            * calc_expectation(self.IncShkDstn, dvda_func, self.aXtraGrid)
        )
        # At a = 0 the zero-wage draw leaves nothing to consume next period:
        # marginal value is infinite there and its inverse is exactly 0.
        EndOfPrdvP_nvrs = np.append(0.0, self.u_func.derinv(EndOfPrdvP_vals))
        EndOfPrdvP_nvrs_func = LinearInterp(self.aGrid, EndOfPrdvP_nvrs)
        EndOfPrdvP_func = MargValueFuncCRRA(EndOfPrdvP_nvrs_func, self.CRRA)

        self.post_decision_stage = PostDecisionStage(
            vp_func=EndOfPrdvP_func,
            vp=EndOfPrdvP_nvrs,
        )

    def make_consumption_solution(self) -> None:
        # The consumption function consistent with the exogenous asset level.
        # This decision comes after the labor-leisure decision and before the
        # end-of-period value function.
        cGrid = self.post_decision_stage.vp
        mGrid = cGrid + self.aGrid

        cFuncEndOfPrd = LinearInterp(mGrid, cGrid)
        vPfuncEndOfPrd = MargValueFuncCRRA(cFuncEndOfPrd, self.CRRA)

        self.consumption_saving_stage = ConsumptionSavingStage(
            c_func=cFuncEndOfPrd,
            vp_func=vPfuncEndOfPrd,
        )

    def make_labor_leisure_solution(self) -> None:
        self.labor_leisure_stage = labor_leisure_on_rectangle(
            self.consumption_saving_stage,
            self.aGrid,
            self.TranShkGrid,
            self.WageRte,
            self.n_func,
            self.Disutility,
            self.CRRA,
        )

    def make_consumer_labor_solution(self) -> None:
        self.solution = LaborSeparableSolution(
            post_decision=self.post_decision_stage,
            consumption_saving=self.consumption_saving_stage,
            labor_leisure=self.labor_leisure_stage,
        )

    def solve(self):
        self.calc_EndOfPrdvP()

        self.make_consumption_solution()
        self.make_labor_leisure_solution()

        self.make_consumer_labor_solution()

        return self.solution


def make_labor_separable_solution_terminal(
    CRRA,
    aXtraGrid,
    WageRte,
    TranShkGrid,
    Disutility,
    LaborFactor,
    LaborCRRA,
    LeisureFactor,
    LeisureCRRA,
):
    """Terminal-period solution: the consumption stage consumes all cash,
    ``c(m) = m``, and the labor-leisure stage solves the period's labor choice
    against it, as in every earlier period, plus the value function.
    """
    require_zero_wage_state(TranShkGrid)

    def c_func_cs(m):
        return m

    v_func_cs = ValueFuncCRRA(c_func_cs, CRRA)
    cs_stage = ConsumptionSavingStage(
        c_func=c_func_cs,
        v_func=v_func_cs,
        vp_func=MargValueFuncCRRA(c_func_cs, CRRA),
    )
    uFunc, nFunc = labor_utility_funcs(
        CRRA,
        Disutility,
        LaborCRRA,
        LaborFactor,
        LeisureCRRA,
        LeisureFactor,
    )
    ll_stage = labor_leisure_on_rectangle(
        cs_stage,
        np.append(0.0, aXtraGrid),
        TranShkGrid[-1],
        WageRte[-1],
        nFunc,
        Disutility,
        CRRA,
    )

    grids = ll_stage.grids
    # The m = 0 row has c = 0 and zero leisure, where both utilities equal their
    # exact limit -inf; the inverse value maps it to 0.
    with np.errstate(divide="ignore"):
        leisure_value = (
            -nFunc(grids["labor"]) if Disutility else nFunc(grids["leisure"])
        )
        vNvrs = uFunc.inverse(leisure_value + v_func_cs(grids["mNrm"]))
    ll_stage.v_func = ValueFuncCRRA(
        interp_on_interp(vNvrs, [grids["bNrm"], grids["tShk"]]),
        CRRA,
    )
    return LaborSeparableSolution(labor_leisure=ll_stage, consumption_saving=cs_stage)


def next_marginal_utility(agent, t, a, rport, shocks):
    """``(Γψ)^{-ρ} u'(c_{t+1})`` for each shock atom, from end-of-period assets
    ``a`` (a column) earning the gross return ``rport``.

    ``shocks`` holds the permanent and transitory atoms in its first two rows.
    Next period's consumption composes the stages: the labor policy at bank
    balance ``b' = rport a / (Γψ)``, then the consumption stage at the cash on
    hand ``b' + w θ' n`` that labor choice produces.
    """
    k = (t + 1) % agent.T_cycle
    growth = agent.PermGroFac[t % agent.T_cycle] * shocks[0]
    theta = shocks[1]
    b_next = rport * a / growth
    theta = np.broadcast_to(theta, b_next.shape)
    sol = agent.solution[t + 1]
    labor = sol.labor_leisure.labor_func(b_next, theta)
    c_next = sol.consumption_saving.c_func(b_next + agent.WageRte[k] * theta * labor)
    return growth**-agent.CRRA * c_next**-agent.CRRA


def consumption_euler_residual(agent, m, t=0):
    """``c_euler / c - 1`` at cash on hand ``m`` in period ``t``, where ``c_euler``
    inverts ``u'(c) = β s R E[(Γψ)^{-ρ} u'(c_{t+1})]`` with next period's
    consumption from ``next_marginal_utility``.
    """
    k = t % agent.T_cycle
    c = agent.solution[t].consumption_saving.c_func(m)
    dstn = agent.IncShkDstn[k]
    a = (m - c)[:, None]
    marg = next_marginal_utility(agent, t, a, agent.Rfree[k], dstn.atoms)
    rhs = agent.DiscFac * agent.LivPrb[k] * agent.Rfree[k] * (marg @ dstn.pmv)
    return rhs ** (-1.0 / agent.CRRA) / c - 1.0


init_labor_separable = init_labor_intensive.copy()
init_labor_separable["Disutility"] = False
init_labor_separable["LaborFactor"] = 1.0
init_labor_separable["LaborCRRA"] = 2.0
init_labor_separable["LeisureFactor"] = 0.5
init_labor_separable["LeisureCRRA"] = 2.0
init_labor_separable["CRRA"] = 5.0
init_labor_separable["DiscFac"] = 0.9

# Permanent income is switched off: it plays no role in demonstrating sequential
# EGM, and carrying it forces permanent-income normalization into the exposition.
# Transitory wage risk remains and is what the labor-leisure stage responds to.
init_labor_separable["PermGroFac"] = [1.0]
init_labor_separable["PermShkStd"] = [0.0]
init_labor_separable["PermShkCount"] = 1

# UnempPrb > 0 with IncUnemp = 0 supplies the zero-wage state the solver needs;
# `require_zero_wage_state` raises at construction otherwise.

# Override the solution_terminal constructor
init_labor_separable["constructors"] = {
    **init_labor_intensive["constructors"],
    "solution_terminal": make_labor_separable_solution_terminal,
}


# Define the LaborSeparableConsumerType class with proper defaults
class LaborSeparableConsumerType(LaborIntMargConsumerType):
    """Agent type for labor-leisure choice with separable utility.
    Agents choose consumption, savings, and labor supply each period.
    """

    time_inv_ = copy(LaborIntMargConsumerType.time_inv_)
    time_inv_ += [
        "Disutility",  # boolean : if True, disutility of labor
        "LaborFactor",  # float : multiplicative factor on labor disutility
        "LaborCRRA",  # float : CRRA of labor disutility
        "LeisureFactor",  # float : multiplicative factor on leisure utility
        "LeisureCRRA",  # float : CRRA of leisure utility
    ]

    default_: ClassVar[dict] = {
        "params": init_labor_separable,
        "solver": make_one_period_oo_solver(LaborSeparableSolver),
    }
