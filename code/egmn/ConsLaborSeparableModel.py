from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import numpy as np
from HARK.ConsumptionSaving.ConsLaborModel import (
    ConsumerLaborSolution,
    LaborIntMargConsumerType,
    init_labor_intensive,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.core import make_one_period_oo_solver
from HARK.distributions import DiscreteDistribution, calc_expectation
from HARK.econforgeinterp import LinearFast
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA, UtilityFuncStoneGeary
from HARK.utilities import NullFunc

from egmn.utilities import interp_on_interp


@dataclass
class PostDecisionStage(MetricObject):
    """This clas contains the value and marginal value functions of the
    post decision state a : assets or savings or liquid wealth.
    """

    distance_criteria = ["vp_func"]
    v_func: ValueFuncCRRA = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    vp: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ConsumptionSavingStage(MetricObject):
    """This class contains the policy function c : consumption given decision-state
    m : cash on hand. Also provides value and marginal value of m given optimal c.
    """

    distance_criteria = ["vp_func"]
    c_func: LinearFast = NullFunc()  # policy this stage is consumption
    v_func: ValueFuncCRRA = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    mNrmMin: float = 0.0


@dataclass
class LaborLeisureStage(MetricObject):
    """This class contains the policy function of the labor decision and the value and
    marginal value functions of the state b : bank balances and theta : wages.
    """

    distance_criteria = ["vp_func"]
    labor_func: LinearFast = NullFunc()
    leisure_func: LinearFast = NullFunc()
    c_func: LinearFast = NullFunc()
    v_func: ValueFuncCRRA = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    bNrmMin: float = 0.0


@dataclass
class LaborSeparableSolution(MetricObject):
    distance_metric = ["consumption_saving", "labor_leisure", "post_decision"]
    labor_leisure: LaborLeisureStage = field(default_factory=LaborLeisureStage)
    consumption_saving: ConsumptionSavingStage = field(
        default_factory=ConsumptionSavingStage
    )
    post_decision: PostDecisionStage = field(default_factory=PostDecisionStage)


class UtilityFuncLeisure(UtilityFuncStoneGeary):
    def __init__(self, CRRA, factor):
        super().__init__(CRRA, factor=factor)


class DisutilityFuncLabor(UtilityFuncStoneGeary):
    def __init__(self, CRRA, factor):
        super().__init__(-CRRA, factor=factor)


@dataclass
class LaborSeparableSolver:
    solution_next: ConsumerLaborSolution
    TranShkDstn: DiscreteDistribution
    IncShkDstn: DiscreteDistribution
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
    aXtraGrid: np.array
    TranShkGrid: np.array
    vFuncBool: float
    CubicBool: float

    def __post_init__(self):
        self.def_utility_funcs()

    def def_utility_funcs(self):
        """Define temporary functions for utility and its derivative and inverse"""
        self.u_func = UtilityFuncCRRA(self.CRRA)

        if self.Disutility:
            # TODO: verify disutil
            # ty of labor solution
            self.n_func = DisutilityFuncLabor(self.LaborCRRA, self.LaborFactor)
        else:
            self.n_func = UtilityFuncLeisure(self.LeisureCRRA, self.LeisureFactor)

    def prepare_to_solve(self):
        # Unpack next period's solution
        self.vp_func_next = self.solution_next.labor_leisure.vp_func

    def prepare_to_calc_EndOfPrdvP(self):
        # we can add zero back because it's not the natural borrowing constraint

        self.zero_bound = self.TranShkGrid[0] == 0.0

        self.aGrid = (
            self.aXtraGrid if self.zero_bound else np.append(0.0, self.aXtraGrid)
        )

    def calc_EndOfPrdvP(self):
        def dvda_func(shock, anrm):
            p_shk = self.PermGroFac * shock[0]
            bnrm = anrm * self.Rfree / p_shk
            return p_shk**-self.CRRA * self.vp_func_next(
                bnrm,
                shock[1].repeat(bnrm.size),
            )

        EndOfPrdvP_vals = calc_expectation(self.IncShkDstn, dvda_func, self.aGrid)
        EndOfPrdvP_nvrs = self.u_func.derinv(EndOfPrdvP_vals)
        if self.zero_bound:
            EndOfPrdvP_nvrs = np.append(0.0, EndOfPrdvP_nvrs)
            aGrid_temp = np.append(0.0, self.aGrid)
        else:
            aGrid_temp = self.aGrid
        EndOfPrdvP_nvrs_func = LinearInterp(aGrid_temp, EndOfPrdvP_nvrs)
        EndOfPrdvP_func = MargValueFuncCRRA(EndOfPrdvP_nvrs_func, self.CRRA)

        self.post_decision_stage = PostDecisionStage(
            vp_func=EndOfPrdvP_func,
            vp=EndOfPrdvP_nvrs,
        )

    def make_consumption_solution(self):
        # this is the consumption function that is consistent with the exogneous asset level
        # this decision comes before end of period post decision value function, but after
        # the labor-leisure decision function
        cGrid = self.post_decision_stage.vp
        aGrid_temp = np.append(0.0, self.aGrid) if self.zero_bound else self.aGrid
        mGrid = cGrid + aGrid_temp

        cFuncEndOfPrd = LinearInterp(mGrid, cGrid)
        vPfuncEndOfPrd = MargValueFuncCRRA(cFuncEndOfPrd, self.CRRA)

        self.consumption_saving_stage = ConsumptionSavingStage(
            c_func=cFuncEndOfPrd,
            vp_func=vPfuncEndOfPrd,
        )

    def make_labor_leisure_solution(self):
        # this mgrid and cgrid are consistent with exogenous asset level
        mgrid = np.append(0.0, self.aXtraGrid)

        # in the previous step we found the endogenous m that is consistent with
        # exogenous asset level, now we can use m (and tshk) as the exogenous grids
        # that will give us the consistent labor, leisure, and bank balances for a
        mnrmat, tshkmat = np.meshgrid(mgrid, self.TranShkGrid, indexing="ij")

        # this is the egm step, given exog m and tshk, find leisure
        lsrmat = self.n_func.derinv(
            self.consumption_saving_stage.vp_func(mnrmat) * self.WageRte * tshkmat,
        )
        # Make sure lsrgrid is not greater than 1.0
        if self.zero_bound:
            lsrmat[:, 0] = 1.0

        lbrmat = 1.0 - lsrmat  # labor is 1 - leisure
        # bank balances = cash on hand - wage * labor
        bnrmat = mnrmat - tshkmat * self.WageRte * lbrmat

        lsrFunc = interp_on_interp(lsrmat, [bnrmat, tshkmat])
        lbrFunc = interp_on_interp(lbrmat, [bnrmat, tshkmat])

        def leisure_func(b, t):
            return np.clip(lsrFunc(b, t), 0.0, 1.0)

        def labor_func(b, t):
            return np.clip(lbrFunc(b, t), 0.0, 1.0)

        bmat = mnrmat

        labor = labor_func(bmat, tshkmat)
        mmat = bmat + self.WageRte * tshkmat * labor
        cmat = self.consumption_saving_stage.c_func(mmat)

        # as in the terminal solution, we construct the consumption function by using
        # the c that was consistent with a, and the b that was consistent with m

        cFunc = interp_on_interp(cmat, [bnrmat, tshkmat])
        vPfunc_now = MargValueFuncCRRA(cFunc, self.CRRA)

        self.make_labor_leisure_stage = LaborLeisureStage(
            labor_func=labor_func,
            leisure_func=leisure_func,
            c_func=cFunc,
            vp_func=vPfunc_now,
        )

    def make_consumer_labor_solution(self):
        self.solution = LaborSeparableSolution(
            post_decision=self.post_decision_stage,
            consumption_saving=self.consumption_saving_stage,
            labor_leisure=self.make_labor_leisure_stage,
        )

    def solve(self):
        self.prepare_to_calc_EndOfPrdvP()
        self.calc_EndOfPrdvP()

        self.make_consumption_solution()
        self.make_labor_leisure_solution()

        self.make_consumer_labor_solution()

        return self.solution


def make_labor_separable_solution_terminal(
    CRRA,
    aXtraGrid,
    LbrCost,
    WageRte,
    TranShkGrid,
    Disutility,
    LaborFactor,
    LaborCRRA,
    LeisureFactor,
    LeisureCRRA,
):
    """
    Constructs the terminal period solution for the Labor Separable model.
    Agent works before consumption, so we need optimal labor and leisure in terminal period.
    """

    # Consumption stage
    def c_func_cs(m):
        return m  # consume all cash in terminal period

    v_func_cs = ValueFuncCRRA(c_func_cs, CRRA)
    vp_func_cs = MargValueFuncCRRA(c_func_cs, CRRA)

    cs_stage = ConsumptionSavingStage(
        c_func=c_func_cs,
        v_func=v_func_cs,
        vp_func=vp_func_cs,
    )

    # Labor/leisure stage
    uFunc = UtilityFuncCRRA(CRRA)

    if Disutility:
        nFunc = DisutilityFuncLabor(LaborCRRA, LaborFactor)
    else:
        nFunc = UtilityFuncLeisure(LeisureCRRA, LeisureFactor)

    def policy_unconstrained(m, theta):
        return nFunc.derinv(vp_func_cs(m) * theta * wagerate)

    t = -1
    tshkgrid = TranShkGrid[t]
    wagerate = WageRte[t]

    zero_bound = True if tshkgrid[0] == 0.0 else False

    mgrid = np.append(0.0, aXtraGrid)
    mnrm_mat, tshk_mat = np.meshgrid(mgrid, tshkgrid, indexing="ij")

    with np.errstate(all="ignore"):
        x = policy_unconstrained(mnrm_mat, tshk_mat)

    if Disutility:
        labor_mat = x
        leisure_mat = 1 - labor_mat
    else:
        leisure_mat = x
        labor_mat = 1 - leisure_mat

    if zero_bound:
        leisure_mat[:, 0] = 1.0
        labor_mat[:, 0] = 0.0

    bnrm_mat = mnrm_mat - tshk_mat * labor_mat * wagerate

    terminal_grids = {
        "mnrm": mnrm_mat,
        "bnrm": bnrm_mat,
        "tshk": tshk_mat,
        "labor": labor_mat,
        "leisure": leisure_mat,
    }

    from egmn.utilities import interp_on_interp

    leisure_func_unc = interp_on_interp(leisure_mat, [bnrm_mat, tshk_mat])
    labor_func_unc = interp_on_interp(labor_mat, [bnrm_mat, tshk_mat])

    leisure_func_terminal = lambda b, t: np.clip(leisure_func_unc(b, t), 0.0, 1.0)
    labor_func_terminal = lambda b, t: np.clip(labor_func_unc(b, t), 0.0, 1.0)

    # Recalculate on rectangular grid
    bnrm_mat = mnrm_mat
    labor_mat = labor_func_terminal(bnrm_mat, tshk_mat)
    leisure_mat = leisure_func_terminal(bnrm_mat, tshk_mat)
    mnrm_mat = bnrm_mat + tshk_mat * labor_mat * wagerate

    if Disutility:
        vnrm_mat = -nFunc(labor_mat) + v_func_cs(mnrm_mat)
    else:
        vnrm_mat = nFunc(leisure_mat) + v_func_cs(mnrm_mat)

    vnrmnvrs_mat = uFunc.inverse(vnrm_mat)

    cFunc_terminal = interp_on_interp(mnrm_mat, [bnrm_mat, tshk_mat])
    vNvrs_func = interp_on_interp(vnrmnvrs_mat, [bnrm_mat, tshk_mat])
    vFunc_terminal = ValueFuncCRRA(vNvrs_func, CRRA)
    vPfunc_terminal = MargValueFuncCRRA(cFunc_terminal, CRRA)

    ll_stage = LaborLeisureStage(
        labor_func=labor_func_terminal,
        leisure_func=leisure_func_terminal,
        c_func=cFunc_terminal,
        v_func=vFunc_terminal,
        vp_func=vPfunc_terminal,
    )

    solution_terminal = LaborSeparableSolution(
        labor_leisure=ll_stage,
        consumption_saving=cs_stage,
    )

    solution_terminal.terminal_grids = terminal_grids

    return solution_terminal


init_labor_separable = init_labor_intensive.copy()
init_labor_separable["Disutility"] = False
init_labor_separable["LaborFactor"] = 1.0
init_labor_separable["LaborCRRA"] = 2.0
init_labor_separable["LeisureFactor"] = 1.0
init_labor_separable["LeisureCRRA"] = 2.0

# Override the solution_terminal constructor
if "constructors" in init_labor_separable:
    init_labor_separable["constructors"] = {
        k: v
        for k, v in init_labor_separable["constructors"].items()
        if k != "solution_terminal"
    }
    init_labor_separable["constructors"]["solution_terminal"] = (
        make_labor_separable_solution_terminal
    )


# Define the LaborSeparableConsumerType class with proper defaults
class LaborSeparableConsumerType(LaborIntMargConsumerType):
    """
    Agent type for labor-leisure choice with separable utility.
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

    default_ = {
        "params": init_labor_separable,
        "solver": make_one_period_oo_solver(LaborSeparableSolver),
    }


@dataclass
class PortfolioPostDecisionStage(MetricObject):
    """This class contains the value and marginal value functions of the problem given
    post-decision states a : savings and s : risky share of portfolio.
    """

    v_func: ValueFuncCRRA = NullFunc()  # value function
    dvda_func: MargValueFuncCRRA = NullFunc()  # marginal value function wrt assets
    # marginal value function wrt risky share
    dvds_func: MargValueFuncCRRA = NullFunc()
    dvda: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # marginal value wrt a on grid
    dvds: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # marginal value wrt s on grid


@dataclass
class PortfolioStage(MetricObject):
    """This class contains the policy function share : risky share of portfolio given
    decision-state a : savings, and additionally provides the value and marginal
    value functions given decision-state a.
    """

    share_func: LinearFast = NullFunc()  # policy this stage is risky share
    v_func: ValueFuncCRRA = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    vp_vals: np.ndarray = field(default_factory=lambda: np.array([]))
    vp_nvrs: np.ndarray = field(default_factory=lambda: np.array([]))
    aNrmMin: float = 0.0


@dataclass
class LaborPortfolioSolution(MetricObject):
    """The ConsLaborPortfolioSolution contains all of the solution stages of this model."""

    post_decision_stage: PostDecisionStage = field(default_factory=PostDecisionStage)
    portfolio_stage: PortfolioStage = field(default_factory=PortfolioStage)
    consumption_stage: ConsumptionSavingStage = field(
        default_factory=ConsumptionSavingStage
    )
    labor_stage: LaborLeisureStage = field(default_factory=LaborLeisureStage)
    isterminal: bool = False


# LaborPortfolioConsumerType will be defined at the end after init_labor_portfolio


@dataclass
class LaborPortfolioSolver(MetricObject):
    solution_next: LaborPortfolioSolution
    TranShkDstn: DiscreteDistribution
    ShockDstn: DiscreteDistribution
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
    ShareGrid: np.ndarray
    TranShkGrid: np.ndarray

    def __post_init__(self):
        self.def_utility_funcs()

    def def_utility_funcs(self):
        """Define temporary functions for utility and its derivative and inverse"""
        self.u = UtilityFuncCRRA(self.CRRA)

        if self.Disutility:
            self.n = DisutilityFuncLabor(self.LaborCRRA, self.LaborFactor)
        else:
            self.n = UtilityFuncLeisure(self.LeisureCRRA, self.LeisureFactor)

    def prepare_to_solve(self):
        """Create grids that will be used in solution stages."""
        # these set of grids will be used in post decision stage
        # portfolio stage and consumption stage.

        # assume for now natural borrowing constraint is less than 0
        # by construction of tShkDstn
        self.zero_bound = self.solution_next.isterminal

        if self.zero_bound:
            self.aNrmGrid = self.aXtraGrid
        else:
            self.aNrmGrid = np.append(0.0, self.aXtraGrid)  # safe to add zero

        self.aNrmMat, self.shareMat = np.meshgrid(
            self.aNrmGrid,
            self.ShareGrid,
            indexing="ij",
        )

        # these set of grids will be used in labor stage
        # these are exogenous
        self.mNrmGrid = self.aXtraGrid  # no zeros

        self.mNrmMat, self.TranShkMat_m = np.meshgrid(
            self.mNrmGrid,
            self.TranShkGrid,
            indexing="ij",
        )

        self.bNrmGrid = np.append(0.0, self.aXtraGrid)
        self.bNrmMat, self.TranShkMat_b = np.meshgrid(
            self.bNrmGrid,
            self.TranShkGrid,
            indexing="ij",
        )

        # name ShockDstn indeces
        self.PermShkIdx = 0
        self.TranShkIdx = 1
        self.RiskyShkIdx = 2

    def post_decision_stage(self, next_stage: LaborLeisureStage):
        vp_func_next = next_stage.vp_func

        def marginal_values(shocks, a_nrm, share):
            r_diff = shocks[self.RiskyShkIdx] - self.Rfree
            r_port = self.Rfree + r_diff * share
            p_shk = self.PermGroFac * shocks[self.PermShkIdx]
            t_shk = shocks[self.TranShkIdx] * np.ones_like(a_nrm)
            b_nrm_next = a_nrm * r_port / p_shk

            vp_next = p_shk ** (-self.CRRA) * vp_func_next(b_nrm_next, t_shk)

            dvda = r_port * vp_next
            dvds = a_nrm * r_diff * vp_next

            return dvda, dvds

        partials = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.ShockDstn,
                marginal_values,
                self.aNrmMat,
                self.shareMat,
            )
        )

        dvda = partials[0]
        dvds = partials[1]

        # make post decision dvda function

        # endogenous grid to calculate consumption
        dvda_nvrs = self.u.derinv(dvda)

        if self.zero_bound:
            dvda_nvrs_temp = np.insert(dvda_nvrs, 0, 0.0, axis=0)
            aNrmGrid_temp = np.append(0.0, self.aNrmGrid)
        else:
            dvda_nvrs_temp = dvda_nvrs
            aNrmGrid_temp = self.aNrmGrid

        dvda_nvrs_func = LinearFast(dvda_nvrs_temp, [aNrmGrid_temp, self.ShareGrid])
        dvda_func = MargValueFuncCRRA(dvda_nvrs_func, self.CRRA)

        # make post decision dvds function
        # NOTE: dvds can be negative (optimal share has dvds=0), so we cannot use
        # MargValueFuncCRRA which assumes positive values. Use direct interpolation.
        dvds_func = LinearFast(dvds, [self.aNrmGrid, self.ShareGrid])

        post_decision_stage_solution = PortfolioPostDecisionStage(
            dvda_func=dvda_func,
            dvds_func=dvds_func,
            dvda=dvda,
            dvds=dvds,
        )

        return post_decision_stage_solution

    def optimize_share(self, foc):
        """Optimization of Share on continuous interval [0,1]"""
        # For each value of aNrm, find the value of Share such that FOC-Share == 0.
        crossing = np.logical_and(foc[..., 1:] <= 0.0, foc[..., :-1] >= 0.0)
        share_idx = np.argmax(crossing, axis=1)
        a_idx = np.arange(self.aNrmGrid.size)
        bot_s = self.ShareGrid[share_idx]
        top_s = self.ShareGrid[share_idx + 1]
        bot_f = foc[a_idx, share_idx]
        top_f = foc[a_idx, share_idx + 1]
        alpha = 1.0 - top_f / (top_f - bot_f)

        opt_share = (1.0 - alpha) * bot_s + alpha * top_s

        # If agent wants to put more than 100% into risky asset, he is constrained
        constraint_top = foc[:, -1] > 0.0
        # Likewise if he wants to put less than 0% into risky asset
        constraint_bot = foc[:, 0] < 0.0

        # For values of aNrm at which the agent wants to put
        # more than 100% into risky asset, constrain them
        opt_share[constraint_top] = 1.0
        opt_share[constraint_bot] = 0.0

        if not self.zero_bound:
            # aNrm=0, so there's no way to "optimize" the portfolio
            opt_share[0] = 1.0  # set 1.0 for continuity

        return opt_share

    def portfolio_stage(self, next_stage: PostDecisionStage):
        opt_share = self.optimize_share(next_stage.dvds)

        dvda = next_stage.dvda_func(self.aNrmGrid, opt_share)
        dvda_nvrs = self.u.derinv(dvda)

        if self.zero_bound:
            opt_share_temp = np.append(opt_share[0], opt_share)
            dvda_nvrs_temp = np.append(0.0, dvda_nvrs)
            aNrmGrid_temp = np.append(0.0, self.aNrmGrid)
        else:
            opt_share_temp = opt_share
            dvda_nvrs_temp = dvda_nvrs
            aNrmGrid_temp = self.aNrmGrid

        share_func = LinearFast(opt_share_temp, [aNrmGrid_temp])

        dvda_nvrs_func = LinearFast(dvda_nvrs_temp, [aNrmGrid_temp])
        dvda_func = MargValueFuncCRRA(dvda_nvrs_func, self.CRRA)

        portfolio_stage_solution = PortfolioStage(
            share_func=share_func,
            vp_func=dvda_func,
            vp_vals=dvda,
            vp_nvrs=dvda_nvrs,
        )

        return portfolio_stage_solution

    def consumption_stage(self, next_stage: PortfolioStage):
        cNrmGrid = next_stage.vp_nvrs
        mNrmGrid = self.aNrmGrid + cNrmGrid

        c_temp = np.append(0.0, cNrmGrid)
        m_temp = np.append(0.0, mNrmGrid)

        c_func = LinearFast(c_temp, [m_temp])
        vp_func = MargValueFuncCRRA(c_func, self.CRRA)

        consumption_stage_solution = ConsumptionSavingStage(
            c_func=c_func,
            vp_func=vp_func,
        )

        return consumption_stage_solution

    def labor_stage(self, next_stage: ConsumptionSavingStage):
        vp_func_next = next_stage.vp_func

        # First use exogenous self.mNrmMat and self.TranShkMat_m

        # unconstrained labor-leisure: solve n'(ℓ) = u'(c) · w · θ
        leisureEndogMat = self.n.derinv(
            vp_func_next(self.mNrmMat) * self.WageRte * self.TranShkMat_m
        )
        laborEndogMat = 1.0 - leisureEndogMat
        bNrmEndogMat = self.mNrmMat - self.WageRte * self.TranShkMat_m * laborEndogMat

        grids = {
            "mNrm": self.mNrmMat,
            "bNrm": bNrmEndogMat,
            "tShk": self.TranShkMat_m,
            "labor": laborEndogMat,
            "leisure": leisureEndogMat,
        }

        leisure_unconstrained_func_by_tShk = []
        for i in range(self.TranShkGrid.size):
            leisure_unconstrained_func_by_tShk.append(
                LinearFast(leisureEndogMat[:, i], [bNrmEndogMat[:, i]]),
            )

        leisure_unconstrained_func = LinearInterpOnInterp1D(
            leisure_unconstrained_func_by_tShk,
            self.TranShkGrid,
        )

        # Now use exogenous self.bNrmMat and self.TranShkMat_b

        leisure_unconstrained = leisure_unconstrained_func(
            self.bNrmMat,
            self.TranShkMat_b,
        )

        leisureExogMat = np.clip(leisure_unconstrained, 0.0, 1.0)  # constrained
        laborExogMat = 1.0 - leisureExogMat

        labor_func = LinearFast(laborExogMat, [self.bNrmGrid, self.TranShkGrid])
        leisure_func = LinearFast(leisureExogMat, [self.bNrmGrid, self.TranShkGrid])

        # Compute cash-on-hand: m = b + w·θ·labor(b, θ)
        mNrmExogMat = self.bNrmMat + self.WageRte * self.TranShkMat_b * laborExogMat

        # Marginal value function: dvdb = u'(c(m)) = vp(m)
        dvdb = next_stage.vp_func(mNrmExogMat)
        dvdb_nvrs = self.u.derinv(dvdb)
        dvdb_nvrs_func = LinearFast(dvdb_nvrs, [self.bNrmGrid, self.TranShkGrid])
        dvdb_func = MargValueFuncCRRA(dvdb_nvrs_func, self.CRRA)

        # Consumption function: c(b, θ) from consumption stage
        cExogMat = next_stage.c_func(mNrmExogMat)
        c_func = LinearFast(cExogMat, [self.bNrmGrid, self.TranShkGrid])

        labor_stage_solution = LaborLeisureStage(
            labor_func=labor_func,
            leisure_func=leisure_func,
            c_func=c_func,
            vp_func=dvdb_func,
        )
        labor_stage_solution.grids = grids

        return labor_stage_solution

    def solve(self):
        labor_stage_next = self.solution_next.labor_stage

        post_decision_solution = self.post_decision_stage(labor_stage_next)
        portfolio_stage_solution = self.portfolio_stage(post_decision_solution)
        consumption_stage_solution = self.consumption_stage(portfolio_stage_solution)
        labor_stage_solution = self.labor_stage(consumption_stage_solution)

        self.solution = LaborPortfolioSolution(
            post_decision_stage=post_decision_solution,
            portfolio_stage=portfolio_stage_solution,
            consumption_stage=consumption_stage_solution,
            labor_stage=labor_stage_solution,
        )

        return self.solution


def make_labor_portfolio_solution_terminal(CRRA):
    """
    Constructs the terminal period solution for the Labor Portfolio model.
    In the terminal period, agents consume everything and don't work or save.
    """
    # In the terminal period the risky share is trivially 0 since there is
    # no continuation; agents consume all resources and save nothing
    portfolio_stage = PortfolioStage(share_func=lambda a: a * 0.0)

    # In terminal period agents consume everything
    util = UtilityFuncCRRA(CRRA)
    consumption_stage = ConsumptionSavingStage(
        c_func=lambda m: m,
        v_func=util,
        vp_func=util.der,
    )

    # In terminal period agents do not work, and so b = m
    # and marginal value is the same as in consumption stage
    labor_stage = LaborLeisureStage(
        labor_func=lambda b, theta: b * 0.0,
        v_func=lambda b, theta: util(b),
        vp_func=lambda b, theta: util.der(b),
    )

    # Create terminal solution object
    return LaborPortfolioSolution(
        portfolio_stage=portfolio_stage,
        consumption_stage=consumption_stage,
        labor_stage=labor_stage,
        isterminal=True,
    )


init_labor_portfolio = init_labor_intensive.copy()

# Merge constructors from both parent classes carefully
labor_constructors = init_labor_intensive.get("constructors", {}).copy()
portfolio_constructors = init_portfolio.get("constructors", {}).copy()

# Update with portfolio params but preserve labor constructors
init_labor_portfolio.update(init_portfolio)

# Merge the constructors - start with portfolio (base), add labor-specific ones
if "constructors" in init_labor_portfolio:
    init_labor_portfolio["constructors"] = portfolio_constructors.copy()
    # Add labor-specific constructors
    init_labor_portfolio["constructors"]["TranShkGrid"] = labor_constructors[
        "TranShkGrid"
    ]
    init_labor_portfolio["constructors"]["LbrCost"] = labor_constructors["LbrCost"]

# Set our custom parameters
init_labor_portfolio["LeisureFactor"] = 0.5
init_labor_portfolio["LeisureCRRA"] = 2.0
init_labor_portfolio["UnempPrb"] = 0.0
init_labor_portfolio["Disutility"] = False
init_labor_portfolio["LaborFactor"] = 0.5
init_labor_portfolio["LaborCRRA"] = 1.0
init_labor_portfolio["CRRA"] = 2.0

# Override the solution_terminal constructor with our custom one
if "constructors" in init_labor_portfolio:
    init_labor_portfolio["constructors"]["solution_terminal"] = (
        make_labor_portfolio_solution_terminal
    )


# Define the LaborPortfolioConsumerType class with proper defaults
class LaborPortfolioConsumerType(PortfolioConsumerType, LaborIntMargConsumerType):
    """
    Agent type combining labor-leisure choice with portfolio allocation.
    Agents choose consumption, savings, labor supply, and portfolio allocation each period.
    """

    # Merge time_vary_ from both parent classes (union of both lists)
    time_vary_ = copy(LaborIntMargConsumerType.time_vary_)
    for item in PortfolioConsumerType.time_vary_:
        if item not in time_vary_:
            time_vary_.append(item)

    # Merge time_inv_ from both parent classes (union of both lists)
    time_inv_ = copy(LaborIntMargConsumerType.time_inv_)
    for item in PortfolioConsumerType.time_inv_:
        if item not in time_inv_:
            time_inv_.append(item)
    # Add our custom time-invariant variables
    time_inv_ += [
        "Disutility",
        "LaborFactor",
        "LaborCRRA",
        "LeisureFactor",
        "LeisureCRRA",
    ]

    default_ = {
        "params": init_labor_portfolio,
        "solver": make_one_period_oo_solver(LaborPortfolioSolver),
    }
