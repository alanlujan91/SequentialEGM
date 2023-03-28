from copy import copy
from dataclasses import dataclass

import numpy as np
from HARK import MetricObject
from HARK.ConsumptionSaving.ConsLaborModel import (
    LaborIntMargConsumerType,
    init_labor_intensive,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.core import make_one_period_oo_solver
from HARK.distribution import DiscreteDistribution, calc_expectation
from HARK.interpolation import (
    LinearFast,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityP_inv,
    UtilityFuncCRRA,
    UtilityFunction,
)
from HARK.utilities import NullFunc


@dataclass
class PostDecisionStage(MetricObject):
    """
    This class contains the value and marginal value functions of the problem given
    post-decision states a : savings and s : risky share of portfolio.
    """

    v_func: ValueFuncCRRA = NullFunc()  # value function
    dvda_func: MargValueFuncCRRA = NullFunc()  # marginal value function wrt assets
    # marginal value function wrt risky share
    dvds_func: MargValueFuncCRRA = NullFunc()
    dvda: np.array = np.array([])  # marginal value wrt a on grid
    dvds: np.array = np.array([])  # marginal value wrt s on grid


@dataclass
class PortfolioStage(MetricObject):
    """
    This class contains the policy function share : risky share of portfolio given
    decision-state a : savings, and additionally provides the value and marginal
    value functions given decision-state a.
    """

    share_func: LinearFast = NullFunc()  # policy this stage is risky share
    v_func: ValueFuncCRRA = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    vp_vals: np.array = np.array([])
    vp_nvrs: np.array = np.array([])
    aNrmMin: float = 0.0


@dataclass
class ConsumptionStage(MetricObject):
    """
    This class contains the policy function c : consumption given decision-state
    m : cash on hand. Also provides value and marginal value of m given optimal c.
    """

    c_func: LinearFast = NullFunc()  # policy this stage is consumption
    v_func: ValueFuncCRRA = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    mNrmMin: float = 0.0


@dataclass
class LaborStage(MetricObject):
    """
    This class contains the policy function of the labor decision and the value and
    marginal value functions of the state b : bank balances and theta : wages.
    """

    labor_func: LinearFast = NullFunc()  # policy this stage is leisure/labor
    v_func: ValueFuncCRRA = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    bNrmMin: float = 0.0


@dataclass
class ConsLaborPortfolioSolution(MetricObject):
    """
    The ConsLaborPortfolioSolution contains all of the solution stages of this model.
    """

    post_decision_stage: PostDecisionStage = PostDecisionStage()
    portfolio_stage: PortfolioStage = PortfolioStage()
    consumption_stage: ConsumptionStage = ConsumptionStage()
    labor_stage: LaborStage = LaborStage()
    isterminal: bool = False


class LaborPortfolioConsumerType(PortfolioConsumerType, LaborIntMargConsumerType):
    time_inv_ = copy(LaborIntMargConsumerType.time_inv_)
    time_inv_ += [
        "DisutilLabor",
        "LaborConstant",
        "LaborShare",
        "LeisureConstant",
        "LeisureShare",
    ]

    def __init__(self, **kwds):
        params = init_labor_portfolio.copy()
        params.update(kwds)

        PortfolioConsumerType.__init__(self, **params)

        self.solve_one_period = make_one_period_oo_solver(ConsLaborPortfolioSolver)

    def update(self):
        LaborIntMargConsumerType.update(self)
        PortfolioConsumerType.update(self)

    def update_LbrCost(self):
        pass

    def update_solution_terminal(self):
        # in the terminal period the risky share is trivially 0 since there is
        # no continuation; agents consume all resources and save nothing
        portfolio_stage = PortfolioStage(share_func=lambda a: a * 0.0)

        # in terminal period agents consume everything

        util = UtilityFuncCRRA(self.CRRA)
        consumption_stage = ConsumptionStage(
            c_func=lambda m: m, v_func=util, vp_func=util.der
        )

        # in terminal period agents do not work, and so b = m
        # and marginal value is the same as in consumption stage
        labor_stage = LaborStage(
            labor_func=lambda b, theta: b * 0.0,
            v_func=lambda b, theta: util(b),
            vp_func=lambda b, theta: util.der(b),
        )

        # create terminal solution object
        self.solution_terminal = ConsLaborPortfolioSolution(
            portfolio_stage=portfolio_stage,
            consumption_stage=consumption_stage,
            labor_stage=labor_stage,
            isterminal=True,
        )


@dataclass
class ConsLaborPortfolioSolver(MetricObject):
    solution_next: ConsLaborPortfolioSolution
    TranShkDstn: DiscreteDistribution
    ShockDstn: DiscreteDistribution
    LivPrb: float
    DiscFac: float
    CRRA: float
    DisutilLabor: bool
    LaborConstant: float
    LaborShare: float
    LeisureConstant: float
    LeisureShare: float
    Rfree: float
    PermGroFac: float
    BoroCnstArt: float
    aXtraGrid: np.ndarray
    ShareGrid: np.ndarray
    TranShkGrid: np.ndarray

    def __post_init__(self):
        self.def_utility_funcs()

    def def_utility_funcs(self):
        """
        Define temporary functions for utility and its derivative and inverse
        """

        self.u = UtilityFuncCRRA(self.CRRA)

        if self.DisutilLabor:
            # use these functions if model defined by disutility of labor
            def n(x):
                return -self.LaborConstant * CRRAutility(1 - x, -self.LaborShare)

            def n_der(x):
                return self.LaborConstant * CRRAutilityP(1 - x, -self.LaborShare)

            def n_derinv(x):
                return 1 - CRRAutilityP_inv(x / self.LaborConstant, -self.LaborShare)

        else:
            # use these functions if model defined by utility of leisure (default)
            def n(x):
                return self.LeisureConstant * CRRAutility(x, self.LeisureShare)

            def n_der(x):
                return self.LeisureConstant * CRRAutilityP(x, self.LeisureShare)

            def n_derinv(x):
                return CRRAutilityP_inv(x / self.LeisureConstant, self.LeisureShare)

        self.n = UtilityFunction(n, n_der, n_derinv)

    def prepare_to_solve(self):
        """
        Create grids that will be used in solution stages.
        """

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
            self.aNrmGrid, self.ShareGrid, indexing="ij"
        )

        # these set of grids will be used in labor stage
        # these are exogenous
        self.mNrmGrid = self.aXtraGrid  # no zeros

        self.mNrmMat, self.TranShkMat_m = np.meshgrid(
            self.mNrmGrid, self.TranShkGrid, indexing="ij"
        )

        self.bNrmGrid = np.append(0.0, self.aXtraGrid)
        self.bNrmMat, self.TranShkMat_b = np.meshgrid(
            self.bNrmGrid, self.TranShkGrid, indexing="ij"
        )

        # name ShockDstn indeces
        self.PermShkIdx = 0
        self.TranShkIdx = 1
        self.RiskyShkIdx = 2

    def post_decision_stage(self, next_stage: LaborStage):
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
                self.ShockDstn, marginal_values, self.aNrmMat, self.shareMat
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
        dvds_nvrs = self.u.derinv(dvds)
        dvds_nvrs_func = LinearFast(dvds_nvrs, [self.aNrmGrid, self.ShareGrid])
        dvds_func = MargValueFuncCRRA(dvds_nvrs_func, self.CRRA)

        post_decision_stage_solution = PostDecisionStage(
            dvda_func=dvda_func,
            dvds_func=dvds_func,
            dvda=dvda,
            dvds=dvds,
        )

        return post_decision_stage_solution

    def optimize_share(self, foc):
        """
        Optimization of Share on continuous interval [0,1]
        """

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
            share_func=share_func, vp_func=dvda_func, vp_vals=dvda, vp_nvrs=dvda_nvrs
        )

        return portfolio_stage_solution

    def consumption_stage(self, next_stage: PortfolioStage):
        cNrmGrid = next_stage.vp_nvrs
        mNrmGrid = self.aNrmGrid + cNrmGrid

        c_temp = np.append(0.0, cNrmGrid)
        m_temp = np.append(0.0, mNrmGrid)

        c_func = LinearFast(c_temp, [m_temp])
        vp_func = MargValueFuncCRRA(c_func, self.CRRA)

        consumption_stage_solution = ConsumptionStage(c_func=c_func, vp_func=vp_func)

        return consumption_stage_solution

    def labor_stage(self, next_stage: ConsumptionStage):
        vp_func_next = next_stage.vp_func

        # First use exogenous self.mNrmMat and self.TranShkMat_m

        # unconstrained labor-leisure
        leisureEndogMat = self.n.inv(vp_func_next(self.mNrmMat) * self.TranShkMat_m)
        laborEndogMat = 1.0 - leisureEndogMat
        bNrmEndogMat = self.mNrmMat - self.TranShkMat_m * laborEndogMat

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
                LinearFast(leisureEndogMat[:, i], [bNrmEndogMat[:, i]])
            )

        leisure_unconstrained_func = LinearInterpOnInterp1D(
            leisure_unconstrained_func_by_tShk, self.TranShkGrid
        )

        # Now use exogenous self.bNrmMat and self.TranShkMat_b

        leisure_unconstrained = leisure_unconstrained_func(
            self.bNrmMat, self.TranShkMat_b
        )

        leisureExogMat = np.clip(leisure_unconstrained, 0.0, 1.0)  # constrained
        laborExogMat = 1.0 - leisureExogMat

        labor_func = LinearFast(laborExogMat, [self.bNrmGrid, self.TranShkGrid])
        leisure_func = LinearFast(leisureExogMat, [self.bNrmGrid, self.TranShkGrid])

        mNrmExogMat_temp = self.bNrmMat + self.TranShkMat_b * laborExogMat
        dvdb = next_stage.vp_func(mNrmExogMat_temp)
        dvdb_nvrs = self.u.derinv(dvdb)
        dvdb_nvrs_func = LinearFast(dvdb_nvrs, [self.bNrmGrid, self.TranShkGrid])
        dvdb_func = MargValueFuncCRRA(dvdb_nvrs_func, self.CRRA)

        labor_stage_solution = LaborStage(labor_func=labor_func, vp_func=dvdb_func)
        labor_stage_solution.leisure_func = leisure_func
        labor_stage_solution.grids = grids

        return labor_stage_solution

    def solve(self):
        labor_stage_next = self.solution_next.labor_stage

        post_decision_solution = self.post_decision_stage(labor_stage_next)
        portfolio_stage_solution = self.portfolio_stage(post_decision_solution)
        consumption_stage_solution = self.consumption_stage(portfolio_stage_solution)
        labor_stage_solution = self.labor_stage(consumption_stage_solution)

        self.solution = ConsLaborPortfolioSolution(
            post_decision_stage=post_decision_solution,
            portfolio_stage=portfolio_stage_solution,
            consumption_stage=consumption_stage_solution,
            labor_stage=labor_stage_solution,
        )

        return self.solution


init_labor_portfolio = init_labor_intensive.copy()
init_labor_portfolio.update(init_portfolio)
init_labor_portfolio["LeisureConstant"] = 0.5
init_labor_portfolio["LeisureShare"] = 2.0
init_labor_portfolio["UnempPrb"] = 0.0
init_labor_portfolio["DisutilLabor"] = False
init_labor_portfolio["LaborConstant"] = 0.5
init_labor_portfolio["LaborShare"] = 1.0
init_labor_portfolio["CRRA"] = 2.0
