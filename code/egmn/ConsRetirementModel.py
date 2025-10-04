from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from HARK.ConsumptionSaving.ConsPortfolioModel import init_portfolio
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.core import make_one_period_oo_solver
from HARK.distribution import DiscreteDistribution, calc_expectation
from HARK.interpolation import (
    BilinearInterp,
    GeneralizedRegressionUnstructuredInterp,
    LinearInterp,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
    calc_log_sum_choice_probs,
)
from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA, UtilityFuncStoneGeary
from HARK.utilities import NullFunc, make_grid_exp_mult


@dataclass
class PostDecisionStage(MetricObject):
    v_func: ValueFuncCRRA = NullFunc()
    dvda_func: MargValueFuncCRRA = NullFunc()
    dvdb_func: MargValueFuncCRRA = NullFunc()


@dataclass
class ConsumptionStage(MetricObject):
    c_func: BilinearInterp = NullFunc()
    v_func: ValueFuncCRRA = NullFunc()
    dvdl_func: MargValueFuncCRRA = NullFunc()
    dvdb_func: MargValueFuncCRRA = NullFunc()


@dataclass
class DepositStage(MetricObject):
    d_func: BilinearInterp = NullFunc()
    c_func: BilinearInterp = NullFunc()
    v_func: ValueFuncCRRA = NullFunc()
    dvdm_func: MargValueFuncCRRA = NullFunc()
    dvdn_func: MargValueFuncCRRA = NullFunc()


@dataclass
class RetiredSolution(MetricObject):
    c_func: MetricObject = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    v_func: ValueFuncCRRA = NullFunc()
    vp_end_func: MargValueFuncCRRA = NullFunc()
    v_end_func: ValueFuncCRRA = NullFunc()


@dataclass
class RetiringSolution(MetricObject):
    c_func: MetricObject = NullFunc()
    d_func: MetricObject = NullFunc()
    vp_func: MargValueFuncCRRA = NullFunc()
    v_func: ValueFuncCRRA = NullFunc()


@dataclass
class WorkingSolution(MetricObject):
    post_decision_stage: PostDecisionStage = field(default_factory=PostDecisionStage)
    consumption_stage: ConsumptionStage = field(default_factory=ConsumptionStage)
    deposit_stage: DepositStage = field(default_factory=DepositStage)


@dataclass
class DiscreteChoiceProbabilities(MetricObject):
    prob_working: MetricObject = NullFunc()
    prob_retiring: MetricObject = NullFunc()


@dataclass
class WorkerSolution(MetricObject):
    deposit_stage: DepositStage = field(default_factory=DepositStage)
    probabilities: DiscreteChoiceProbabilities = field(
        default_factory=DiscreteChoiceProbabilities
    )


@dataclass
class RetirementSolution(MetricObject):
    worker_solution: WorkerSolution = field(default_factory=WorkerSolution)
    retired_solution: RetiredSolution = field(default_factory=RetiredSolution)
    working_solution: WorkingSolution = field(default_factory=WorkingSolution)
    retiring_solution: RetiringSolution = field(default_factory=RetiringSolution)


class RetirementConsumerType(RiskyAssetConsumerType):
    time_inv_ = deepcopy(RiskyAssetConsumerType.time_inv_)
    time_inv_ = time_inv_ + [
        "DisutilLabor",
        "IncUnempRet",
        "TasteShkStd",
        "RfreeA",
        "RfreeB",
        "TaxDeduct",
    ]

    def __init__(self, **kwds):
        params = init_retirement_pension.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        super().__init__(**params)

        # Add consumer-type specific objects, copying to create independent versions
        self.solve_one_period = make_one_period_oo_solver(RetirementSolver)

    def update(self):
        self.update_grids()
        super().update()

        # self.update_solution_terminal()

    def update_solution_terminal(self):
        def c_func_retired(m):
            return m

        # retired problem
        self.retired_solution = RetiredSolution(
            c_func=c_func_retired,
            vp_func=MargValueFuncCRRA(c_func_retired, self.CRRA),
            v_func=ValueFuncCRRA(c_func_retired, self.CRRA),
        )

        def c_func(m, n):
            return m + n

        vp_func = MargValueFuncCRRA(c_func, self.CRRA)
        v_func = ValueFuncCRRA(c_func, self.CRRA)

        self.retiring_solution = RetiringSolution(
            c_func=c_func,
            vp_func=vp_func,
            v_func=v_func,
        )

        def d_func(m, n):
            return m * 0.0

        deposit_stage = DepositStage(
            c_func=c_func,
            d_func=d_func,
            dvdm_func=vp_func,
            dvdn_func=vp_func,
            v_func=v_func,
        )

        self.working_solution = WorkingSolution(deposit_stage=deposit_stage)

        # because working and retiring are the same,
        # no need for  weighted average
        self.worker_solution = WorkerSolution(deposit_stage=deposit_stage)

        self.solution_terminal = RetirementSolution(
            worker_solution=self.worker_solution,
            retired_solution=self.retired_solution,
            working_solution=self.working_solution,
            retiring_solution=self.retiring_solution,
        )

    def update_grids(self):
        # retirement

        self.aRetGrid = make_grid_exp_mult(
            0.0,
            self.aRetMax,
            self.aRetCount,
            self.aRetNestFac,
        )

        # post decision grids and exogenous grids for
        # consumption stage, include 0.0 as limiting case
        self.aGrid = make_grid_exp_mult(0.0, self.aMax, self.aCount, self.aNestFac)
        self.bGrid = make_grid_exp_mult(0.0, self.bMax, self.bCount, self.bNestFac)
        self.aMat, self.bMat = np.meshgrid(self.aGrid, self.bGrid, indexing="ij")

        # exogenous grids for pension deposit stage
        self.lGrid = make_grid_exp_mult(
            self.epsilon,
            self.lMax,
            self.lCount,
            self.lNestFac,
        )
        self.blGrid = make_grid_exp_mult(0.0, self.blMax, self.blCount, self.blNestFac)
        self.lMat, self.blMat = np.meshgrid(self.lGrid, self.blGrid, indexing="ij")

        # common worker grids
        self.mGrid = make_grid_exp_mult(
            self.epsilon,
            self.mMax,
            self.mCount,
            self.mNestFac,
        )
        self.nGrid = make_grid_exp_mult(0.0, self.nMax, self.nCount, self.nNestFac)
        self.mMat, self.nMat = np.meshgrid(self.mGrid, self.nGrid, indexing="ij")

        self.add_to_time_inv(
            "aRetGrid",
            "mGrid",
            "nGrid",
            "mMat",
            "nMat",
            "aGrid",
            "bGrid",
            "aMat",
            "bMat",
            "lGrid",
            "blGrid",
            "lMat",
            "blMat",
        )


@dataclass
class RetirementSolver:
    solution_next: RetirementSolution
    DiscFac: float
    CRRA: float
    DisutilLabor: float
    RfreeA: float
    RfreeB: float
    TaxDeduct: float
    TranShkDstn: DiscreteDistribution
    IncUnempRet: float
    TasteShkStd: float
    aRetGrid: np.array
    mGrid: np.array
    nGrid: np.array
    mMat: np.ndarray
    nMat: np.ndarray
    aGrid: np.array
    bGrid: np.array
    aMat: np.ndarray
    bMat: np.ndarray
    lGrid: np.array
    blGrid: np.array
    lMat: np.ndarray
    blMat: np.array

    def __post_init__(self):
        self.def_utility_funcs()

    def def_utility_funcs(self):
        self.u = UtilityFuncCRRA(self.CRRA)

        # pension deposit function: tax deduction from pension deposits
        # which is gradually decreasing in the level of deposits
        # CRRA = 1 makes it log function
        self.g = UtilityFuncStoneGeary(CRRA=1.0, factor=self.TaxDeduct, shifter=1.0)

    def solve_retired_problem(self, solution_next):
        vp_func_next = solution_next.vp_func
        v_func_next = solution_next.v_func

        # for retired problem
        # as long as there is retired income there is no risk of m_next = 0
        # and agent won't hit borrowing constraint
        # TODO: what if self.IncUnempRet = 0.0?
        mGrid_next = self.aRetGrid * self.RfreeA + self.IncUnempRet
        vp_end = self.DiscFac * self.RfreeA * vp_func_next(mGrid_next)
        vp_end_nvrs = self.u.derinv(vp_end)
        vp_end_nvrs_func = LinearInterp(self.aRetGrid, vp_end_nvrs)
        vp_end_func = MargValueFuncCRRA(vp_end_nvrs_func, self.CRRA)

        cGrid = vp_end_nvrs  # endogenous grid method
        mGrid = cGrid + self.aRetGrid

        # need to add artificial borrowing constraint at 0.0
        c_func = LinearInterp(np.append(0.0, mGrid), np.append(0.0, cGrid))
        vp_func = MargValueFuncCRRA(c_func, self.CRRA)

        # make retired value function
        # start by creating v_end_func
        v_end = self.DiscFac * v_func_next(mGrid_next)
        # value transformed through inverse utility
        v_end_nvrs = self.u.inv(v_end)
        v_end_nvrs_func = LinearInterp(self.aRetGrid, v_end_nvrs)
        v_end_func = ValueFuncCRRA(v_end_nvrs_func, self.CRRA)

        # calculate current value using mGrid that is consistent with agrid
        v = self.u(cGrid) + v_end
        # Construct the beginning-of-period value function
        v_nvrs = self.u.inv(v)  # value transformed through inverse utility
        v_nvrs_func = LinearInterp(np.append(0.0, mGrid), np.append(0.0, v_nvrs))
        v_func = ValueFuncCRRA(v_nvrs_func, self.CRRA)

        retired_solution = RetiredSolution(
            c_func=c_func,
            vp_func=vp_func,
            v_func=v_func,
            vp_end_func=vp_end_func,
            v_end_func=v_end_func,
        )

        return retired_solution

    def solve_retiring_problem(self, retired_solution):
        # this is kind of pointless
        retiring_solution = RetiringSolution(
            c_func=lambda m, n: retired_solution.c_func(m + n),
            d_func=lambda m, n: m * 0.0,
            vp_func=lambda m, n: retired_solution.vp_func(m + n),
            v_func=lambda m, n: retired_solution.v_func(m + n),
        )

        return retiring_solution

    def solve_post_decision_stage(self, deposit_stage_next):
        dvdm_func_next = deposit_stage_next.dvdm_func
        dvdn_func_next = deposit_stage_next.dvdn_func
        v_func_next = deposit_stage_next.v_func

        # First calculate marginal value functions
        def conditional_funcs(shock, abal, bbal):
            mnrm_next = self.RfreeA * abal + shock
            nnrm_next = self.RfreeB * bbal

            dvda = dvdm_func_next(mnrm_next, nnrm_next)
            dvdb = dvdn_func_next(mnrm_next, nnrm_next)
            v_end = v_func_next(mnrm_next, nnrm_next)
            return dvda, dvdb, v_end

        conditional_values = self.DiscFac * calc_expectation(
            self.TranShkDstn,
            conditional_funcs,
            self.aMat,
            self.bMat,
        )

        # TODO: what happens at a, b = 0.0?
        # probably nothing at a = 0 as long as min(shock) > 0
        dvda, dvdb, v_end = conditional_values

        dvda_nvrs = self.u.derinv(self.RfreeA * dvda)
        dvda_nvrs_func = BilinearInterp(dvda_nvrs, self.aGrid, self.bGrid)
        dvda_func = MargValueFuncCRRA(dvda_nvrs_func, self.CRRA)

        dvdb_nvrs = self.u.derinv(self.RfreeB * dvdb)
        dvdb_nvrs_func = BilinearInterp(dvdb_nvrs, self.aGrid, self.bGrid)
        dvdb_func = MargValueFuncCRRA(dvdb_nvrs_func, self.CRRA)

        # also calculate end of period value function

        # value transformed through inverse utility
        v_end_nvrs = self.u.inv(v_end)
        v_end_nvrs_func = BilinearInterp(v_end_nvrs, self.aGrid, self.bGrid)
        v_end_func = ValueFuncCRRA(v_end_nvrs_func, self.CRRA)

        post_decision_stage = PostDecisionStage(
            v_func=v_end_func,
            dvda_func=dvda_func,
            dvdb_func=dvdb_func,
        )

        # sometimes the best items to pass to next stage aren't functions
        post_decision_stage.dvda_nvrs = dvda_nvrs
        post_decision_stage.dvdb_nvrs = dvdb_nvrs
        post_decision_stage.value = v_end

        return post_decision_stage

    def interp_on_interp(self, values, grids):
        temp = []
        x, y = grids
        grid = y[0]
        for i in range(grid.size):
            temp.append(LinearInterp(x[:, i], values[:, i]))

        return LinearInterpOnInterp1D(temp, grid)

    def solve_consumption_stage(self, post_decision_stage):
        dvda_nvrs_next = post_decision_stage.dvda_nvrs
        dvdb_nvrs_next = post_decision_stage.dvdb_nvrs
        value_next = post_decision_stage.value

        cMat = dvda_nvrs_next  # endogenous grid method
        lMat = cMat + self.aMat

        # at l = 0, c = 0 so we need to add this limit
        lMat_temp = np.insert(lMat, 0, 0.0, axis=0)
        cMat_temp = np.insert(cMat, 0, 0.0, axis=0)
        np.insert(self.bMat, 0, 0.0, axis=0)

        # bMat is a regular grid, lMat is not so we'll need to use Warped

        c_func = self.interp_on_interp(cMat_temp, [lMat_temp, self.bMat])
        dvdl_func = MargValueFuncCRRA(c_func, self.CRRA)

        # again, at l = 0, c = 0 and a = 0, so repeat dvdb[0]
        dvdb_nvrs_temp = np.insert(dvdb_nvrs_next, 0, dvdb_nvrs_next[0], axis=0)
        dvdb_nvrs_func = self.interp_on_interp(dvdb_nvrs_temp, [lMat_temp, self.bMat])
        dvdb_func = MargValueFuncCRRA(dvdb_nvrs_func, self.CRRA)

        # make value function
        value = self.u(cMat) - self.DisutilLabor + value_next
        v_nvrs = self.u.inv(value)
        v_nvrs_temp = np.insert(v_nvrs, 0, 0.0, axis=0)

        # bMat is regular grid so we can use WarpedInterpOnInterp2D
        v_nvrs_func = self.interp_on_interp(v_nvrs_temp, [lMat_temp, self.bMat])
        v_func = ValueFuncCRRA(v_nvrs_func, self.CRRA)

        consumption_stage = ConsumptionStage(
            c_func=c_func,
            v_func=v_func,
            dvdl_func=dvdl_func,
            dvdb_func=dvdb_func,
        )

        return consumption_stage

    def solve_deposit_stage(self, consumption_stage):
        c_func_next = consumption_stage.c_func
        v_func_next = consumption_stage.v_func
        dvdl_func_next = consumption_stage.dvdl_func
        dvdb_func_next = consumption_stage.dvdb_func

        dvdl_next = dvdl_func_next(self.lMat, self.blMat)
        dvdb_next = dvdb_func_next(self.lMat, self.blMat)

        # endogenous grid method
        dMat = self.g.derinv(dvdl_next / dvdb_next - 1.0)

        mMat = self.lMat + dMat
        nMat = self.blMat - dMat - self.g(dMat)

        gaussian_interp_grid0 = GeneralizedRegressionUnstructuredInterp(
            self.lMat,
            [mMat, nMat],
            model="gaussian-process",
            std=True,
            model_kwargs={"normalize_y": True},
        )

        gaussian_interp_grid1 = GeneralizedRegressionUnstructuredInterp(
            self.blMat,
            [mMat, nMat],
            model="gaussian-process",
            std=True,
            model_kwargs={"normalize_y": True},
        )

        # interpolate grids
        lMat_temp = gaussian_interp_grid0(self.mMat, self.nMat)
        blMat_temp = gaussian_interp_grid1(self.mMat, self.nMat)

        # calculate derivatives
        dvdl_next = dvdl_func_next(lMat_temp, blMat_temp)
        dvdb_next = dvdb_func_next(lMat_temp, blMat_temp)

        # endogenous grid method
        dMat2 = self.g.derinv(dvdl_next / dvdb_next - 1.0)
        mMat2 = lMat_temp + dMat2
        nMat2 = blMat_temp - dMat2 - self.g(dMat2)

        # concatenate grids
        dMat = np.concatenate((dMat.flatten(), dMat2.flatten()))
        mMat = np.concatenate((mMat.flatten(), mMat2.flatten()))
        nMat = np.concatenate((nMat.flatten(), nMat2.flatten()))

        cond = dMat > -1.0
        dMat = dMat[cond]
        nMat = nMat[cond]
        mMat = mMat[cond]

        gaussian_interp = GeneralizedRegressionUnstructuredInterp(
            dMat,
            [mMat, nMat],
            model="gaussian-process",
            std=True,
            model_kwargs={"normalize_y": True},
        )

        # evaluate d on common grid
        dMat = gaussian_interp(self.mMat, self.nMat)
        dMat = np.maximum(0.0, dMat)
        lMat = self.mMat - dMat
        blMat = self.nMat + dMat + self.g(dMat)

        # evaluate c on common grid
        cMat = c_func_next(lMat, blMat)
        # there is no consumption or deposit when there is no cash on hand
        mGrid_temp = np.append(0.0, self.mGrid)
        dMat_temp = np.insert(dMat, 0, 0.0, axis=0)
        cMat_temp = np.insert(cMat, 0, 0.0, axis=0)

        d_func = BilinearInterp(dMat_temp, mGrid_temp, self.nGrid)
        c_func = BilinearInterp(cMat_temp, mGrid_temp, self.nGrid)
        dvdm_func = MargValueFuncCRRA(c_func, self.CRRA)

        dvdb_next = dvdb_func_next(lMat, blMat)

        dvdn_nvrs = self.u.derinv(dvdb_next)
        dvdn_nvrs_temp = np.insert(dvdn_nvrs, 0, dvdn_nvrs[0], axis=0)
        dvdn_nvrs_func = BilinearInterp(dvdn_nvrs_temp, mGrid_temp, self.nGrid)
        dvdn_func = MargValueFuncCRRA(dvdn_nvrs_func, self.CRRA)

        # make value function
        value = v_func_next(lMat, blMat)
        v_nvrs = self.u.inv(value)
        # insert value of 0 at m = 0
        v_nvrs_temp = np.insert(v_nvrs, 0, 0.0, axis=0)
        # mMat and nMat are irregular grids so we need Curvilinear2DInterp
        v_nvrs_func = BilinearInterp(v_nvrs_temp, mGrid_temp, self.nGrid)
        v_func = ValueFuncCRRA(v_nvrs_func, self.CRRA)

        deposit_stage = DepositStage(
            c_func=c_func,
            d_func=d_func,
            v_func=v_func,
            dvdm_func=dvdm_func,
            dvdn_func=dvdn_func,
        )

        deposit_stage.interp = gaussian_interp

        return deposit_stage

    def solve_working_problem(self, worker_solution_next):
        deposit_stage_next = worker_solution_next.deposit_stage

        post_decision_stage = self.solve_post_decision_stage(deposit_stage_next)
        consumption_stage = self.solve_consumption_stage(post_decision_stage)
        deposit_stage = self.solve_deposit_stage(consumption_stage)

        working_solution = WorkingSolution(
            post_decision_stage=post_decision_stage,
            consumption_stage=consumption_stage,
            deposit_stage=deposit_stage,
        )

        return working_solution

    def solve_worker_problem(self, working_solution, retiring_solution):
        vWorking_func = working_solution.deposit_stage.v_func
        cWorking_func = working_solution.deposit_stage.c_func
        dWorking_func = working_solution.deposit_stage.d_func
        dvdmWorking_func = working_solution.deposit_stage.dvdm_func
        dvdnWorking_func = working_solution.deposit_stage.dvdn_func

        mMat_temp = np.insert(self.mMat, 0, 0.0, axis=0)
        nMat_temp = np.insert(self.nMat, 0, self.nGrid, axis=0)

        vRetiring = retiring_solution.v_func(mMat_temp, nMat_temp)
        cRetiring = retiring_solution.c_func(mMat_temp, nMat_temp)
        vPRetiring = retiring_solution.vp_func(mMat_temp, nMat_temp)

        vWorking = vWorking_func(mMat_temp, nMat_temp)
        cWorking = cWorking_func(mMat_temp, nMat_temp)
        dWorking = dWorking_func(mMat_temp, nMat_temp)
        dvdmWorking = dvdmWorking_func(mMat_temp, nMat_temp)
        dvdnWorking = dvdnWorking_func(mMat_temp, nMat_temp)

        vWorker, prbs = calc_log_sum_choice_probs(
            [vWorking, vRetiring],
            self.TasteShkStd,
        )

        vWorkerNvrs = self.u.inv(vWorker)
        vWorkerNvrs[0, 0] = 0.0

        prbWorking, prbRetiring = prbs
        prbWorking[0, 0] = 0.0
        prbRetiring[0, 0] = 1.0

        mGrid_temp = np.append(0.0, self.mGrid)

        vWorkerNvrsFunc = BilinearInterp(vWorkerNvrs, mGrid_temp, self.nGrid)
        vWorkerFunc = ValueFuncCRRA(vWorkerNvrsFunc, self.CRRA)

        # if m = 0, work for sure is the limit
        prbRetiring = prbs[0]
        prbWorking = prbs[1]

        prbWorkingFunc = BilinearInterp(prbWorking, mGrid_temp, self.nGrid)
        prbRetiringFunc = BilinearInterp(prbRetiring, mGrid_temp, self.nGrid)

        # agent who is working and has no cash consumes 0
        cWorker = prbWorking * cWorking + prbWorking * cRetiring
        dWorker = prbWorking * dWorking

        cWorkerFunc = BilinearInterp(cWorker, mGrid_temp, self.nGrid)
        dWorkerFunc = BilinearInterp(dWorker, mGrid_temp, self.nGrid)

        # need to add an empty axis, value doesn't Matter because at m=0
        # agent retires with probability 1.0
        dvdmWorker = prbWorking * dvdmWorking + prbRetiring * vPRetiring
        dvdmWorkerNvrs = self.u.derinv(dvdmWorker)
        dvdmWorkerNvrsFunc = BilinearInterp(dvdmWorkerNvrs, mGrid_temp, self.nGrid)
        dvdmWorkerFunc = MargValueFuncCRRA(dvdmWorkerNvrsFunc, self.CRRA)

        dvdnWorker = prbWorking * dvdnWorking + prbRetiring * vPRetiring
        dvdnWorkerNvrs = self.u.derinv(dvdnWorker)
        dvdnWorkerNvrsFunc = BilinearInterp(dvdnWorkerNvrs, mGrid_temp, self.nGrid)
        dvdnWorkerFunc = MargValueFuncCRRA(dvdnWorkerNvrsFunc, self.CRRA)

        deposit_solution = DepositStage(
            c_func=cWorkerFunc,
            d_func=dWorkerFunc,
            dvdm_func=dvdmWorkerFunc,
            dvdn_func=dvdnWorkerFunc,
            v_func=vWorkerFunc,
        )

        probabilities = DiscreteChoiceProbabilities(
            prob_working=prbWorkingFunc,
            prob_retiring=prbRetiringFunc,
        )

        worker_solution = WorkerSolution(
            deposit_stage=deposit_solution,
            probabilities=probabilities,
        )

        return worker_solution

    def solve(self):
        retired_solution_next = self.solution_next.retired_solution
        worker_solution_next = self.solution_next.worker_solution

        self.retired_solution = self.solve_retired_problem(retired_solution_next)
        self.retiring_solution = self.solve_retiring_problem(self.retired_solution)
        self.working_solution = self.solve_working_problem(worker_solution_next)
        self.worker_solution = self.solve_worker_problem(
            self.working_solution,
            self.retiring_solution,
        )

        solution = RetirementSolution(
            worker_solution=self.worker_solution,
            retired_solution=self.retired_solution,
            working_solution=self.working_solution,
            retiring_solution=self.retiring_solution,
        )

        return solution


init_retirement_pension = init_portfolio.copy()
init_retirement_pension["RfreeA"] = 1.02
init_retirement_pension["RfreeB"] = 1.04
init_retirement_pension["DiscFac"] = 0.98
init_retirement_pension["CRRA"] = 2.0
init_retirement_pension["DisutilLabor"] = 0.25
init_retirement_pension["TaxDeduct"] = 0.10
init_retirement_pension["LivPrb"] = [1.0]
init_retirement_pension["PermGroFac"] = [1.0]
init_retirement_pension["TranShkStd"] = [0.10]
init_retirement_pension["TranShkCount"] = 7
init_retirement_pension["PermShkStd"] = [0.0]
init_retirement_pension["PermShkCount"] = 1
init_retirement_pension["UnempPrb"] = 0.0  # Prob of unemployment while working
init_retirement_pension["IncUnemp"] = 0.0
# Prob of unemployment while retired
init_retirement_pension["UnempPrbRet"] = 0.0
init_retirement_pension["IncUnempRet"] = 0.50
init_retirement_pension["TasteShkStd"] = 0.1

init_retirement_pension["epsilon"] = 1e-8

init_retirement_pension["aRetCount"] = 50
init_retirement_pension["aRetMax"] = 25.0
init_retirement_pension["aRetNestFac"] = 2

init_retirement_pension["mCount"] = 50
init_retirement_pension["mMax"] = 10
init_retirement_pension["mNestFac"] = 2

init_retirement_pension["nCount"] = 50
init_retirement_pension["nMax"] = 10
init_retirement_pension["nNestFac"] = 2

init_retirement_pension["lCount"] = 50
init_retirement_pension["lMax"] = 10
init_retirement_pension["lNestFac"] = 2

init_retirement_pension["blCount"] = 50
init_retirement_pension["blMax"] = 10
init_retirement_pension["blNestFac"] = 2

init_retirement_pension["aCount"] = 50
init_retirement_pension["aMax"] = 10
init_retirement_pension["aNestFac"] = 2

init_retirement_pension["bCount"] = 50
init_retirement_pension["bMax"] = 10
init_retirement_pension["bNestFac"] = 2
