# this code is outdated

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    utility,
    utility_inv,
    utility_invP,
    utilityP,
    utilityP_inv,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import init_portfolio
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.core import MetricObject, make_one_period_oo_solver
from HARK.distribution import DiscreteDistribution, calc_expectation
from HARK.fast import BilinearInterpFast, LinearInterpFast
from HARK.interpolation import (
    ConstantFunction,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
    calc_log_sum_choice_probs,
)
from HARK.utilities import NullFunc, make_grid_exp_mult
from scipy.interpolate import CloughTocher2DInterpolator


@dataclass
class PostDecisionStage(MetricObject):
    vFunc: ValueFuncCRRA = NullFunc()
    dvdaFunc: MargValueFuncCRRA = NullFunc()
    dvdbFunc: MargValueFuncCRRA = NullFunc()


@dataclass
class ConsumptionStage(MetricObject):
    cFunc: BilinearInterpFast = NullFunc()
    vFunc: ValueFuncCRRA = NullFunc()
    dvdlFunc: MargValueFuncCRRA = NullFunc()
    dvdbFunc: MargValueFuncCRRA = NullFunc()


@dataclass
class DepositStage(MetricObject):
    dFunc: BilinearInterpFast = NullFunc()
    cFunc: BilinearInterpFast = NullFunc()
    vFunc: ValueFuncCRRA = NullFunc()
    dvdmFunc: MargValueFuncCRRA = NullFunc()
    dvdnFunc: MargValueFuncCRRA = NullFunc()


@dataclass
class RetiredSolution(MetricObject):
    cFunc: MetricObject = NullFunc()
    vPfunc: MargValueFuncCRRA = NullFunc()
    vFunc: ValueFuncCRRA = NullFunc()
    vPEndOfPrdfunc: MargValueFuncCRRA = NullFunc()
    vEndOfPrdFunc: ValueFuncCRRA = NullFunc()


@dataclass
class RetiringSolution(MetricObject):
    cFunc: MetricObject = NullFunc()
    vPfunc: MargValueFuncCRRA = NullFunc()
    vFunc: ValueFuncCRRA = NullFunc()


@dataclass
class WorkingSolution(MetricObject):
    post_decision_stage: PostDecisionStage = PostDecisionStage()
    consumption_stage: ConsumptionStage = ConsumptionStage()
    deposit_stage: DepositStage = DepositStage()


@dataclass
class DiscreteChoiceProbabilities(MetricObject):
    prob_working: MetricObject = NullFunc()
    prob_retiring: MetricObject = NullFunc()


@dataclass
class WorkerSolution(MetricObject):
    deposit_stage: DepositStage = DepositStage()
    probabilities: DiscreteChoiceProbabilities = DiscreteChoiceProbabilities()


@dataclass
class RetPenContribSolution(MetricObject):
    worker_solution: WorkerSolution = WorkerSolution()
    retired_solution: RetiredSolution = RetiredSolution()
    working_solution: WorkingSolution = WorkingSolution()
    retiring_solution: RetiringSolution = RetiringSolution()


class PensionContribConsumerType(IndShockConsumerType):
    time_inv_ = deepcopy(IndShockConsumerType.time_inv_)
    time_inv_ = time_inv_ + [
        "DisutlLabr",
        "IncUnempRet",
        "TastShkStd",
        "RfreeA",
        "RfreeB",
        "TaxDedct",
    ]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_retirement_pension.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        RiskyAssetConsumerType.__init__(self, verbose=verbose, quiet=quiet, **params)

        # Add consumer-type specific objects, copying to create independent versions
        self.solve_one_period = make_one_period_oo_solver(PensionContribSolver)

        self.update()  # Make assets grid, income process, terminal solution

    def update(self):
        self.update_grids()
        IndShockConsumerType.update(self)

        # self.update_solution_terminal()

    def update_solution_terminal(self):
        # retired problem
        cGrid_ret = self.mRetGrid  # consume everything

        cFunc_terminal_ret = LinearInterpFast(self.mRetGrid, cGrid_ret)
        vPfunc_terminal_ret = MargValueFuncCRRA(cFunc_terminal_ret, self.CRRA)
        vFunc_terminal_ret = ValueFuncCRRA(cFunc_terminal_ret, self.CRRA)

        self.retired_solution = RetiredSolution(
            cFunc=cFunc_terminal_ret,
            vPfunc=vPfunc_terminal_ret,
            vFunc=vFunc_terminal_ret,
        )

        # working problem
        # the assumption here is that a working agent in the last period
        # is not actually working; #todo check if this is true in paper
        cMat_wrk = self.mMat + self.nMat  # consume everything
        cMat_wrk_temp = np.insert(cMat_wrk, 0, 0.0, axis=0)
        cFunc_terminal_wrk = BilinearInterpFast(
            cMat_wrk_temp, np.append(0.0, self.mGrid), self.nGrid
        )
        vPFunc_terminal_wrk = MargValueFuncCRRA(cFunc_terminal_wrk, self.CRRA)
        vFunc_terminal_wrk = ValueFuncCRRA(cFunc_terminal_wrk, self.CRRA)

        dFunc_terminal_wrk = ConstantFunction(0.0)

        deposit_stage = DepositStage(
            cFunc=cFunc_terminal_wrk,
            dFunc=dFunc_terminal_wrk,
            dvdmFunc=vPFunc_terminal_wrk,
            dvdnFunc=vPFunc_terminal_wrk,
            vFunc=vFunc_terminal_wrk,
        )

        self.working_solution = WorkingSolution(deposit_stage=deposit_stage)

        self.retiring_solution = RetiringSolution(
            cFunc=cFunc_terminal_wrk,
            vPfunc=vPFunc_terminal_wrk,
            vFunc=vFunc_terminal_wrk,
        )

        # because working and retiring are the same,
        # no need for  weighted average
        self.worker_solution = WorkerSolution(deposit_stage=deposit_stage)

        self.solution_terminal = RetPenContribSolution(
            worker_solution=self.worker_solution,
            retired_solution=self.retired_solution,
            working_solution=self.working_solution,
            retiring_solution=self.retiring_solution,
        )

    def update_grids(self):
        # retirement

        self.mRetGrid = make_grid_exp_mult(
            0.0, self.mRetMax, self.mRetCount, self.mRetNestFac
        )
        self.aRetGrid = make_grid_exp_mult(
            0.0, self.aRetMax, self.aRetCount, self.aRetNestFac
        )

        # worker grids
        self.mGrid = make_grid_exp_mult(
            self.epsilon, self.mMax, self.mCount, self.mNestFac
        )
        self.nGrid = make_grid_exp_mult(0.0, self.nMax, self.nCount, self.nNestFac)
        self.mMat, self.nMat = np.meshgrid(self.mGrid, self.nGrid, indexing="ij")

        # pure consumption grids
        self.aGrid = make_grid_exp_mult(0.0, self.aMax, self.aCount, self.aNestFac)
        self.bGrid = make_grid_exp_mult(0.0, self.bMax, self.bCount, self.bNestFac)
        self.aMat, self.bMat = np.meshgrid(self.aGrid, self.bGrid, indexing="ij")

        # pension deposit grids
        self.lGrid = make_grid_exp_mult(
            self.epsilon, self.lMax, self.lCount, self.lNestFac
        )
        self.blGrid = make_grid_exp_mult(0.0, self.blMax, self.blCount, self.blNestFac)
        self.lMat, self.blMat = np.meshgrid(self.lGrid, self.blGrid, indexing="ij")

        self.add_to_time_inv(
            "mRetGrid",
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
class PensionContribSolver(MetricObject):
    solution_next: RetPenContribSolution
    DiscFac: float
    CRRA: float
    DisutlLabr: float
    RfreeA: float
    RfreeB: float
    TaxDedct: float
    TranShkDstn: DiscreteDistribution
    IncUnempRet: float
    TastShkStd: float
    mRetGrid: np.array
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
        self.u = lambda x: utility(x, self.CRRA)
        self.uP = lambda x: utilityP(x, self.CRRA)
        self.uPinv = lambda x: utilityP_inv(x, self.CRRA)
        self.uinv = lambda x: utility_inv(x, self.CRRA)
        self.uinvP = lambda x: utility_invP(x, self.CRRA)

        # pension deposit function: tax deduction from pension deposits
        # which is gradually decreasing in the level of deposits

        self.g = lambda x: self.TaxDedct * np.log(1 + x)
        self.gP = lambda x: self.TaxDedct / (1 + x)
        self.gPinv = lambda x: self.TaxDedct / x - 1

    def solve_retired_problem(self, retired_solution_next):
        vPfunc_next = retired_solution_next.vPfunc
        vFunc_next = retired_solution_next.vFunc

        # for retired problem
        # as long as there is retired income there is no risk of m_next = 0
        # and agent won't hit borrowing constraint
        mGrid_next = self.aRetGrid * self.RfreeA + self.IncUnempRet
        vPEndOfPrd = self.DiscFac * self.RfreeA * vPfunc_next(mGrid_next)
        vPEndOfPrdNvrs = self.uPinv(vPEndOfPrd)
        vPEndOfPrdNvrsFunc = LinearInterpFast(self.aRetGrid, vPEndOfPrdNvrs)
        vPEndOfPrdFunc = MargValueFuncCRRA(vPEndOfPrdNvrsFunc, self.CRRA)

        cGrid = vPEndOfPrdNvrs  # endogenous grid method
        mGrid = cGrid + self.aRetGrid

        # need to add artificial borrowing constraint at 0.0
        cFunc = LinearInterpFast(np.append(0.0, mGrid), np.append(0.0, cGrid))
        vPfunc = MargValueFuncCRRA(cFunc, self.CRRA)

        # make retired value function
        # start by creating vEndOfPrdFunc
        vEndOfPrd = self.DiscFac * vFunc_next(mGrid_next)
        # value transformed through inverse utility
        vEndOfPrdNvrs = self.uinv(vEndOfPrd)
        vEndOfPrdNvrsFunc = LinearInterpFast(self.aRetGrid, vEndOfPrdNvrs)
        vEndOfPrdFunc = ValueFuncCRRA(vEndOfPrdNvrsFunc, self.CRRA)

        # calculate current value using mGrid that is consistent with agrid
        vGrid = self.u(cGrid) + vEndOfPrd
        # Construct the beginning-of-period value function
        vGridNvrs = self.uinv(vGrid)  # value transformed through inverse utility
        vGridNvrsFunc = LinearInterpFast(
            np.append(0.0, mGrid), np.append(0.0, vGridNvrs)
        )
        vFunc = ValueFuncCRRA(vGridNvrsFunc, self.CRRA)

        retired_solution = RetiredSolution(
            cFunc=cFunc,
            vPfunc=vPfunc,
            vFunc=vFunc,
            vPEndOfPrdfunc=vPEndOfPrdFunc,
            vEndOfPrdFunc=vEndOfPrdFunc,
        )

        return retired_solution

    def solve_retiring_problem(self, retired_solution):
        cFunc_retired = retired_solution.cFunc
        vFunc_retired = retired_solution.vFunc
        vPfunc_retired = retired_solution.vPfunc

        mMat_temp = np.insert(self.mMat, 0, 0.0, axis=0)
        nMat_temp = np.insert(self.nMat, 0, self.nGrid, axis=0)

        # Solve the retirement problem on exogenous MxN grid
        # to compute the value and policy functions 1 to 1
        cRetiring = cFunc_retired(mMat_temp + nMat_temp)
        vPRetiring = vPfunc_retired(mMat_temp + nMat_temp)
        vRetiring = vFunc_retired(mMat_temp + nMat_temp)

        # add when m = 0
        mGrid_temp = np.append(0.0, self.mGrid)

        cFunc = BilinearInterpFast(cRetiring, mGrid_temp, self.nGrid)
        vPfunc = MargValueFuncCRRA(cFunc, self.CRRA)

        vNvrs = self.uinv(vRetiring)
        vNvrsFunc = BilinearInterpFast(
            vNvrs,
            mGrid_temp,
            self.nGrid,
        )
        vFunc = ValueFuncCRRA(vNvrsFunc, self.CRRA)

        worker_retiring_solution = RetiringSolution(
            cFunc=cFunc, vPfunc=vPfunc, vFunc=vFunc
        )

        worker_retiring_solution.cRetiring = cRetiring
        worker_retiring_solution.vRetiring = vRetiring
        worker_retiring_solution.vPRetiring = vPRetiring

        return worker_retiring_solution

    def solve_post_decision_stage(self, deposit_stage_next):
        dvdmFunc_next = deposit_stage_next.dvdmFunc
        dvdnFunc_next = deposit_stage_next.dvdnFunc
        vFunc_next = deposit_stage_next.vFunc

        # First calculate marginal value functions
        def dvdaFunc(shock, abal, bbal):
            mnrm_next = self.RfreeA * abal + shock
            nnrm_next = self.RfreeB * bbal
            return dvdmFunc_next(mnrm_next, nnrm_next)

        dvda = (
            self.DiscFac
            * self.RfreeA
            * calc_expectation(self.TranShkDstn, dvdaFunc, self.aMat, self.bMat)
        )

        dvdaNvrs = self.uPinv(dvda)
        dvdaNvrsFunc = BilinearInterpFast(dvdaNvrs, self.aGrid, self.bGrid)
        dvdaEndOfPrdFunc = MargValueFuncCRRA(dvdaNvrsFunc, self.CRRA)

        def dvdbFunc(shock, abal, bbal):
            mnrm_next = self.RfreeA * abal + shock
            nnrm_next = self.RfreeB * bbal
            return dvdnFunc_next(mnrm_next, nnrm_next)

        dvdb = (
            self.DiscFac
            * self.RfreeB
            * calc_expectation(self.TranShkDstn, dvdbFunc, self.aMat, self.bMat)
        )

        dvdbNvrs = self.uPinv(dvdb)
        dvdbNvrsFunc = BilinearInterpFast(dvdbNvrs, self.aGrid, self.bGrid)
        dvdbFunc = MargValueFuncCRRA(dvdbNvrsFunc, self.CRRA)

        # also calculate end of period value function

        def vFunc(shock, abal, bbal):
            mnrm_next = self.RfreeA * abal + shock
            nnrm_next = self.RfreeB * bbal
            return vFunc_next(mnrm_next, nnrm_next)

        value = self.DiscFac * calc_expectation(
            self.TranShkDstn, vFunc, self.aMat, self.bMat
        )

        # value transformed through inverse utility
        vNvrs = self.uinv(value)
        vNvrsFunc = BilinearInterpFast(vNvrs, self.aGrid, self.bGrid)
        vFunc = ValueFuncCRRA(vNvrsFunc, self.CRRA)

        post_decision_stage = PostDecisionStage(
            vFunc=vFunc, dvdaFunc=dvdaEndOfPrdFunc, dvdbFunc=dvdbFunc
        )

        # sometimes the best items to pass to next stage aren't functions
        post_decision_stage.dvdaNvrs = dvdaNvrs
        post_decision_stage.dvdbNvrs = dvdbNvrs
        post_decision_stage.value = value

        return post_decision_stage

    def solve_consumption_stage(self, post_decision_stage):
        dvdaNvrs_next = post_decision_stage.dvdaNvrs
        dvdbNvrs_next = post_decision_stage.dvdbNvrs
        value_next = post_decision_stage.value

        cMat = dvdaNvrs_next  # endogenous grid method
        lMat = cMat + self.aMat

        # at l = 0, c = 0 so we need to add this limit
        lMat_temp = np.insert(lMat, 0, 0.0, axis=0)
        cMat_temp = np.insert(cMat, 0, 0.0, axis=0)

        # bMat is a regular grid, lMat is not so we'll need to use LinearInterpOnInterp1D
        cFunc_by_bbal = []
        for bi in range(self.bGrid.size):
            cFunc_by_bbal.append(LinearInterpFast(lMat_temp[:, bi], cMat_temp[:, bi]))

        cFunc = LinearInterpOnInterp1D(cFunc_by_bbal, self.bGrid)
        dvdlFunc = MargValueFuncCRRA(cFunc, self.CRRA)

        # again, at l = 0, c = 0 and a = 0, so repeat dvdb[0]
        dvdbNvrs_temp = np.insert(dvdbNvrs_next, 0, dvdbNvrs_next[0], axis=0)

        dvdbNvrsFunc_by_bbal = []
        for bi in range(self.bGrid.size):
            dvdbNvrsFunc_by_bbal.append(
                LinearInterpFast(lMat_temp[:, bi], dvdbNvrs_temp[:, bi])
            )

        dvdbFunc = MargValueFuncCRRA(
            LinearInterpOnInterp1D(dvdbNvrsFunc_by_bbal, self.bGrid), self.CRRA
        )

        # make value function
        value = self.u(cMat) - self.DisutlLabr + value_next
        vNvrs = self.uinv(value)
        vNvrs_temp = np.insert(vNvrs, 0, 0.0, axis=0)

        # bMat is regular grid so we can use LinearInterpOnInterp1D
        vNvrsFunc_by_bbal = []
        for bi in range(self.bGrid.size):
            vNvrsFunc_by_bbal.append(
                LinearInterpFast(lMat_temp[:, bi], vNvrs_temp[:, bi])
            )

        vNvrsFunc = LinearInterpOnInterp1D(vNvrsFunc_by_bbal, self.bGrid)
        vFunc = ValueFuncCRRA(vNvrsFunc, self.CRRA)

        consumption_stage = ConsumptionStage(
            cFunc=cFunc,
            vFunc=vFunc,
            dvdlFunc=dvdlFunc,
            dvdbFunc=dvdbFunc,
        )

        return consumption_stage

    def solve_deposit_stage(self, consumption_stage):
        cFunc_next = consumption_stage.cFunc
        vFunc_next = consumption_stage.vFunc
        dvdlFunc_next = consumption_stage.dvdlFunc
        dvdbFunc_next = consumption_stage.dvdbFunc

        dvdl_next = dvdlFunc_next(self.lMat, self.blMat)
        dvdb_next = dvdbFunc_next(self.lMat, self.blMat)

        # endogenous grid method
        dMat = self.gPinv(dvdl_next / dvdb_next - 1.0)

        mMat = self.lMat + dMat
        nMat = self.blMat - dMat - self.g(dMat)

        # remove nans and add anchoring point
        idx = np.isfinite(mMat + nMat + dMat)
        dGrid = np.append(0.0, dMat[idx])
        mGrid = np.append(0.0, mMat[idx])
        nGrid = np.append(0.0, nMat[idx])

        idx_lt = np.logical_or(nGrid < -1, mGrid < -1)
        idx_gt = np.logical_or(nGrid > 15, mGrid > 15)
        # idx_gt = False
        dGrid = dGrid[~np.logical_or(idx_lt, idx_gt)]
        mGrid = mGrid[~np.logical_or(idx_lt, idx_gt)]
        nGrid = nGrid[~np.logical_or(idx_lt, idx_gt)]

        dGrid = np.maximum(0.0, dGrid)

        # create interpolator
        linear_interp = CloughTocher2DInterpolator(list(zip(mGrid, nGrid)), dGrid)

        # evaluate d on common grid
        dMat = np.nan_to_num(linear_interp(self.mMat, self.nMat))
        dMat = np.maximum(0.0, dMat)
        lMat = self.mMat - dMat
        blMat = self.nMat + dMat + self.g(dMat)

        # evaluate c on common grid
        cMat = cFunc_next(lMat, blMat)
        # there is no consumption or deposit when there is no cash on hand
        mGrid_temp = np.append(0.0, self.mGrid)
        dMat_temp = np.insert(dMat, 0, 0.0, axis=0)
        cMat_temp = np.insert(cMat, 0, 0.0, axis=0)

        dFunc = BilinearInterpFast(dMat_temp, mGrid_temp, self.nGrid)
        cFunc = BilinearInterpFast(cMat_temp, mGrid_temp, self.nGrid)
        dvdmFunc = MargValueFuncCRRA(cFunc, self.CRRA)

        dvdb_next = dvdbFunc_next(lMat, blMat)

        dvdnNvrs = self.uPinv(dvdb_next)
        dvdnNvrs_temp = np.insert(dvdnNvrs, 0, dvdnNvrs[0], axis=0)
        dvdnNvrsFunc = BilinearInterpFast(dvdnNvrs_temp, mGrid_temp, self.nGrid)
        dvdnFunc = MargValueFuncCRRA(dvdnNvrsFunc, self.CRRA)

        # make value function
        value = vFunc_next(lMat, blMat)
        vNvrs = self.uinv(value)
        # insert value of 0 at m = 0
        vNvrs_temp = np.insert(vNvrs, 0, 0.0, axis=0)
        # mMat and nMat are irregular grids so we need Curvilinear2DInterp
        vNvrsFunc = BilinearInterpFast(vNvrs_temp, mGrid_temp, self.nGrid)
        vFunc = ValueFuncCRRA(vNvrsFunc, self.CRRA)

        deposit_stage = DepositStage(
            cFunc=cFunc,
            dFunc=dFunc,
            vFunc=vFunc,
            dvdmFunc=dvdmFunc,
            dvdnFunc=dvdnFunc,
        )

        deposit_stage.linear_interp = linear_interp

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
        vOutrFunc = working_solution.deposit_stage.vFunc
        cOutrFunc = working_solution.deposit_stage.cFunc
        dOutrFunc = working_solution.deposit_stage.dFunc
        dvdmOutrFunc = working_solution.deposit_stage.dvdmFunc
        dvdnOutrFunc = working_solution.deposit_stage.dvdnFunc

        vRetiring = retiring_solution.vRetiring
        cRetiring = retiring_solution.cRetiring
        vPRetiring = retiring_solution.vPRetiring

        vWorking = vOutrFunc(self.mMat, self.nMat)
        cWorking = cOutrFunc(self.mMat, self.nMat)
        dWorking = dOutrFunc(self.mMat, self.nMat)
        dvdmWorking = dvdmOutrFunc(self.mMat, self.nMat)
        dvdnWorking = dvdnOutrFunc(self.mMat, self.nMat)

        # plug in values when m = 0
        vWorking_temp = np.insert(vWorking, 0, -np.inf, axis=0)
        cWorking_temp = np.insert(cWorking, 0, 0.0, axis=0)
        dWorking_temp = np.insert(dWorking, 0, 0.0, axis=0)
        dvdmWorking_temp = np.insert(dvdmWorking, 0, -np.inf, axis=0)
        dvdnWorking_temp = np.insert(dvdnWorking, 0, -np.inf, axis=0)

        vWorker, prbs = calc_log_sum_choice_probs(
            np.stack((vWorking_temp, vRetiring)), self.TastShkStd
        )
        # at m, n = 0, the value is negative infinity regardless of the choice
        vWorker[0, 0] = -np.inf
        prbWorking = prbs[0]
        prbRetiring = prbs[1]
        # for continuity; the tie breaker is that at m, n = 0, the probability of
        # working is 0.0 and the probability of retiring is 1.0
        prbWorking[0, 0] = 0.0
        prbRetiring[0, 0] = 1.0

        vWorkerNvrs = self.uinv(vWorker)
        mGrid_temp = np.append(0.0, self.mGrid)

        vWorkerNvrsFunc = BilinearInterpFast(vWorkerNvrs, mGrid_temp, self.nGrid)
        vWorkerFunc = ValueFuncCRRA(vWorkerNvrsFunc, self.CRRA)

        prbWorking = prbs[0]
        prbRetiring = prbs[1]

        prbWorkingFunc = BilinearInterpFast(prbWorking, mGrid_temp, self.nGrid)
        prbRetiringFunc = BilinearInterpFast(prbRetiring, mGrid_temp, self.nGrid)

        # agent who is working and has no cash consumes 0
        cWorker = prbWorking * cWorking_temp + prbWorking * cRetiring
        dWorker = prbWorking * dWorking_temp

        cWorkerFunc = BilinearInterpFast(cWorker, mGrid_temp, self.nGrid)
        dWorkerFunc = BilinearInterpFast(dWorker, mGrid_temp, self.nGrid)

        # need to add an empty axis, value doesn't Matter because at m=0
        # agent retires with probability 1.0
        dvdmWorking[0] = 0.0  # does not Matter because retiring
        dvdmWorker = prbWorking * dvdmWorking_temp + prbRetiring * vPRetiring
        dvdmWorkerNvrs = self.uPinv(dvdmWorker)
        dvdmWorkerNvrsFunc = BilinearInterpFast(dvdmWorkerNvrs, mGrid_temp, self.nGrid)
        dvdmWorkerFunc = MargValueFuncCRRA(dvdmWorkerNvrsFunc, self.CRRA)

        dvdnWorker = prbWorking * dvdnWorking_temp + prbRetiring * vPRetiring
        dvdnWorkerNvrs = self.uPinv(dvdnWorker)
        dvdnWorkerNvrsFunc = BilinearInterpFast(dvdnWorkerNvrs, mGrid_temp, self.nGrid)
        dvdnWorkerFunc = MargValueFuncCRRA(dvdnWorkerNvrsFunc, self.CRRA)

        deposit_solution = DepositStage(
            cFunc=cWorkerFunc,
            dFunc=dWorkerFunc,
            dvdmFunc=dvdmWorkerFunc,
            dvdnFunc=dvdnWorkerFunc,
            vFunc=vWorkerFunc,
        )

        probabilities = DiscreteChoiceProbabilities(
            prob_working=prbWorkingFunc, prob_retiring=prbRetiringFunc
        )

        worker_solution = WorkerSolution(
            deposit_stage=deposit_solution, probabilities=probabilities
        )

        return worker_solution

    def solve(self):
        retired_solution_next = self.solution_next.retired_solution
        worker_solution_next = self.solution_next.worker_solution

        self.retired_solution = self.solve_retired_problem(retired_solution_next)
        self.retiring_solution = self.solve_retiring_problem(self.retired_solution)
        self.working_solution = self.solve_working_problem(worker_solution_next)
        self.worker_solution = self.solve_worker_problem(
            self.working_solution, self.retiring_solution
        )

        solution = RetPenContribSolution(
            worker_solution=self.worker_solution,
            retired_solution=self.retired_solution,
            working_solution=self.working_solution,
            retiring_solution=self.retiring_solution,
        )

        return solution


init_retirement_pension = init_portfolio.copy()
T_cycle = 19  # 19 solve cycles and 1 retirement cycle
init_retirement_pension["T_cycle"] = T_cycle
init_retirement_pension["T_age"] = T_cycle + 1
init_retirement_pension["RfreeA"] = 1.02
init_retirement_pension["RfreeB"] = 1.04
init_retirement_pension["DiscFac"] = 0.98
init_retirement_pension["CRRA"] = 2.0
init_retirement_pension["DisutlLabr"] = 0.25
init_retirement_pension["TaxDedct"] = 0.10
init_retirement_pension["LivPrb"] = [1.0] * T_cycle
init_retirement_pension["PermGroFac"] = [1.0] * T_cycle
init_retirement_pension["TranShkStd"] = [0.10] * T_cycle
init_retirement_pension["TranShkCount"] = 16
init_retirement_pension["PermShkStd"] = [0.0] * T_cycle
init_retirement_pension["PermShkCount"] = 1
init_retirement_pension["UnempPrb"] = 0.0  # Prob of unemployment while working
init_retirement_pension["IncUnemp"] = 0.0
# Prob of unemployment while retired
init_retirement_pension["UnempPrbRet"] = 0.0
init_retirement_pension["IncUnempRet"] = 0.50
init_retirement_pension["TastShkStd"] = 0.1

init_retirement_pension["epsilon"] = 1e-6

init_retirement_pension["mRetCount"] = 600
init_retirement_pension["mRetMax"] = 50.0
init_retirement_pension["mRetNestFac"] = 2

init_retirement_pension["aRetCount"] = 600
init_retirement_pension["aRetMax"] = 25.0
init_retirement_pension["aRetNestFac"] = 2

init_retirement_pension["mCount"] = 600
init_retirement_pension["mMax"] = 10.0
init_retirement_pension["mNestFac"] = 2

init_retirement_pension["nCount"] = 600
init_retirement_pension["nMax"] = 12.0
init_retirement_pension["nNestFac"] = 2

init_retirement_pension["lCount"] = 600
init_retirement_pension["lMax"] = 20
init_retirement_pension["lNestFac"] = 1

init_retirement_pension["blCount"] = 600
init_retirement_pension["blMax"] = 20
init_retirement_pension["blNestFac"] = 1

init_retirement_pension["aCount"] = 600
init_retirement_pension["aMax"] = 20
init_retirement_pension["aNestFac"] = 2

init_retirement_pension["bCount"] = 600
init_retirement_pension["bMax"] = 20
init_retirement_pension["bNestFac"] = 2
