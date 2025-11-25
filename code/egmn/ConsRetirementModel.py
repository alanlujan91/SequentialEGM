from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from HARK.ConsumptionSaving.ConsPortfolioModel import init_portfolio
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.core import make_one_period_oo_solver
from HARK.distributions import DiscreteDistribution, calc_expectation
from HARK.interpolation import (
    BilinearInterp,
    LinearInterp,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
    calc_log_sum_choice_probs,
)

# Note: GeneralizedRegressionUnstructuredInterp (GPR wrapper) removed from HARK master
# Replacement available: from egmn.gpr_interp import UnstructuredInterpGPR
# Currently using scipy.interpolate.LinearNDInterpolator as fast fallback
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

        # At l=0, c=0, a=0, so evaluate dvdb at (a=0, b) for each b
        # This equals ∂v²/∂b(0, b) from post-decision stage
        dvdb_at_a0 = post_decision_stage.dvdb_func(
            np.zeros_like(self.bGrid), self.bGrid
        )
        dvdb_at_a0_nvrs = self.u.derinv(dvdb_at_a0)
        dvdb_nvrs_temp = np.insert(dvdb_nvrs_next, 0, dvdb_at_a0_nvrs, axis=0)
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

        # Check for problems in marginal value functions
        ratio = dvdl_next / dvdb_next
        if np.any(ratio < 1.0) or np.any(np.isnan(ratio)) or np.any(np.isinf(ratio)):
            n_bad = np.sum((ratio < 1.0) | np.isnan(ratio) | np.isinf(ratio))
            if n_bad > 0.9 * ratio.size:  # >90% bad
                print(f"  CRITICAL: {n_bad}/{ratio.size} bad ratios dvdl/dvdb")
                print(
                    f"    dvdl range: [{np.nanmin(dvdl_next):.2e}, {np.nanmax(dvdl_next):.2e}]"
                )
                print(
                    f"    dvdb range: [{np.nanmin(dvdb_next):.2e}, {np.nanmax(dvdb_next):.2e}]"
                )
                print(
                    f"    ratio range: [{np.nanmin(ratio):.2e}, {np.nanmax(ratio):.2e}]"
                )
                print(f"    lMat range: [{self.lMat.min():.2e}, {self.lMat.max():.2e}]")
                print(
                    f"    blMat range: [{self.blMat.min():.2e}, {self.blMat.max():.2e}]"
                )

        # endogenous grid method
        dMat = self.g.derinv(dvdl_next / dvdb_next - 1.0)

        mMat = self.lMat + dMat
        nMat = self.blMat - dMat - self.g(dMat)

        # Note: Can upgrade to UnstructuredInterpGPR for higher accuracy on warped grids
        # Currently using BilinearInterp as fast fallback
        gaussian_interp_grid0 = BilinearInterp(self.lMat, mMat[0, :], nMat[:, 0])
        gaussian_interp_grid1 = BilinearInterp(self.blMat, mMat[0, :], nMat[:, 0])

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

        # Filter NaN and degenerate points from Sequential EGM
        # (Some grid points may produce invalid solutions that need to be excluded)
        valid = ~np.isnan(dMat) & ~np.isnan(mMat) & ~np.isnan(nMat) & (dMat > -1.0)
        dMat = dMat[valid]
        mMat = mMat[valid]
        nMat = nMat[valid]

        # For long horizons, use fast LinearND interpolation with clipped extrapolation
        # (GPR is too slow for T>5: 3 GPRs/period × T periods = 3T GPR fits!)
        if len(mMat) < 10:
            # Degenerate case: use simple fallback
            print(
                f"WARNING: Only {len(mMat)} valid points. Using zero deposit fallback."
            )
            m_min, m_max = self.mMat.min(), self.mMat.max()
            n_min, n_max = self.nMat.min(), self.nMat.max()
            mMat = np.array([m_min, m_max, m_min, m_max])
            nMat = np.array([n_min, n_min, n_max, n_max])
            dMat = np.zeros(4)

        gaussian_interp = LinearNDInterpolator(
            np.column_stack([mMat, nMat]), dMat, fill_value=0.0
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

        # At m=0, d=0, we have l=0, so evaluate dvdb at (l=0, b) for each b
        # This equals ∂v¹/∂b(0, n) = ∂v⁰/∂n(0, n) by envelope
        dvdb_at_m0 = dvdb_func_next(np.zeros_like(self.nGrid), self.nGrid)
        dvdn_at_m0_nvrs = self.u.derinv(dvdb_at_m0)

        dvdn_nvrs_temp = np.insert(dvdn_nvrs, 0, dvdn_at_m0_nvrs, axis=0)
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


# =====================================================
# Constructor Functions
# =====================================================


def make_retirement_grids(
    epsilon,
    aRetCount,
    aRetMax,
    aRetNestFac,
    aCount,
    aMax,
    aNestFac,
    bCount,
    bMax,
    bNestFac,
    lCount,
    lMax,
    lNestFac,
    blCount,
    blMax,
    blNestFac,
    mCount,
    mMax,
    mNestFac,
    nCount,
    nMax,
    nNestFac,
):
    """
    Constructs grids for retirement model.

    Parameters
    ----------
    epsilon : float
        Small number to avoid zero in logs
    aRetCount, aRetMax, aRetNestFac : int, float, int
        Grid parameters for retirement assets
    aCount, aMax, aNestFac : int, float, int
        Grid parameters for post-decision assets a
    bCount, bMax, bNestFac : int, float, int
        Grid parameters for post-decision assets b
    lCount, lMax, lNestFac : int, float, int
        Grid parameters for labor income l
    blCount, blMax, blNestFac : int, float, int
        Grid parameters for combined (b, l) grid
    mCount, mMax, mNestFac : int, float, int
        Grid parameters for market resources m
    nCount, nMax, nNestFac : int, float, int
        Grid parameters for pension assets n

    Returns
    -------
    dict
        Dictionary containing all grids
    """
    # retirement
    aRetGrid = make_grid_exp_mult(0.0, aRetMax, aRetCount, aRetNestFac)

    # post decision grids for consumption stage
    aGrid = make_grid_exp_mult(0.0, aMax, aCount, aNestFac)
    bGrid = make_grid_exp_mult(0.0, bMax, bCount, bNestFac)
    aMat, bMat = np.meshgrid(aGrid, bGrid, indexing="ij")

    # exogenous grids for pension deposit stage
    lGrid = make_grid_exp_mult(epsilon, lMax, lCount, lNestFac)
    blGrid = make_grid_exp_mult(0.0, blMax, blCount, blNestFac)
    lMat, blMat = np.meshgrid(lGrid, blGrid, indexing="ij")

    # common worker grids
    mGrid = make_grid_exp_mult(epsilon, mMax, mCount, mNestFac)
    nGrid = make_grid_exp_mult(0.0, nMax, nCount, nNestFac)
    mMat, nMat = np.meshgrid(mGrid, nGrid, indexing="ij")

    return {
        "aRetGrid": aRetGrid,
        "aGrid": aGrid,
        "bGrid": bGrid,
        "aMat": aMat,
        "bMat": bMat,
        "lGrid": lGrid,
        "blGrid": blGrid,
        "lMat": lMat,
        "blMat": blMat,
        "mGrid": mGrid,
        "nGrid": nGrid,
        "mMat": mMat,
        "nMat": nMat,
    }


def make_retirement_solution_terminal(CRRA):
    """
    Constructs the terminal period solution for retirement model.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion

    Returns
    -------
    RetirementSolution
        Terminal period solution object
    """

    # retired problem - consume everything
    def c_func_retired(m):
        return m

    retired_solution = RetiredSolution(
        c_func=c_func_retired,
        vp_func=MargValueFuncCRRA(c_func_retired, CRRA),
        v_func=ValueFuncCRRA(c_func_retired, CRRA),
    )

    # retiring - consume everything including pension
    def c_func(m, n):
        return m + n

    vp_func = MargValueFuncCRRA(c_func, CRRA)
    v_func = ValueFuncCRRA(c_func, CRRA)

    retiring_solution = RetiringSolution(
        c_func=c_func,
        vp_func=vp_func,
        v_func=v_func,
    )

    # terminal working solution - deposit nothing
    def d_func(m, n):
        return m * 0.0

    deposit_stage = DepositStage(
        c_func=c_func,
        d_func=d_func,
        dvdm_func=vp_func,
        dvdn_func=vp_func,
        v_func=v_func,
    )

    working_solution = WorkingSolution(deposit_stage=deposit_stage)

    # worker solution (same as working for terminal)
    worker_solution = WorkerSolution(deposit_stage=deposit_stage)

    return RetirementSolution(
        worker_solution=worker_solution,
        retired_solution=retired_solution,
        working_solution=working_solution,
        retiring_solution=retiring_solution,
    )


# =====================================================
# Parameter Dictionary
# =====================================================

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

# Build grids directly (not via constructors since they're static)
grids = make_retirement_grids(
    epsilon=init_retirement_pension["epsilon"],
    aRetCount=init_retirement_pension["aRetCount"],
    aRetMax=init_retirement_pension["aRetMax"],
    aRetNestFac=init_retirement_pension["aRetNestFac"],
    aCount=init_retirement_pension["aCount"],
    aMax=init_retirement_pension["aMax"],
    aNestFac=init_retirement_pension["aNestFac"],
    bCount=init_retirement_pension["bCount"],
    bMax=init_retirement_pension["bMax"],
    bNestFac=init_retirement_pension["bNestFac"],
    lCount=init_retirement_pension["lCount"],
    lMax=init_retirement_pension["lMax"],
    lNestFac=init_retirement_pension["lNestFac"],
    blCount=init_retirement_pension["blCount"],
    blMax=init_retirement_pension["blMax"],
    blNestFac=init_retirement_pension["blNestFac"],
    mCount=init_retirement_pension["mCount"],
    mMax=init_retirement_pension["mMax"],
    mNestFac=init_retirement_pension["mNestFac"],
    nCount=init_retirement_pension["nCount"],
    nMax=init_retirement_pension["nMax"],
    nNestFac=init_retirement_pension["nNestFac"],
)
init_retirement_pension.update(grids)

# Add terminal solution constructor
if "constructors" not in init_retirement_pension:
    init_retirement_pension["constructors"] = {}

init_retirement_pension["constructors"]["solution_terminal"] = (
    make_retirement_solution_terminal
)


# =====================================================
# Agent Type Definition
# =====================================================


class RetirementConsumerType(RiskyAssetConsumerType):
    """
    Consumer with retirement choice: can work or retire, with separate pension account.
    When working, agent chooses how much to deposit into a tax-advantaged pension account.
    Agent faces discrete choice each period of whether to work or retire (absorbing state).
    """

    time_inv_ = deepcopy(RiskyAssetConsumerType.time_inv_)
    time_inv_ = time_inv_ + [
        "DisutilLabor",
        "IncUnempRet",
        "TasteShkStd",
        "RfreeA",
        "RfreeB",
        "TaxDeduct",
        # Grid-related parameters
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
    ]

    default_ = {
        "params": init_retirement_pension,
        "solver": make_one_period_oo_solver(RetirementSolver),
    }
