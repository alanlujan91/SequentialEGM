from __future__ import annotations

from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field

import estimagic as em
import numpy as np

# Using UnstructuredInterpGPR instead of scipy interpolators (paper focuses on GPR)
from scipy.optimize import Bounds, LinearConstraint, minimize

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.LegacyOOsolvers import ConsIndShockSolver
from HARK.ConsumptionSaving.ConsPortfolioModel import init_portfolio
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.core import make_one_period_oo_solver
from HARK.distributions import DiscreteDistribution, DiscreteDistributionLabeled
from HARK.econforgeinterp import LinearFast
from HARK.interpolation import (
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)

# Note: GeneralizedRegressionUnstructuredInterp (GPR wrapper) removed from HARK master
# Replacement available: from egmn.gpr_interp import UnstructuredInterpGPR
# Currently using scipy.interpolate.LinearNDInterpolator as fast fallback
from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA, UtilityFunction
from HARK.utilities import NullFunc, make_assets_grid


@dataclass
class PostDecisionStage(MetricObject):
    v_func: ValueFuncCRRA = NullFunc()
    dvda_func: MargValueFuncCRRA = NullFunc()
    dvdb_func: MargValueFuncCRRA = NullFunc()


@dataclass
class ConsumptionStage(MetricObject):
    c_func: LinearFast = NullFunc()
    v_func: ValueFuncCRRA = NullFunc()
    dvdl_func: MargValueFuncCRRA = NullFunc()
    dvdb_func: MargValueFuncCRRA = NullFunc()


@dataclass
class DepositStage(MetricObject):
    d_func: LinearFast = NullFunc()
    v_func: ValueFuncCRRA = NullFunc()
    dvdm_func: MargValueFuncCRRA = NullFunc()
    dvdn_func: MargValueFuncCRRA = NullFunc()


@dataclass
class PensionSolution(MetricObject):
    post_decision_stage: PostDecisionStage = field(default_factory=PostDecisionStage)
    deposit_stage: DepositStage = field(default_factory=DepositStage)
    consumption_stage: ConsumptionStage = field(default_factory=ConsumptionStage)


GridParameters = namedtuple(
    "GridParameters",
    "aXtraMin, aXtraMax, aXtraCount, aXtraNestFac, aXtraExtra",
    defaults=[[]],
)


# Class definition moved to end of file after constructors and init dict


@dataclass
class PensionSolver(MetricObject):
    solution_next: PensionSolution
    DiscFac: float
    CRRA: float
    DisutilLabor: float
    Rfree: float
    TaxDeduct: float
    TranShkDstn: DiscreteDistribution
    IncUnempRet: DiscreteDistribution
    TasteShkStd: DiscreteDistribution
    ShockDstn: DiscreteDistributionLabeled
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

        def g(x):
            return self.TaxDeduct * np.log(1 + x)

        def gp(x):
            return self.TaxDeduct / (1 + x)

        def gp_inv(x):
            return self.TaxDeduct / x - 1

        self.g = UtilityFunction(g, gp, gp_inv)

    def solve_post_decision(self, deposit_stage_next):
        """Should use aMat and bMat.

        Parameters
        ----------
        deposit_stage_next : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        """
        # unpack next period's solution
        dvdm_func_next = deposit_stage_next.dvdm_func
        dvdn_func_next = deposit_stage_next.dvdn_func
        v_func_next = deposit_stage_next.v_func

        def value_and_marginal_funcs(shock, aBal, bBal):
            variables = {}

            psi = shock["PermShk"]
            mNrm_next = aBal * self.Rfree / psi + shock["TranShk"]
            nNrm_next = bBal * shock["Risky"] / psi

            variables["dvda"] = (
                self.DiscFac
                * self.Rfree
                * psi ** (-self.CRRA)
                * dvdm_func_next(mNrm_next, nNrm_next)
            )
            variables["dvdb"] = (
                self.DiscFac
                * psi ** (-self.CRRA)
                * shock["Risky"]
                * dvdn_func_next(mNrm_next, nNrm_next)
            )
            variables["v"] = (
                self.DiscFac
                * psi ** (1 - self.CRRA)
                * v_func_next(mNrm_next, nNrm_next)
            )

            return variables

        # First calculate marginal value functions

        def dvda_func(shock, aBal, bBal):
            psi = shock["PermShk"]
            mNrm_next = aBal * self.Rfree / psi + shock["TranShk"]
            nNrm_next = bBal * shock["Risky"] / psi
            return psi ** (-self.CRRA) * dvdm_func_next(mNrm_next, nNrm_next)

        dvda_end_of_prd = (
            self.DiscFac
            * self.Rfree
            * self.ShockDstn.expected(dvda_func, self.aMat, self.bMat)
        )

        dvda_end_of_prd_nvrs = self.u.derinv(dvda_end_of_prd)
        dvda_end_of_prd_nvrs_func = LinearFast(
            dvda_end_of_prd_nvrs,
            [self.aGrid, self.bGrid],
        )
        dvda_end_of_prd_func = MargValueFuncCRRA(dvda_end_of_prd_nvrs_func, self.CRRA)

        def dvdb_func(shock, aBal, bBal):
            psi = shock["PermShk"]
            mNrm_next = aBal * self.Rfree / psi + shock["TranShk"]
            nNrm_next = bBal * shock["Risky"] / psi
            return (
                psi ** (-self.CRRA)
                * shock["Risky"]
                * dvdn_func_next(mNrm_next, nNrm_next)
            )

        dvdb_end_of_prd = self.DiscFac * self.ShockDstn.expected(
            dvdb_func,
            self.aMat,
            self.bMat,
        )

        dvdb_end_of_prd_nvrs = self.u.derinv(dvdb_end_of_prd)
        dvdb_end_of_prd_nvrs_func = LinearFast(
            dvdb_end_of_prd_nvrs,
            [self.aGrid, self.bGrid],
        )
        dvdb_end_of_prd_func = MargValueFuncCRRA(dvdb_end_of_prd_nvrs_func, self.CRRA)

        # also calculate end of period value function

        def v_func(shock, aBal, bBal):
            psi = shock["PermShk"]
            mNrm_next = aBal * self.Rfree / psi + shock["TranShk"]
            nNrm_next = bBal * shock["Risky"] / psi
            return psi ** (1 - self.CRRA) * v_func_next(mNrm_next, nNrm_next)

        v_end_of_prd = self.DiscFac * self.ShockDstn.expected(
            v_func,
            self.aMat,
            self.bMat,
        )

        # value transformed through inverse utility
        v_end_of_prd_nvrs = self.u.inv(v_end_of_prd)
        v_end_of_prd_nvrs_func = LinearFast(v_end_of_prd_nvrs, [self.aGrid, self.bGrid])
        v_end_of_prd_func = ValueFuncCRRA(v_end_of_prd_nvrs_func, self.CRRA)

        post_decision_stage = PostDecisionStage(
            v_func=v_end_of_prd_func,
            dvda_func=dvda_end_of_prd_func,
            dvdb_func=dvdb_end_of_prd_func,
        )
        post_decision_stage.dvda_nvrs = dvda_end_of_prd_nvrs
        post_decision_stage.dvdb_nvrs = dvdb_end_of_prd_nvrs
        post_decision_stage.vals = v_end_of_prd

        return post_decision_stage

    def solve_consumption_decision(self, post_decision_stage):
        """Should use lMat and blMat

        Parameters
        ----------
        post_decision_stage : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        """
        dvda_end_of_prd_nvrs = post_decision_stage.dvda_nvrs
        dvdb_end_of_prd_nvrs = post_decision_stage.dvdb_nvrs
        v_end_of_prd = post_decision_stage.vals

        cMat = dvda_end_of_prd_nvrs  # endogenous grid method
        lMat = cMat + self.aMat

        # at l = 0, c = 0 so we need to add this limit
        lMat_temp = np.insert(lMat, 0, 0.0, axis=0)
        cMat_temp = np.insert(cMat, 0, 0.0, axis=0)

        # bMat is a regular grid, lMat is not so we'll need to use LinearInterpOnInterp1D
        c_innr_func_by_bBal = []
        for bi in range(self.bGrid.size):
            c_innr_func_by_bBal.append(LinearFast(cMat_temp[:, bi], [lMat_temp[:, bi]]))

        c_innr_func = LinearInterpOnInterp1D(c_innr_func_by_bBal, self.bGrid)
        dvdl_innr_func = MargValueFuncCRRA(c_innr_func, self.CRRA)

        # again, at l = 0, c = 0 and a = 0, so repeat dvdb[0]
        # dvdb_end_of_prd_nvrs_temp = np.insert(dvdb_end_of_prd_nvrs, 0, 0.0, axis=0)
        dvdb_end_of_prd_nvrs_temp = np.insert(
            dvdb_end_of_prd_nvrs,
            0,
            dvdb_end_of_prd_nvrs[0],
            axis=0,
        )

        dvdb_innr_nvrs_func_by_bBal = []
        for bi in range(self.bGrid.size):
            dvdb_innr_nvrs_func_by_bBal.append(
                LinearFast(dvdb_end_of_prd_nvrs_temp[:, bi], [lMat_temp[:, bi]]),
            )

        dvdb_innr_func = MargValueFuncCRRA(
            LinearInterpOnInterp1D(dvdb_innr_nvrs_func_by_bBal, self.bGrid),
            self.CRRA,
        )

        # make value function
        v_innr = self.u(cMat) + v_end_of_prd
        v_innr_nvrs = self.u.inv(v_innr)
        v_now_nvrs_temp = np.insert(v_innr_nvrs, 0, 0.0, axis=0)

        # bMat is regular grid so we can use LinearInterpOnInterp1D
        v_innr_nvrs_func_by_bBal = []
        for bi in range(self.bGrid.size):
            v_innr_nvrs_func_by_bBal.append(
                LinearFast(v_now_nvrs_temp[:, bi], [lMat_temp[:, bi]]),
            )

        v_innr_nvrs_func = LinearInterpOnInterp1D(v_innr_nvrs_func_by_bBal, self.bGrid)
        v_innr_func = ValueFuncCRRA(v_innr_nvrs_func, self.CRRA)

        consumption_stage = ConsumptionStage(
            c_func=c_innr_func,
            v_func=v_innr_func,
            dvdl_func=dvdl_innr_func,
            dvdb_func=dvdb_innr_func,
        )

        return consumption_stage

    def solve_deposit_decision(self, consumption_stage):
        c_func_next = consumption_stage.c_func
        v_func_next = consumption_stage.v_func
        dvdl_func_next = consumption_stage.dvdl_func
        dvdb_func_next = consumption_stage.dvdb_func

        dvdl_next = dvdl_func_next(self.lMat, self.blMat)
        dvdb_next = dvdb_func_next(self.lMat, self.blMat)

        # endogenous grid method, again
        dMat = self.g.inv(dvdl_next / dvdb_next - 1.0)

        mMat = self.lMat + dMat
        nMat = self.blMat - dMat - self.g(dMat)

        consumption_stage.grids_before_cleanup = {
            "dMat": dMat,
            "mMat": mMat,
            "nMat": nMat,
            "lMat": self.lMat,
            "blMat": self.blMat,
        }

        # Filter NaN values from Sequential EGM before interpolation
        # This is critical as EGM on rectilinear grids produces degenerate points
        mMat_flat = mMat.flatten()
        nMat_flat = nMat.flatten()
        lMat_flat = self.lMat.flatten()
        blMat_flat = self.blMat.flatten()

        # Filter for first interpolator (l values)
        valid_grid0 = (
            ~np.isnan(mMat_flat)
            & ~np.isnan(nMat_flat)
            & ~np.isnan(lMat_flat)
            & np.isfinite(mMat_flat)
            & np.isfinite(nMat_flat)
            & np.isfinite(lMat_flat)
        )

        # Filter for second interpolator (bl values)
        valid_grid1 = (
            ~np.isnan(mMat_flat)
            & ~np.isnan(nMat_flat)
            & ~np.isnan(blMat_flat)
            & np.isfinite(mMat_flat)
            & np.isfinite(nMat_flat)
            & np.isfinite(blMat_flat)
        )

        # Use UnstructuredInterpGPR - paper focuses on GPR interpolation
        from egmn.gpr_interp import UnstructuredInterpGPR

        gaussian_interp_grid0 = UnstructuredInterpGPR(
            np.column_stack([mMat_flat[valid_grid0], nMat_flat[valid_grid0]]),
            lMat_flat[valid_grid0],
            alpha=1e-6,
            normalize_y=True,
        )
        gaussian_interp_grid1 = UnstructuredInterpGPR(
            np.column_stack([mMat_flat[valid_grid1], nMat_flat[valid_grid1]]),
            blMat_flat[valid_grid1],
            alpha=1e-6,
            normalize_y=True,
        )

        # Interpolate grids - replace any NaN in input grids with valid values
        # (extrapolation at boundaries may produce NaN)
        mMat_query = np.nan_to_num(self.mMat, nan=0.0)
        nMat_query = np.nan_to_num(self.nMat, nan=0.0)

        lMat_temp = gaussian_interp_grid0(mMat_query, nMat_query)
        blMat_temp = gaussian_interp_grid1(mMat_query, nMat_query)

        # calculate derivatives
        dvdl_next = dvdl_func_next(lMat_temp, blMat_temp)
        dvdb_next = dvdb_func_next(lMat_temp, blMat_temp)

        # endogenous grid method
        dMat2 = self.g.inv(dvdl_next / dvdb_next - 1.0)
        mMat2 = lMat_temp + dMat2
        nMat2 = blMat_temp - dMat2 - self.g(dMat2)

        # concatenate grids
        dMat = np.concatenate((dMat.flatten(), dMat2.flatten()))
        mMat = np.concatenate((mMat.flatten(), mMat2.flatten()))
        nMat = np.concatenate((nMat.flatten(), nMat2.flatten()))

        # Filter degenerate points from Sequential EGM:
        # 1. Remove NaN values (infeasible/extreme states from EGM inversion)
        # 2. Remove economically unreasonable values (dMat <= -1.0)
        valid = ~np.isnan(dMat) & ~np.isnan(mMat) & ~np.isnan(nMat) & (dMat > -1.0)
        dMat = dMat[valid]
        mMat = mMat[valid]
        nMat = nMat[valid]

        # Use UnstructuredInterpGPR for higher accuracy and visualization support
        # (has .grids and .values attributes needed for plot_scatter_hist)
        from egmn.gpr_interp import UnstructuredInterpGPR

        gaussian_interp = UnstructuredInterpGPR(
            np.column_stack([mMat, nMat]),
            dMat,
            alpha=1e-6,  # Small noise for numerical stability
            normalize_y=True,  # Normalize for better numerical stability
        )

        # evaluate d on common grid
        dMat = gaussian_interp(self.mMat, self.nMat)
        # Project negative deposits to zero (standard EGM constraint handling)
        # This enforces d >= 0 but doesn't rigorously check Kuhn-Tucker conditions
        # More rigorous: DCEGM (Iskhakov et al. 2017) would handle upper/lower envelopes
        dMat = np.maximum(0.0, dMat)
        lMat = self.mMat - dMat
        blMat = self.nMat + dMat + self.g(dMat)

        # evaluate c on common grid
        cMat = c_func_next(lMat, blMat)
        # there is no consumption or deposit when there is no cash on hand
        mGrid_temp = np.append(0.0, self.mGrid)
        dMat_temp = np.insert(dMat, 0, 0.0, axis=0)
        cMat_temp = np.insert(cMat, 0, 0.0, axis=0)

        d_outr_func = LinearFast(dMat_temp, [mGrid_temp, self.nGrid])
        c_outr_func = LinearFast(cMat_temp, [mGrid_temp, self.nGrid])
        dvdm_outr_func = MargValueFuncCRRA(c_outr_func, self.CRRA)

        dvdb_next = dvdb_func_next(lMat, blMat)

        dvdn_outr_nvrs = self.u.derinv(dvdb_next)

        # At m=0, d=0, we have l=0, a=0, so evaluate dvdb at (l=0, b) for each b
        # This equals ∂v²/∂b(a=0, b) by envelope condition
        # This is the correct marginal value at the state space boundary
        dvdb_at_m0 = dvdb_func_next(np.zeros_like(self.nGrid), self.nGrid)
        dvdn_at_m0_nvrs = self.u.derinv(dvdb_at_m0)

        dvdn_outr_nvrs_temp = np.insert(dvdn_outr_nvrs, 0, dvdn_at_m0_nvrs, axis=0)
        dvdn_outr_nvrs_func = LinearFast(dvdn_outr_nvrs_temp, [mGrid_temp, self.nGrid])
        dvdn_outr_func = MargValueFuncCRRA(dvdn_outr_nvrs_func, self.CRRA)

        # make value function
        v_outr = v_func_next(lMat, blMat)
        v_outr_nvrs = self.u.inv(v_outr)
        # insert value of 0 at m = 0
        v_outr_nvrs_temp = np.insert(v_outr_nvrs, 0, 0.0, axis=0)
        # mMatand nMatare irregular grids so we need Curvilinear2DInterp
        v_now_nvrs_func = LinearFast(v_outr_nvrs_temp, [mGrid_temp, self.nGrid])
        v_outr_func = ValueFuncCRRA(v_now_nvrs_func, self.CRRA)

        deposit_stage = DepositStage(
            d_func=d_outr_func,
            v_func=v_outr_func,
            dvdm_func=dvdm_outr_func,
            dvdn_func=dvdn_outr_func,
        )

        deposit_stage.c_func = c_outr_func
        deposit_stage.gaussian_interp = gaussian_interp

        return deposit_stage

    def solve_deposit_decision_vfi(self, consumption_stage):
        v_func_next = consumption_stage.v_func

        def objective(d_nrm, m_nrm=None, n_nrm=None):
            l_nrm = m_nrm - d_nrm
            b_nrm = n_nrm + d_nrm + self.g(d_nrm)

            v_func = v_func_next(l_nrm, b_nrm)

            output = {}

            output["contributions"] = v_func
            output["value"] = np.sum(v_func)

            return output

        res = em.maximize(
            objective,
            params=self.mMat / 2,
            criterion_kwargs={"m_nrm": self.mMat, "n_nrm": self.nMat},
            algorithm="scipy_lbfgsb",
            numdiff_options={"n_cores": 6},
            multistart=True,
            lower_bounds=np.zeros_like(self.mMat),
            upper_bounds=self.mMat,
        )

        dMat = res.params

        # for mi in range(self.mGrid.size):
        #     for ni in range(self.nGrid.size):
        #         m_nrm = self.mGrid[mi]
        #         n_nrm = self.nGrid[ni]
        #         res = minimize_scalar(
        #             objective,
        #             args=(
        #                 m_nrm,
        #                 n_nrm,
        #             ),
        #             bounds=(0, m_nrm),
        #             method="bounded",
        #         )
        #         dMat[mi, ni] = res.x

        # add d = 0 when no liquid cash
        dMat_temp = np.insert(dMat, 0, 0.0, axis=0)

        d_func = LinearFast(dMat_temp, [np.append(0.0, self.mGrid), self.nGrid])

        deposit_stage = DepositStage(
            d_func=d_func,
        )
        deposit_stage.extras = {
            "d_nrm": dMat,
            "m_nrm": self.mMat,
            "n_nrm": self.nMat,
            "res": res,
        }

        return deposit_stage

    def solve_deposit_decision_with_jac(self, consumption_stage):
        dvdl_func_next = consumption_stage.dvdl_func
        dvdb_func_next = consumption_stage.dvdb_func
        v_func_next = consumption_stage.v_func

        def objective(d_nrm, m_nrm, n_nrm):
            l_nrm = m_nrm - d_nrm
            b_nrm = n_nrm + d_nrm + self.g(d_nrm)

            return -v_func_next(l_nrm, b_nrm)

        def jacobian(d_nrm, m_nrm, n_nrm):
            l_nrm = m_nrm - d_nrm
            b_nrm = n_nrm + d_nrm + self.g(d_nrm)

            dvdl = dvdl_func_next(l_nrm, b_nrm)
            dvdb = dvdb_func_next(l_nrm, b_nrm)

            return -dvdl + dvdb * (1 + self.g.der(d_nrm))

        dMat = np.empty_like(self.mMat)

        for mi in range(self.mGrid.size):
            for ni in range(self.nGrid.size):
                m_nrm = self.mGrid[mi]
                n_nrm = self.nGrid[ni]
                res = minimize(
                    objective,
                    jac=jacobian,
                    x0=m_nrm / 2,
                    args=(
                        m_nrm,
                        n_nrm,
                    ),
                    bounds=[(0, m_nrm)],
                )
                dMat[mi, ni] = res.x

        # add d = 0 when no liquid cash
        dMat_temp = np.insert(dMat, 0, 0.0, axis=0)

        d_func = LinearFast(dMat_temp, [np.append(0.0, self.mGrid), self.nGrid])

        deposit_stage = DepositStage(
            d_func=d_func,
        )

        return deposit_stage

    def solve_vfi(self):
        deposit_solution_next = self.solution_next.deposit_stage
        post_decision_solution = self.solve_post_decision(deposit_solution_next)

        def objective(control, m_nrm, n_nrm):
            c_nrm = control[0]
            d_nrm = control[1]
            a_nrm = m_nrm - c_nrm - d_nrm
            b_nrm = n_nrm + d_nrm + self.g(d_nrm)
            value = self.u(c_nrm) + post_decision_solution.v_func(a_nrm, b_nrm)
            return -value

        cMat = np.empty_like(self.mMat)
        dMat = np.empty_like(self.mMat)
        for mi in range(self.mGrid.size):
            for ni in range(self.nGrid.size):
                m_nrm = self.mGrid[mi]
                n_nrm = self.nGrid[mi]

                bounds = Bounds([0, m_nrm], [0, m_nrm])
                linear_constraint = LinearConstraint([1, 1], [0], [m_nrm])

                sol = minimize(
                    objective,
                    x0=np.array([m_nrm / 2, m_nrm / 2]),
                    args=(
                        m_nrm,
                        n_nrm,
                    ),
                    method="trust-constr",
                    constraints=[linear_constraint],
                    bounds=bounds,
                )

                cMat[mi, ni] = sol.x[0]
                dMat[mi, ni] = sol.x[1]

    def solve(self):
        if hasattr(self.solution_next, "deposit_stage"):
            deposit_solution_next = self.solution_next.deposit_stage
        else:
            deposit_solution_next = self.solution_next

        post_decision_solution = self.solve_post_decision(deposit_solution_next)
        consumption_solution = self.solve_consumption_decision(post_decision_solution)
        deposit_solution = self.solve_deposit_decision(consumption_solution)

        solution = PensionSolution(
            post_decision_stage=post_decision_solution,
            consumption_stage=consumption_solution,
            deposit_stage=deposit_solution,
        )

        # solution.extras = self.solve_deposit_decision_vfi(consumption_solution)

        return solution


# =====================================================
# Constructor Functions
# =====================================================


def make_pension_grids(
    epsilon,
    mCount,
    mMax,
    mNestFac,
    nCount,
    nMax,
    nNestFac,
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
):
    """
    Constructs grids for pension model.

    Returns
    -------
    dict
        Dictionary containing all grids
    """
    # worker grids
    mGrid = make_assets_grid(epsilon, mMax, mCount, [], mNestFac)
    nGrid = make_assets_grid(0.0, nMax, nCount, [], nNestFac)
    mMat, nMat = np.meshgrid(mGrid, nGrid, indexing="ij")

    # pure consumption grids
    aGrid = make_assets_grid(0.0, aMax, aCount, [], aNestFac)
    bGrid = make_assets_grid(0.0, bMax, bCount, [], bNestFac)
    aMat, bMat = np.meshgrid(aGrid, bGrid, indexing="ij")

    # pension deposit grids
    lGrid = make_assets_grid(epsilon, lMax, lCount, [], lNestFac)
    blGrid = make_assets_grid(0.0, blMax, blCount, [], blNestFac)
    lMat, blMat = np.meshgrid(lGrid, blGrid, indexing="ij")

    return {
        "mGrid": mGrid,
        "nGrid": nGrid,
        "mMat": mMat,
        "nMat": nMat,
        "aGrid": aGrid,
        "bGrid": bGrid,
        "aMat": aMat,
        "bMat": bMat,
        "lGrid": lGrid,
        "blGrid": blGrid,
        "lMat": lMat,
        "blMat": blMat,
    }


def make_pension_solution_terminal(CRRA):
    """
    Constructs the terminal period solution for pension model.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion

    Returns
    -------
    PensionSolution
        Terminal period solution object
    """

    # consume everything in terminal period
    def c_func(mNrm, nNrm):
        return mNrm + nNrm

    # deposit nothing in terminal period
    def d_func(mNrm, nNrm):
        return 0.0

    u = UtilityFuncCRRA(CRRA)

    def v_func(mNrm, nNrm):
        return u(c_func(mNrm, nNrm))

    def vp_func(mNrm, nNrm):
        return u.der(c_func(mNrm, nNrm))

    consumption_stage = ConsumptionStage(
        c_func=c_func,
        v_func=v_func,
        dvdl_func=vp_func,
        dvdb_func=vp_func,
    )

    deposit_stage = DepositStage(
        d_func=d_func,
        v_func=v_func,
        dvdm_func=vp_func,
        dvdn_func=vp_func,
    )

    return PensionSolution(
        deposit_stage=deposit_stage,
        consumption_stage=consumption_stage,
    )


# =====================================================
# Parameter Dictionary
# =====================================================

init_pension_contrib = init_portfolio.copy()
# Deep copy constructors to avoid modifying init_portfolio
if "constructors" in init_pension_contrib:
    from copy import deepcopy as _deepcopy

    init_pension_contrib["constructors"] = _deepcopy(
        init_pension_contrib["constructors"]
    )
init_pension_contrib["Rfree"] = [1.02]  # List for finite-horizon compatibility
init_pension_contrib["RiskyAvg"] = [1.04]  # List for finite-horizon compatibility
init_pension_contrib["RiskyStd"] = [0.0]  # List for finite-horizon compatibility
init_pension_contrib["RiskyCount"] = 1
# Disable ShareLimit calculation when no risky asset (RiskyStd=0)
# This prevents portfolio optimization errors
init_pension_contrib["AdjustPrb"] = 0.0  # Never adjust portfolio (no risky asset)
init_pension_contrib["DiscFac"] = 0.98
init_pension_contrib["CRRA"] = 2.0
init_pension_contrib["DisutilLabor"] = 0.25
init_pension_contrib["TaxDeduct"] = 0.10
init_pension_contrib["LivPrb"] = [1.0]
init_pension_contrib["PermGroFac"] = [1.0]
init_pension_contrib["TranShkStd"] = [0.10]
init_pension_contrib["TranShkCount"] = 7
init_pension_contrib["PermShkStd"] = [0.0]
init_pension_contrib["PermShkCount"] = 1
init_pension_contrib["UnempPrb"] = 0.0  # Prob of unemployment while working
init_pension_contrib["IncUnemp"] = 0.0
# Prob of unemployment while retired
init_pension_contrib["UnempPrbRet"] = 0.0
init_pension_contrib["IncUnempRet"] = 0.50
init_pension_contrib["TasteShkStd"] = 0.10

init_pension_contrib["epsilon"] = 1e-6

init_pension_contrib["mCount"] = 50
init_pension_contrib["mMax"] = 10
init_pension_contrib["mNestFac"] = 2

init_pension_contrib["nCount"] = 50
init_pension_contrib["nMax"] = 12
init_pension_contrib["nNestFac"] = 2

init_pension_contrib["lCount"] = 50
init_pension_contrib["lMax"] = 9
init_pension_contrib["lNestFac"] = 2

init_pension_contrib["blCount"] = 50
init_pension_contrib["blMax"] = 13
init_pension_contrib["blNestFac"] = 2

init_pension_contrib["aCount"] = 50
init_pension_contrib["aMax"] = 8
init_pension_contrib["aNestFac"] = 2

init_pension_contrib["bCount"] = 50
init_pension_contrib["bMax"] = 14
init_pension_contrib["bNestFac"] = 2

# Build grids directly
grids = make_pension_grids(
    epsilon=init_pension_contrib["epsilon"],
    mCount=init_pension_contrib["mCount"],
    mMax=init_pension_contrib["mMax"],
    mNestFac=init_pension_contrib["mNestFac"],
    nCount=init_pension_contrib["nCount"],
    nMax=init_pension_contrib["nMax"],
    nNestFac=init_pension_contrib["nNestFac"],
    aCount=init_pension_contrib["aCount"],
    aMax=init_pension_contrib["aMax"],
    aNestFac=init_pension_contrib["aNestFac"],
    bCount=init_pension_contrib["bCount"],
    bMax=init_pension_contrib["bMax"],
    bNestFac=init_pension_contrib["bNestFac"],
    lCount=init_pension_contrib["lCount"],
    lMax=init_pension_contrib["lMax"],
    lNestFac=init_pension_contrib["lNestFac"],
    blCount=init_pension_contrib["blCount"],
    blMax=init_pension_contrib["blMax"],
    blNestFac=init_pension_contrib["blNestFac"],
)
init_pension_contrib.update(grids)

# Add terminal solution constructor and remove ShareLimit (no risky asset)
# Remove ShareLimit first - we have no risky asset (RiskyStd=0)
# This prevents portfolio optimization errors during instantiation
if (
    "constructors" in init_pension_contrib
    and "ShareLimit" in init_pension_contrib["constructors"]
):
    del init_pension_contrib["constructors"]["ShareLimit"]
    # Set ShareLimit to list (all in risky asset, but RiskyStd=0 so no actual risk)
    init_pension_contrib["ShareLimit"] = [1.0]  # List for finite-horizon compatibility

if "constructors" not in init_pension_contrib:
    init_pension_contrib["constructors"] = {}

# Parent's ShockDstn constructor already creates labeled distributions!
# No need to wrap it - combine_IncShkDstn_and_RiskyDstn returns IndexDistribution
# with DiscreteDistributionLabeled as engine

init_pension_contrib["constructors"]["solution_terminal"] = (
    make_pension_solution_terminal
)


# =====================================================
# Agent Type Definition
# =====================================================


class PensionConsumerType(RiskyAssetConsumerType):
    """
    Consumer with pension contribution decision: chooses how much to deposit
    into a tax-advantaged pension account each period.
    """

    time_inv_ = deepcopy(RiskyAssetConsumerType.time_inv_)
    time_inv_ = time_inv_ + [
        "DisutilLabor",
        "IncUnempRet",
        "TasteShkStd",
        "TaxDeduct",
        # Grid-related parameters
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
        "params": init_pension_contrib,
        "solver": make_one_period_oo_solver(PensionSolver),
    }


class RetirementConsumerType(PensionConsumerType):
    def __init__(self, **kwds):
        params = init_pension_retirement.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        PensionConsumerType.__init__(self, **params)

        # Add consumer-type specific objects, copying to create independent versions
        contrib_solver = make_one_period_oo_solver(PensionSolver)
        retirement_solver = make_one_period_oo_solver(ConsIndShockSolver)

        solvers = []

        for t in range(self.T_cycle):
            if t < self.T_retire:
                solvers.append(contrib_solver)
            else:
                solvers.append(retirement_solver)

        self.solve_one_period = solvers
        self.add_to_time_vary("solve_one_period")

        self.update()  # Make assets grid, income process, terminal solution

    def update_solution_terminal(self):
        return IndShockConsumerType.update_solution_terminal(self)


init_pension_retirement = init_pension_contrib.copy()
T_cycle = 15
T_retire = 10
retire_length = T_cycle - T_retire
init_pension_retirement["T_retire"] = 10
init_pension_retirement["T_cycle"] = 15
init_pension_retirement["UnempPrb"] = init_portfolio["UnempPrb"]
init_pension_retirement["IncUnemp"] = init_portfolio["IncUnemp"]
init_pension_retirement["UnempPrbRet"] = init_portfolio["UnempPrbRet"]
init_pension_retirement["IncUnempRet"] = init_portfolio["IncUnempRet"]
init_pension_retirement["LivPrb"] = init_portfolio["LivPrb"] * T_cycle
init_pension_retirement["PermGroFac"] = init_portfolio["PermGroFac"] * T_cycle
init_pension_retirement["PermShkStd"] = [0.0] * T_cycle
init_pension_retirement["TranShkStd"] = [0.01] * T_retire + [0.0] * retire_length
init_pension_retirement["PermShkCount"] = 1
init_pension_retirement["TranShkCount"] = 7
