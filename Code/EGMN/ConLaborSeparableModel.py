from copy import copy
from dataclasses import dataclass

import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import utility, utilityP, utilityP_inv
from HARK.ConsumptionSaving.ConsLaborModel import (
    ConsumerLaborSolution,
    LaborIntMargConsumerType,
    init_labor_intensive,
)
from HARK.core import make_one_period_oo_solver
from HARK.distribution import DiscreteDistribution, calc_expectation
from HARK.interpolation import LinearInterp, LinearInterpOnInterp1D, MargValueFuncCRRA
from HARK.metric import MetricObject


class LaborSeparableConsumerType(LaborIntMargConsumerType):
    time_inv_ = copy(LaborIntMargConsumerType.time_inv_)
    time_inv_ += [
        "Disutility",
        "LaborConstant",
        "LaborShare",
        "LeisureConstant",
        "LeisureShare",
    ]

    def __init__(self, **kwds):
        params = init_labor_separable.copy()
        params.update(kwds)

        super().__init__(**params)

        self.solve_one_period = make_one_period_oo_solver(SeparableLaborSolver)
        self.update()

    def update(self):
        super().update()
        self.update_solution_terminal()

    def update_LbrCost(self):
        pass

    def update_solution_terminal(self):
        tshkgrid = self.TranShkGrid[-1]

        # check if agent will experience unemployment
        # at unemployment the solution is to not work
        zero_bound = True if tshkgrid[0] == 0.0 else False

        # use assets grid as cash on hand
        mgrid = np.append(0.0, self.aXtraGrid)
        cgrid = mgrid  # consume all cash in terminal period

        cFunc_post_labr = LinearInterp(mgrid, cgrid)
        uPfunc_post_labr = MargValueFuncCRRA(cFunc_post_labr, self.CRRA)

        # construct matrices of exogenous cash on hand and transitory shocks
        mnrmat, tshkmat = np.meshgrid(mgrid, tshkgrid, indexing="ij")
        cnrmat = mnrmat

        # note: it would be interesting to not apply an upper bound here
        # and instead build a lower envelope to see what agents would do
        # if they were not constrained by the labor constraint
        # make sure leisure is not greater than 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            # this is an egm step to find optimal leisure in the terminal period
            x = uPfunc_post_labr(mnrmat) * tshkmat

            if self.Disutility:
                lbrmat = utilityP_inv(x / self.LaborConstant, -self.LaborShare)
                lbrmat = np.clip(lbrmat, 0.0, 1.0)
                lsrmat = 1 - lbrmat
            else:
                lsrmat = utilityP_inv(x / self.LeisureConstant, self.LeisureShare)
                lsrmat = np.clip(lsrmat, 0.0, 1.0)
                lbrmat = 1 - lsrmat

        if zero_bound:
            # if agent is unemployed, their leisure is 1.0 and labor is 0.0
            lsrmat[:, 0] = 1.0
            lbrmat[:, 0] = 0.0

        # bank balances = cash on hand - wage * labor
        bnrmat = mnrmat - tshkmat * lbrmat

        # construct leisure function as LinearInterpOnInterp1D
        # because bnrmat is endogenous and irregular (not tiled)
        lsrFunc_by_tShk = []
        for tshk in range(tshkgrid.size):
            lsrFunc_by_tShk.append(LinearInterp(bnrmat[:, tshk], lsrmat[:, tshk]))

        self.lsrFunc_terminal = LinearInterpOnInterp1D(lsrFunc_by_tShk, tshkgrid)
        self.lsrFunc_terminal.values = (lsrmat, bnrmat, tshkmat)

        # construct labor function as LinearInterpOnInterp1D
        lbrFunc_by_tShk = []
        for tshk in range(tshkgrid.size):
            lbrFunc_by_tShk.append(LinearInterp(bnrmat[:, tshk], lbrmat[:, tshk]))

        self.lbrFunc_terminal = LinearInterpOnInterp1D(lbrFunc_by_tShk, tshkgrid)
        self.lbrFunc_terminal.values = (lbrmat, bnrmat, tshkmat)

        # construct consumption function using same values
        # but with an endogenous bank balance grid
        cFunc_by_tShk = []
        for tshk in range(tshkgrid.size):
            cFunc_by_tShk.append(LinearInterp(bnrmat[:, tshk], cnrmat[:, tshk]))

        self.cFunc_terminal = LinearInterpOnInterp1D(cFunc_by_tShk, tshkgrid)
        self.cFunc_terminal.values = (cnrmat, bnrmat, tshkmat)

        self.vPfunc_terminal = MargValueFuncCRRA(self.cFunc_terminal, self.CRRA)

        # create terminal solution object
        self.solution_terminal = ConsumerLaborSolution(
            cFunc=self.cFunc_terminal,
            LbrFunc=self.lbrFunc_terminal,
            vPfunc=self.vPfunc_terminal,
        )

        self.solution_terminal.LsrFunc = self.lsrFunc_terminal


@dataclass
class SeparableLaborSolver:
    solution_next: ConsumerLaborSolution
    TranShkDstn: DiscreteDistribution
    LivPrb: float
    DiscFac: float
    CRRA: float
    Disutility: bool
    LaborConstant: float
    LaborShare: float
    LeisureConstant: float
    LeisureShare: float
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
        """
        Define temporary functions for utility and its derivative and inverse
        """

        self.u = lambda x: utility(x, self.CRRA)
        self.uP = lambda x: utilityP(x, self.CRRA)
        self.uPinv = lambda x: utilityP_inv(x, self.CRRA)

        if self.Disutility:
            self.n = lambda x: -self.LaborConstant * utility(1 - x, -self.LaborShare)
            self.nP = lambda x: self.LaborConstant * utilityP(1 - x, -self.LaborShare)
            self.nPinv = lambda x: 1 - utilityP_inv(
                x / self.LaborConstant, -self.LaborShare
            )
        else:
            self.n = lambda x: self.LeisureConstant * utility(x, self.LeisureShare)
            self.nP = lambda x: self.LeisureConstant * utilityP(x, self.LeisureShare)
            self.nPinv = lambda x: utilityP_inv(
                x / self.LaborConstant, self.LeisureShare
            )

    def prepare_to_solve(self):
        # Unpack next period's solution
        self.vPfunc_next = self.solution_next.vPfunc

    def prepare_to_calc_EndOfPrdvP(self):
        # we can add zero back because it's not the natural borrowing constraint
        self.aGrid = np.append(0.0, self.aXtraGrid)
        self.bGrid = self.aGrid * self.Rfree

    def calc_EndOfPrdvP(self):
        def dvdb_func(shock, bnrm):
            # This calculation is inefficient because it has to interpolate over shocks
            # because of the way this does expectations, there's no off-the-grid shocks
            # have to make sure shock and bnrm are same dimension
            return self.vPfunc_next(bnrm, shock.repeat(bnrm.size))

        EndOfPrdvP_vals = calc_expectation(self.TranShkDstn, dvdb_func, self.bGrid)
        EndOfPrdvP_nvrs = self.uPinv(EndOfPrdvP_vals)
        EndOfPrdvP_nvrs_func = LinearInterp(self.bGrid, EndOfPrdvP_nvrs)
        EndOfPrdvP_func = MargValueFuncCRRA(EndOfPrdvP_nvrs_func, self.CRRA)

        self.EndOfPrdvPNvrs = EndOfPrdvP_nvrs
        self.EndOfPrdvPfunc = EndOfPrdvP_func

    def make_consumption_solution(self):
        # this is the consumption function that is consistent with the exogneous asset level
        # this decision comes before end of period post decision value function, but after
        # the labor-leisure decision function
        self.cGrid = self.EndOfPrdvPNvrs
        self.mGrid = self.cGrid + self.aGrid

        self.cFuncEndOfPrd = LinearInterp(self.mGrid, self.cGrid)
        self.vPfuncEndOfPrd = MargValueFuncCRRA(self.cFuncEndOfPrd, self.CRRA)

    def make_labor_leisure_solution(self):
        tshkgrid = self.TranShkGrid
        zero_bound = True if tshkgrid[0] == 0.0 else False

        # this mgrid and cgrid are consistent with exogenous asset level
        mgrid = np.append(0.0, self.aXtraGrid)

        # in the previous step we found the endogenous m that is consistent with
        # exogenous asset level, now we can use m (and tshk) as the exogenous grids
        # that will give us the consistent labor, leisure, and bank balances for a
        mnrmat, tshkmat = np.meshgrid(mgrid, tshkgrid, indexing="ij")

        # this is the egm step, given exog m and tshk, find leisure
        lsrmat = self.nPinv(self.vPfuncEndOfPrd(mnrmat) * tshkmat)
        # Make sure lsrgrid is not greater than 1.0
        lsrmat = np.clip(lsrmat, 0.0, 1.0)
        if zero_bound:
            lsrmat[:, 0] = 1.0

        lbrmat = 1.0 - lsrmat  # labor is 1 - leisure
        # bank balances = cash on hand - wage * labor
        bnrmat = mnrmat - tshkmat * lbrmat

        lsrFunc_by_tShk = []
        for tShk in range(tshkgrid.size):
            lsrFunc_by_tShk.append(LinearInterp(bnrmat[:, tShk], lsrmat[:, tShk]))

        self.lsrFunc = LinearInterpOnInterp1D(lsrFunc_by_tShk, tshkgrid)

        lbrFunc_by_tShk = []
        for tShk in range(tshkgrid.size):
            lbrFunc_by_tShk.append(LinearInterp(bnrmat[:, tShk], lbrmat[:, tShk]))

        self.lbrFunc = LinearInterpOnInterp1D(lbrFunc_by_tShk, tshkgrid)

        cgrid = self.cFuncEndOfPrd(mgrid)

        # as in the terminal solution, we construct the consumption function by using
        # the c that was consistent with a, and the b that was consistent with m
        cFunc_by_tShk = []
        for tShk in range(tshkgrid.size):
            cFunc_by_tShk.append(LinearInterp(bnrmat[:, tShk], cgrid))

        self.cFunc = LinearInterpOnInterp1D(cFunc_by_tShk, tshkgrid)
        self.vPfunc_now = MargValueFuncCRRA(self.cFunc, self.CRRA)

    def make_consumer_labor_solution(self):
        self.solution = ConsumerLaborSolution(
            cFunc=self.cFunc,
            LbrFunc=self.lbrFunc,
            vPfunc=self.vPfunc_now,
        )

        self.solution.LsrFunc = self.lsrFunc
        self.solution.cFuncEndOfPrd = self.cFuncEndOfPrd

    def solve(self):
        self.prepare_to_calc_EndOfPrdvP()
        self.calc_EndOfPrdvP()

        self.make_consumption_solution()
        self.make_labor_leisure_solution()

        self.make_consumer_labor_solution()

        return self.solution


init_labor_separable = init_labor_intensive.copy()
init_labor_separable["Disutility"] = True
init_labor_separable["LaborConstant"] = 0.6
init_labor_separable["LaborShare"] = 1.0
init_labor_separable["LeisureConstant"] = 0.0
init_labor_separable["LeisureShare"] = 0.0
# init_labor_separable["UnempPrb"] = 0.0
