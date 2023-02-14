'''
This file has an extension of MarkovConsumerType that is used for the Fiscal project.
'''
import warnings
import numpy as np
from HARK.distribution import DiscreteDistribution, Bernoulli, Uniform
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import MargValueFunc, ConsumerSolution
from HARK.interpolation import LinearInterp, LowerEnvelope
from HARK.core import distanceMetric
from Parameters import makeMacroMrkvArray, makeCondMrkvArrays, makeFullMrkvArray, T_sim
import matplotlib.pyplot as plt
from copy import copy

# Define a modified MarkovConsumerType
class FiscalType(MarkovConsumerType):
    time_inv_ = MarkovConsumerType.time_inv_ 
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        MarkovConsumerType.__init__(self,cycles=1,time_flow=True,**kwds)
        self.shock_vars += ['update_draw']
        self.solveOnePeriod = solveConsMarkovALT
        
    def preSolve(self):
        MarkovConsumerType.preSolve(self)
        self.updateSolutionTerminal()
        
    def initializeSim(self):
        MarkovConsumerType.initializeSim(self)
        if hasattr(self,'use_prestate'):
            self.restoreState()
        else:   # set to ergodic unemployment rate during normal times
            init_unemp_dist = DiscreteDistribution(1.0-self.Urate_normal, np.array([0,1]), seed=self.RNG.randint(0,2**31-1))
            self.MrkvNow[:] = init_unemp_dist.drawDiscrete(self.AgentCount)
            if not hasattr(self,'mortality_off'):
                self.calcAgeDistribution()
                self.initializeAges()
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow[:] = self.Mrkv_univ
        self.MacroMrkvNow = (np.floor(self.MrkvNow/3)).astype(int)
        self.MicroMrkvNow = self.MrkvNow%3
        self.EconomyMrkvNow = self.MacroMrkvNow #For aggregate model only
        self.EconomyMrkvNow_hist = [0] * self.T_sim #For aggregate model only

        
        
    def getMortality(self):
        '''
        A modified version of getMortality that reads mortality history if the
        attribute read_mortality exists.  This is a workaround to make sure the
        history of death events is identical across simulations.
        '''
        if (self.read_shocks or hasattr(self,'read_mortality')):
            who_dies = self.who_dies_fixed_hist[self.t_sim,:]
        else:
            who_dies = self.simDeath()
        self.simBirth(who_dies)
        self.who_dies = who_dies
        return None
    
    
    def simDeath(self):
        if hasattr(self,'mortality_off'):
            return np.zeros(self.AgentCount, dtype=bool)
        else:
            return MarkovConsumerType.simDeath(self)

    def getShocks(self):
        MarkovConsumerType.getShocks(self)
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow = self.MrkvNow_temp # Make sure real sequence is recorded
        self.update_draw = self.RNG.permutation(np.array(range(self.AgentCount))) # A list permuted integers, low draws will update their aggregate Markov state
            
    def getStates(self):
        MarkovConsumerType.getStates(self)
        
        # Initialize the random draw of Pi*N agents who update
        how_many_update = int(round(self.UpdatePrb*self.AgentCount))
        self.update = self.update_draw < how_many_update
        # Only updaters change their perception of the Markov state
        if hasattr(self,'MrkvNowPcvd'):
            self.MrkvNowPcvd[self.update] = self.MrkvNow[self.update]
        else: # This only triggers in the first simulated period
            self.MrkvNowPcvd = np.ones(self.AgentCount,dtype=int)*self.MrkvNow
        #$$$$$$$$$$ 
        # update the idiosyncratic state (employed, unemployed with benefits, unemployed without benefits)
        # but leave the macro state as it is (idiosyncratic state is 'modulo 3')
        self.MrkvNowPcvd = np.remainder(self.MrkvNow,3) + 3*np.floor_divide(self.MrkvNowPcvd,3)
  
    def getMacroMarkovStates(self):
        J = self.MacroMrkvArray.shape[0]
        # divide markov matrix into (independent) recession probabilities and other
        recession_transitions = self.MacroMrkvArray[0:2,0:2]
        other_transitions = self.MacroMrkvArray[np.array(range(int(J/2)))*2,:][:,np.array(range(int(J/2)))*2]
        recession_draws = Uniform(seed=self.RNG.randint(0,2**31-1)).draw(self.AgentCount)
        other_draws = Uniform(seed=self.RNG.randint(0,2**31-1)).draw(self.AgentCount)
        
        # Determine which agents are in which states right now
        MacroMrkvPrev = self.MacroMrkvNow
        MacroMrkvNow = np.zeros(self.AgentCount,dtype=int)
        MacroMrkvBoolArray = (np.zeros((J,self.AgentCount))).astype(bool)
        for j in range(J):
            MacroMrkvBoolArray[j,:] = MacroMrkvPrev == j
        
        # Draw new Markov states for each agent
        RecessionCutoffs = np.cumsum(recession_transitions,axis=1)
        OtherCutoffs = np.cumsum(other_transitions,axis=1)
        for j in range(J):
            Recession = np.searchsorted(RecessionCutoffs[j%2,:],recession_draws[MacroMrkvBoolArray[j,:]]).astype(int)
            Other = np.searchsorted(OtherCutoffs[np.floor(j/2).astype(int),:],other_draws[MacroMrkvBoolArray[j,:]]).astype(int)
            MacroMrkvNow[MacroMrkvBoolArray[j,:]] = 2*Other+Recession
        self.MacroMrkvNow = MacroMrkvNow.astype(int)
        
    def getMicroMarkvStates_guts(self, unemployment_draw):
        dont_change = self.t_age == 0 # Don't change Markov state for those who were just born 
        # if self.t_sim == 0: # Respect initial distribution of Markov states
        #     dont_change[:] = True
        
        # Determine which agents are in which states right now
        J = self.CondMrkvArrays[0].shape[0]
        MicroMrkvPrev = copy(self.MicroMrkvNow)
        MicroMrkvNow = np.zeros(self.AgentCount,dtype=int)
        MicroMrkvBoolArray = np.zeros((J,self.AgentCount))
        for j in range(J):
            MicroMrkvBoolArray[j,:] = MicroMrkvPrev == j
        
        # Draw new Markov states for each agent
        for i in range(self.MacroMrkvArray.shape[0]):
            Cutoffs = np.cumsum(self.CondMrkvArrays[i],axis=1)
            macro_match = self.MacroMrkvNow == i
            for j in range(J):
                these = np.logical_and(macro_match, MicroMrkvBoolArray[j,:])
                MicroMrkvNow[these] = np.searchsorted(Cutoffs[j,:],unemployment_draw[these]).astype(int)
        MicroMrkvNow[dont_change] = MicroMrkvNow[dont_change]
        self.MicroMrkvNow = MicroMrkvNow.astype(int)
        
    def getMicroMarkovStates(self):
        self.unemployment_draw = Uniform(seed=self.RNG.randint(0,2**31-1)).draw(self.AgentCount)
        self.getMicroMarkvStates_guts(self.unemployment_draw)
           
    def getMarkovStates(self):
        self.getMacroMarkovStates()
        self.getMicroMarkovStates()
        MrkvNow = 3*self.MacroMrkvNow + self.MicroMrkvNow
        self.MrkvNow = MrkvNow.astype(int)
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow_temp = self.MrkvNow
            self.MrkvNow = self.Mrkv_univ*np.ones(self.AgentCount, dtype=int)
            # ^^ Store the real states but force income shocks to be based on one particular state
            
    #$$$$$$$$$$    
    def updateMrkvArray(self):
        MacroMrkvArray = makeMacroMrkvArray(self.Rspell, self.PolicyUBspell, self.TaxCutPeriods, self.TaxCutContinuationProb_Rec, self.TaxCutContinuationProb_Bas)
        CondMrkvArrays = makeCondMrkvArrays(self.Urate_normal, self.Uspell_normal, self.UBspell_normal, self.Urate_recession, self.Uspell_recession, self.UBspell_extended, self.TaxCutPeriods)
        MrkvArray = makeFullMrkvArray(MacroMrkvArray, CondMrkvArrays)
        self.MrkvArray  = MrkvArray
    
    def calcAgeDistribution(self):
        '''
        Calculates the long run distribution of t_cycle in the population.
        '''
        if self.T_cycle==1:
            T_cycle_actual = 400
            LivPrb_array = [[self.LivPrb[0][0]]]*T_cycle_actual
        else:
            T_cycle_actual = self.T_cycle
            LivPrb_array = self.LivPrb
        AgeMarkov = np.zeros((T_cycle_actual+1,T_cycle_actual+1))
        for t in range(T_cycle_actual):
            p = LivPrb_array[t][0]
            AgeMarkov[t,t+1] = p
            AgeMarkov[t,0] = 1. - p
        AgeMarkov[-1,0] = 1.
        
        AgeMarkovT = np.transpose(AgeMarkov)
        vals, vecs = np.linalg.eig(AgeMarkovT)
        dist = np.abs(np.abs(vals) - 1.)
        idx = np.argmin(dist)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore warning about casting complex eigenvector to float
            LRagePrbs = vecs[:,idx].astype(float)
        LRagePrbs /= np.sum(LRagePrbs)
        age_vec = np.arange(T_cycle_actual+1).astype(int)
        self.LRageDstn = DiscreteDistribution(LRagePrbs, age_vec,
                                seed=self.RNG.randint(0,2**31-1))
        
        
    def initializeAges(self):
        '''
        Assign initial values of t_cycle to simulated agents, using the attribute
        LRageDstn as the distribution of discrete ages.
        '''
        age = self.LRageDstn.drawDiscrete(self.AgentCount)
        age = age.astype(int)
        if self.T_cycle!=1:
            self.t_cycle = age
        self.t_age = age
    
    def switchToCounterfactualMode(self):
        '''
        Very small method that swaps in the "big" Markov-state versions of some
        solution attributes, replacing the "small" two-state versions that are used
        only to generate the pre-recession initial distbution of state variables.
        It then prepares this type to create alternate shock histories so it can
        run counterfactual experiments.
        '''
        del self.solution
        self.delFromTimeVary('solution')
        
        # Swap in "big" versions of the Markov-state-varying attributes
        self.LivPrb = self.LivPrb_big
        self.PermGroFac = self.PermGroFac_big
        self.MrkvArray = self.MrkvArray_big
        self.Rfree = self.Rfree_big
        self.IncomeDstn = self.IncomeDstn_big
        
        # Adjust simulation parameters for the counterfactual experiments
        self.T_sim = T_sim
        self.track_vars = ['cNrmNow','pLvlNow','aNrmNow','mNrmNow','MrkvNowPcvd','MacroMrkvNow','MicroMrkvNow','cLvlNow','cLvl_splurgeNow']
        self.use_prestate = None
        #print('Finished type ' + str(self.seed) + '!')
        
    def makeMrkvShockHistory(self):
        self.initializeSim()
        self.history['MrkvNow'] = np.zeros((self.T_sim, self.AgentCount)) + np.nan

        # Make and store the history of shocks for each period
        for t in range(self.T_sim):
            self.getMarkovStates()
            if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
                self.MrkvNow = self.MrkvNow_temp # Make sure real sequence is recorded
            self.history['MrkvNow'][self.t_sim,:] = getattr(self, 'MrkvNow')
            self.t_sim += 1
            self.t_age = self.t_age + 1  # Age all consumers by one period
            self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
            self.t_cycle[self.t_cycle == self.T_cycle] = 0  # Resetting to zero for those who have reached the end

    def makeAlternateShockHistories(self):
        '''
        Make a history of Markov states and income shocks starting from each Markov state.
        '''
        
        print('makeAlternateShockHistories called')
        
        #J = self.MrkvArray[0].shape[0]
        J = 18 # hitwithrecessionshock only shocks agents into the first 18 markov states
        MrkvHistAll = np.zeros((J,self.T_sim,self.AgentCount), dtype=int)
        self.Mrkv_univ = 0
        self.read_shocks = False
        self.makeShockHistory()
        self.read_mortality = True # Make sure that every death history is the same
        self.who_dies_fixed_hist = self.history['who_dies'].copy()
        self.update_draw_fixed_hist = self.history['update_draw'].copy()
        self.perm_shock_fixed_hist = self.history['PermShkNow'].copy()
        self.tran_shock_fixed_hist = self.history['TranShkNow'].copy()
        
        for j in range(J):
            self.Mrkv_univ = j 
            self.read_shocks = False
            self.makeMrkvShockHistory()
            MrkvHistAll[j,:,:] = self.history['MrkvNow']
        
        # Store as attributes of self
        self.MrkvHistAll = MrkvHistAll
        self.Mrkv_univ = None
        # self.MrkvArray_prev = self.MrkvArray
        self.R_shared_prev = self.R_shared
        del(self.read_mortality)
        
        
    def solveIfChanged(self):
        '''
        Re-solve the lifecycle model only if the attributes MrkvArray
        do not match those in MrkvArray_prev .
        '''
        # Check whether MrkvArray has changed (and whether they exist at all!)
        try: 
            same_MrkvArray = distanceMetric(self.MrkvArray, self.MrkvArray_prev) == 0.
            if (same_MrkvArray):
                return
        except:
            pass
        
        # Re-solve the model, then note the values in MrkvArray
        self.solve()
        self.MrkvArray_prev = self.MrkvArray
        
        
    def makeShocksIfChanged(self):
        '''
        Re-draw the histories of Markov states and income shocks only if the attributes
        MrkvArray and R_shared do not match those in MrkvArray_prev and R_shared_prev.
        '''
        
        # Check whether MrkvArray and R_shared have changed (and whether they exist at all!)
        try: 
            same_MrkvArray = distanceMetric(self.MrkvArray, self.MrkvArray_prev) == 0.
            same_shared = self.R_shared == self.R_shared_prev
            if (same_MrkvArray and same_shared):
                return
        except:
            pass
        
        # Re-draw the shock histories, then note the values in MrkvArray and R_shared
        #print('MrkArray Updated')
        self.makeAlternateShockHistories()
        
    
    def saveState(self):
        '''
        Record the current state of simulation variables for later use.
        '''
        self.aNrm_base = self.aNrmNow.copy()
        self.pLvl_base = self.pLvlNow.copy()
        self.Mrkv_base = self.MrkvNow.copy()
        self.cycle_base  = self.t_cycle.copy()
        self.age_base  = self.t_age.copy()
        self.t_sim_base = self.t_sim
        self.PlvlAgg_base = self.PlvlAggNow


    def restoreState(self):
        '''
        Restore the state of the simulation to some baseline values.
        '''
        self.aNrmNow = self.aNrm_base.copy()
        self.pLvlNow = self.pLvl_base.copy()
        self.MrkvNow = self.Mrkv_base.copy()
        self.t_cycle = self.cycle_base.copy()
        self.t_age   = self.age_base.copy()
        self.PlvlAggNow = self.PlvlAgg_base
        
    def hitWithRecessionShock(self):
        '''
        Alter the Markov state of each simulated agent, jumping some people into
        recession states
        '''
        # Shock unemployment up to ergodic unemployment level in normal or recession state
        if self.RecessionShock:
            this_Urate = self.Urate_recession
        else:
            this_Urate = self.Urate_normal
        
        # Draw new Markov states for each agents who are employed
        draws = Uniform(seed=self.RNG.randint(0,2**31-1)).draw(self.AgentCount)
        draws = self.RNG.permutation(draws)
        MrkvNew = self.MrkvNow
        old_Urate = self.Urate_normal
        draws_empy2umemp = draws > 1.0-(this_Urate-old_Urate)/(1.0-old_Urate)
        MrkvNew[np.logical_and(np.equal(self.MrkvNow,0), draws_empy2umemp) ] = 2 # Move people from employment to unemployment such that total unemployment rate is as required. Don't touch already unemployed people.
        
        #$$$$$$$$$$
        if (self.RecessionShock and not self.R_shared): # If the recssion actually occurs,
            MrkvNew += 3 # then put everyone into the recession 
            # This is (momentarily) skipped over if the recession state is shared
            # rather than idiosyncratic.  See a few lines below.
        if self.ExtendedUIShock:
            MrkvNew += 6 # put everyone in the extended UI states
        if self.TaxCutShock:
            MrkvNew +=12 # put everyone in the tax cut states
            #might not change if we keep the order, if +12 relates to 1q of the reform
        if (self.ExtendedUIShock and self.TaxCutShock):
            print("Cannot handle UI and TaxCut experiments at the same time (yet)")
            return
        
        # Move agents to those Markov states 
        self.MrkvNow = MrkvNew
        # print(self.MrkvNow)

        # Take the appropriate shock history for each agent, depending on their state
        #J = self.MrkvArray[0].shape[0]
        J = 18
        for j in range(J):
            these = self.MrkvNow == j
            self.history['MrkvNow'][:,these] = self.MrkvHistAll[j,:,:][:,these]
        tax_cut_multiplier = np.ones_like(self.history['MrkvNow'])
        tax_cut_multiplier[np.greater(self.history['MrkvNow'], 11)] *= self.TaxCutIncFactor #$$$$$$$$$$ assumes all markov states above 11 are tax cut states
        employed = np.equal(self.history['MrkvNow']%3, 0)
        self.history['PermShkNow'][employed] = self.perm_shock_fixed_hist[employed]
        self.history['TranShkNow'][employed] = self.tran_shock_fixed_hist[employed]*tax_cut_multiplier[employed]
        unemp_without_benefits = np.equal(self.history['MrkvNow']%3, 1)
        self.history['PermShkNow'][unemp_without_benefits] = 1.0
        self.history['TranShkNow'][unemp_without_benefits] = self.IncUnempNoBenefits
        unemp_with_benefits = np.equal(self.history['MrkvNow']%3, 2)
        self.history['PermShkNow'][unemp_with_benefits] = 1.0
        self.history['TranShkNow'][unemp_with_benefits] = self.IncUnemp
        self.history['who_dies'] = self.who_dies_fixed_hist
        self.history['update_draw'] = self.update_draw_fixed_hist
      
#        NEED TO FIX BELOW IF WE WANT SHARED RECESSION - NECESSARY TO CHANGE IF WE WANT SHOCKS TO BE CONTINGENT ON RECESSION STATE
#        POSSIBLE FIX - TAKE HISTORY FROM PermShkHistCond up to the point where the recession ends, then take history starting 
#        IN NON_RECESSION STATE THAT FOLLOWS AFTER (NEED TO CALC PROBABILITIES OF BEING IN EACH STATE AFTER RECESSION ENDS BASED ON STATE IN RECESSION)    
        if self.R_shared:
            print( "R_shared not implemented yet" )
#        # If the recession is a common/shared event, rather than idiosyncratic, bump
#        # everyone into the lockdown state for *exactly* T_lockdown periods
#        if (self.RecessionShock and self.R_shared):
#            T = self.T_recession
#            self.history['MrkvNow'][0:T,:] += 2
                   
    def getControls(self):
        '''
        Calculates consumption for each consumer of this type using the consumption functions.
        Parameters
        ----------
        None
        Returns
        -------
        None
        '''
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        J = self.MrkvArray[0].shape[0]
        
        MrkvBoolArray = np.zeros((J,self.AgentCount), dtype=bool)
        for j in range(J):
            MrkvBoolArray[j,:] = j == self.MrkvNowPcvd # agents choose control based on *perceived* Markov state
        
        for t in range(self.T_cycle):
            right_t = t == self.t_cycle
            for j in range(J):
                these = np.logical_and(right_t, MrkvBoolArray[j,:])
                cNrmNow[these], MPCnow[these] = self.solution[t].cFunc[j].eval_with_derivative(self.mNrmNow[these])
        self.cNrmNow = cNrmNow
        self.MPCnow  = MPCnow
                    
                
def solveConsMarkovALT(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                 MrkvArray,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption-saving problem with risky income and
    stochastic transitions between discrete states, in a Markov fashion.  Has
    identical inputs as solveConsIndShock, except for a discrete
    Markov transitionrule MrkvArray.  Markov states can differ in their interest
    factor, permanent growth factor, and income distribution, so the inputs Rfree,
    PermGroFac, and IncomeDstn are arrays or lists specifying those values in each
    (succeeding) Markov state.
    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : DiscreteDistribution
        A representation of permanent and transitory income shocks that might
        arrive at the beginning of next period.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : np.array
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroFac : np.array
        Expected permanent income growth factor at the end of this period
        for each Markov state in the succeeding period.
    MrkvArray : np.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.  Not used.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.  Not used.
    Returns
    -------
    solution : ConsumerSolution
        The solution to the single period consumption-saving problem. Includes
        a consumption function cFunc (using cubic or linear splines), a marg-
        inal value function vPfunc, a minimum acceptable level of normalized
        market resources mNrmMin.  All of these attributes are lists or arrays, 
        with elements corresponding to the current Markov state.  E.g.
        solution.cFunc[0] is the consumption function when in the i=0 Markov
        state this period.
    '''
    # Get sizes of grids
    aCount = aXtraGrid.size
    StateCount = MrkvArray.shape[0]

    # Loop through next period's states, assuming we reach each one at a time.
    # Construct EndOfPrdvP_cond functions for each state.
    BoroCnstNat_cond = []
    EndOfPrdvPfunc_cond = []
    for j in range(StateCount):
        # Unpack next period's solution
        vPfuncNext = solution_next.vPfunc[j]
        mNrmMinNext = solution_next.mNrmMin[j]

        # Unpack the income shocks
        ShkPrbsNext = IncomeDstn[j].pmf
        PermShkValsNext = IncomeDstn[j].X[0]
        TranShkValsNext = IncomeDstn[j].X[1]
        ShkCount = ShkPrbsNext.size
        aXtra_tiled = np.tile(np.reshape(aXtraGrid, (aCount, 1)), (1, ShkCount))

        # Make tiled versions of the income shocks
        # Dimension order: aNow, Shk
        ShkPrbsNext_tiled = np.tile(np.reshape(ShkPrbsNext, (1, ShkCount)), (aCount, 1))
        PermShkValsNext_tiled = np.tile(np.reshape(PermShkValsNext, (1, ShkCount)), (aCount, 1))
        TranShkValsNext_tiled = np.tile(np.reshape(TranShkValsNext, (1, ShkCount)), (aCount, 1))

        # Find the natural borrowing constraint
        aNrmMin_candidates = PermGroFac[j]*PermShkValsNext_tiled/Rfree[j]*(mNrmMinNext - TranShkValsNext_tiled[0, :])
        aNrmMin = np.max(aNrmMin_candidates)
        BoroCnstNat_cond.append(aNrmMin)

        # Calculate market resources next period (and a constant array of capital-to-labor ratio)
        aNrmNow_tiled = aNrmMin + aXtra_tiled
        mNrmNext_array = Rfree[j]*aNrmNow_tiled/PermShkValsNext_tiled + TranShkValsNext_tiled

        # Find marginal value next period at every income shock realization and every aggregate market resource gridpoint
        vPnext_array = Rfree[j]*PermShkValsNext_tiled**(-CRRA)*vPfuncNext(mNrmNext_array)

        # Calculate expectated marginal value at the end of the period at every asset gridpoint
        EndOfPrdvP = DiscFac*np.sum(vPnext_array*ShkPrbsNext_tiled, axis=1)

        # Make the conditional end-of-period marginal value function
        EndOfPrdvPnvrs = EndOfPrdvP**(-1./CRRA)
        EndOfPrdvPnvrsFunc = LinearInterp(np.insert(aNrmMin + aXtraGrid, 0, aNrmMin), np.insert(EndOfPrdvPnvrs, 0, 0.0))
        EndOfPrdvPfunc_cond.append(MargValueFunc(EndOfPrdvPnvrsFunc, CRRA))

    # Now loop through *this* period's discrete states, calculating end-of-period
    # marginal value (weighting across state transitions), then construct consumption
    # and marginal value function for each state.
    cFuncNow = []
    vPfuncNow = []
    mNrmMinNow = []
    for i in range(StateCount):
        # Find natural borrowing constraint for this state
        aNrmMin_candidates = np.zeros(StateCount) + np.nan
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:  # Irrelevant if transition is impossible
                aNrmMin_candidates[j] = BoroCnstNat_cond[j]
        aNrmMin = np.nanmax(aNrmMin_candidates)
        
        # Find the minimum allowable market resources
        if BoroCnstArt is not None:
            mNrmMin = np.maximum(BoroCnstArt, aNrmMin)
        else:
            mNrmMin = aNrmMin
        mNrmMinNow.append(mNrmMin)

        # Make tiled grid of aNrm
        aNrmNow = aNrmMin + aXtraGrid
        
        # Loop through feasible transitions and calculate end-of-period marginal value
        EndOfPrdvP = np.zeros(aCount)
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:
                temp = MrkvArray[i, j]*EndOfPrdvPfunc_cond[j](aNrmNow)
                EndOfPrdvP += temp
        EndOfPrdvP *= LivPrb[i] # Account for survival out of the current state

        # Calculate consumption and the endogenous mNrm gridpoints for this state
        cNrmNow = EndOfPrdvP**(-1./CRRA)
        mNrmNow = aNrmNow + cNrmNow

        # Make a piecewise linear consumption function
        c_temp = np.insert(cNrmNow, 0, 0.0)  # Add point at bottom
        m_temp = np.insert(mNrmNow, 0, aNrmMin)
        cFuncUnc = LinearInterp(m_temp, c_temp)
        cFuncCnst = LinearInterp(np.array([mNrmMin, mNrmMin+1.0]), np.array([0.0, 1.0]))
        cFuncNow.append(LowerEnvelope(cFuncUnc,cFuncCnst))

        # Construct the marginal value function using the envelope condition
        m_temp = aXtraGrid + mNrmMin
        c_temp = cFuncNow[i](m_temp)
        uP = c_temp**(-CRRA)
        vPnvrs = uP**(-1./CRRA)
        vPnvrsFunc = LinearInterp(np.insert(m_temp, 0, mNrmMin), np.insert(vPnvrs, 0, 0.0))
        vPfuncNow.append(MargValueFunc(vPnvrsFunc, CRRA))
        
    # Pack up and return the solution
    solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow)
    return solution_now