'''
This file has an extension of MarkovConsumerType that is used for the Fiscal project.
'''
import warnings
import numpy as np
from HARK.distribution import DiscreteDistribution, Uniform
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution
from HARK.ConsumptionSaving.ConsAggShockModel import MargValueFunc2D, AggShockConsumerType
from HARK.interpolation import LinearInterp, BilinearInterp, VariableLowerBoundFunc2D, \
                                LinearInterpOnInterp1D, LowerEnvelope2D, UpperEnvelope, ConstantFunction
from HARK import Market
from HARK.core import distanceMetric, HARKobject
from EstimParameters import makeFullMrkvArray, T_sim, makeCondMrkvArrays_base
from copy import copy, deepcopy
import matplotlib.pyplot as plt

# Define a modified MarkovConsumerType
class AggFiscalType(MarkovConsumerType):
    time_inv_ = MarkovConsumerType.time_inv_ 
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        MarkovConsumerType.__init__(self,cycles=1,time_flow=True,**kwds)
        self.shock_vars += ['update_draw','unemployment_draw']
        self.solveOnePeriod = solveAggConsMarkovALT
        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(MarkovConsumerType.time_vary_)
        self.time_inv = deepcopy(MarkovConsumerType.time_inv_)
        self.delFromTimeInv('vFuncBool', 'CubicBool')
        self.addToTimeVary('IncomeDstn','PermShkDstn','TranShkDstn')
        self.addToTimeInv('aXtraGrid')
        
    def updateSolutionTerminal(self):
        AggShockConsumerType.updateSolutionTerminal(self)
        # Make replicated terminal period solution
        StateCount = self.MrkvArray[-1].shape[0]
        self.solution_terminal.cFunc = StateCount*[self.solution_terminal.cFunc]
        self.solution_terminal.vPfunc = StateCount*[self.solution_terminal.vPfunc]
        self.solution_terminal.mNrmMin = StateCount*[self.solution_terminal.mNrmMin]
        
    def preSolve(self):
        self.MrkvArray = self.MrkvArray
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
        self.MacroMrkvNow = (np.floor(self.MrkvNow/self.num_base_MrkvStates)).astype(int)
        self.MicroMrkvNow = self.MrkvNow%self.num_base_MrkvStates
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
    def getEconomyData(self, Economy):
        '''
        Imports economy-determined objects into self from a Market.
        Parameters
        ----------
        Economy : Market
            The "macroeconomy" in which this instance "lives".  
        Returns
        -------
        None
        '''
        self.T_sim = Economy.act_T                   # Need to be able to track as many periods as economy runs
        self.Cgrid = Economy.CgridBase               # Ratio of consumption to steady state consumption
        self.CFunc = Economy.CFunc                   # Next period's consumption ratio function
        self.ADFunc = Economy.ADFunc                 # Function that takes aggregate consumption to agg. demand function
        self.addToTimeInv('Cgrid', 'CFunc','ADFunc','num_base_MrkvStates')
        # self.PermGroFacAgg = Economy.PermGroFacAgg   # Aggregate permanent productivity growth
        #self.addToTimeInv('Cgrid', 'CFunc', 'PermGroFacAgg','ADFunc')
        
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
        
    def makeIdiosyncraticShockHistories(self):     
        self.Mrkv_univ = 0
        self.read_shocks = False
        self.makeShockHistory()
        self.who_dies_fixed_hist    = self.history['who_dies'].copy()
        self.update_draw_fixed_hist = self.history['update_draw'].copy()
        self.perm_shock_fixed_hist  = self.history['PermShkNow'].copy()
        self.tran_shock_fixed_hist  = self.history['TranShkNow'].copy()
        self.unemployment_draw_fixed_hist = self.history['unemployment_draw'].copy()
        self.Mrkv_univ = None
        
    def hitWithRecessionShock(self, shock_type):
        '''
        Alter the Markov state of each simulated agent, jumping some people into
        recession states
        '''
        # Shock unemployment up to ergodic unemployment level in normal or recession state
        if shock_type=="recession" or shock_type=="recessionUI" or shock_type=="recessionTaxCut" or shock_type=="recessionCheck":
            this_Urate = self.Urate_recession
        elif shock_type=="base":
            this_Urate = self.Urate_normal
        
        # Draw new Markov states for each agents who are employed
        draws = Uniform(seed=self.RNG.randint(0,2**31-1)).draw(self.AgentCount)
        draws = self.RNG.permutation(draws)
        MrkvNew = self.MrkvNow
        old_Urate = self.Urate_normal
        draws_empy2umemp = draws > 1.0-(this_Urate-old_Urate)/(1.0-old_Urate)
        MrkvNew[np.logical_and(np.equal(self.MrkvNow,0), draws_empy2umemp) ] = 1 # Move people from employment to unemployment such that total unemployment rate is as required. Don't touch already unemployed people.
        
        if shock_type=="base":
            MrkvNew = MrkvNew #no shock
        elif shock_type=="recession" or shock_type=="recessionUI" or shock_type=="recessionTaxCut" or shock_type=="recessionCheck": # If the recssion actually occurs,
            MrkvNew += 2*self.num_base_MrkvStates # then put everyone into the recession 
        # Move agents to those Markov states 
        self.MrkvNow = MrkvNew
       
        self.history['MrkvNow'] = np.ones_like(self.history['PermShkNow'])
        t_age_start = copy(self.t_age)
        self.MicroMrkvNow = self.MrkvNow % self.num_base_MrkvStates
        self.MacroMrkvNow = np.floor(self.MrkvNow/self.num_base_MrkvStates).astype(int)
        MicroMrkvNow_start = copy(self.MicroMrkvNow)
        MacroMrkvNow_start = copy(self.MacroMrkvNow)
        for t in range(self.T_sim):
            self.t_age = 1 - self.who_dies_fixed_hist[t] # hack to get newborns have t_age=0
            self.MacroMrkvNow = self.EconomyMrkvNow_hist[t] 
            unemployment_draw = self.unemployment_draw_fixed_hist[t]
            self.getMicroMarkvStates_guts(unemployment_draw)
            MrkvNow = self.num_base_MrkvStates*self.MacroMrkvNow + self.MicroMrkvNow
            self.history['MrkvNow'][t] = MrkvNow.astype(int)
        self.t_age = t_age_start
        self.MicroMrkvNow = MicroMrkvNow_start
        self.MacroMrkvNow = MacroMrkvNow_start
        self.MrkvNow = self.num_base_MrkvStates*self.MacroMrkvNow + self.MicroMrkvNow
        
        tax_cut_multiplier  = np.ones_like(self.history['MrkvNow'])
        CheckAmount         = np.zeros_like(self.history['MrkvNow'])
        if shock_type=="recessionTaxCut":
            tax_cut_states = np.logical_and(np.greater(self.history['MrkvNow'], 2*self.num_base_MrkvStates), np.less(self.history['MrkvNow'],9*2*self.num_base_MrkvStates)) #$$$$$$$$$$ assumes all markov states above 11 and below 36 are tax cut states
            tax_cut_multiplier[tax_cut_states] *= self.TaxCutIncFactor 
        elif shock_type=="recessionCheck":
            CheckAmount = np.equal(self.history['MrkvNow'],3*self.num_base_MrkvStates) * self.CheckStimLvl # only if MrkvState is 3
            #This only works because check occurs in first period
            CheckAmount[0] = CheckAmount[0] / self.pLvlNow        
            for agent in range(len(CheckAmount[0])):
                # Stimulus is a function of permanent income
                if self.pLvlNow[agent] < self.CheckStimLvl_PLvl_Cutoff_start:
                    AgentSpecificScalar = 1
                elif self.pLvlNow[agent] > self.CheckStimLvl_PLvl_Cutoff_end:
                    AgentSpecificScalar = 0
                else:
                    AgentSpecificScalar = 1-(self.pLvlNow[agent]-self.CheckStimLvl_PLvl_Cutoff_start)/(self.CheckStimLvl_PLvl_Cutoff_end-self.CheckStimLvl_PLvl_Cutoff_start)
                CheckAmount[0][agent] *= AgentSpecificScalar
                
        employed = np.equal(self.history['MrkvNow']%self.num_base_MrkvStates, 0)
        self.history['PermShkNow'][employed] = self.perm_shock_fixed_hist[employed]
        self.history['TranShkNow'][employed] = self.tran_shock_fixed_hist[employed]*tax_cut_multiplier[employed] + CheckAmount[employed] / self.perm_shock_fixed_hist[employed]
        unemp_without_benefits = np.equal(self.history['MrkvNow']%self.num_base_MrkvStates, self.num_base_MrkvStates-1)
        self.history['PermShkNow'][unemp_without_benefits] = 1.0
        self.history['TranShkNow'][unemp_without_benefits] = self.IncUnempNoBenefits + CheckAmount[unemp_without_benefits] 
        unemp_with_benefits = np.logical_not(np.logical_or(employed,unemp_without_benefits))
        self.history['PermShkNow'][unemp_with_benefits] = 1.0
        self.history['TranShkNow'][unemp_with_benefits] = self.IncUnemp + CheckAmount[unemp_with_benefits] 
        
        self.history['who_dies'] = self.who_dies_fixed_hist
        self.history['update_draw'] = self.update_draw_fixed_hist
        self.history['unemployment_draw'] = self.unemployment_draw_fixed_hist
        
    def switchToCounterfactualMode(self, shock_type):
        del self.solution
        self.delFromTimeVary('solution')
        self.switch_shock_type(shock_type)
        # Adjust simulation parameters for the counterfactual experiments
        self.T_sim = T_sim
        self.track_vars = ['cNrmNow','pLvlNow','aNrmNow','mNrmNow','MrkvNowPcvd','MacroMrkvNow','MicroMrkvNow','cLvlNow','cLvl_splurgeNow']
        self.use_prestate = None
        self.track_vars += ['unemployment_draw']
        
    def switch_shock_type(self, shock_type):
        # Swap in "big" versions of the Markov-state-varying attributes
        if shock_type == "base":
            self.MrkvArray = self.MrkvArray_base
            self.IncomeDstn = self.IncomeDstn_base
#            self.CondMrkvArrays = self.CondMrkvArrays_recession
        elif shock_type == "recession":
            self.MrkvArray = self.MrkvArray_recession
            self.IncomeDstn = self.IncomeDstn_recession
            self.CondMrkvArrays = self.CondMrkvArrays_recession
        elif shock_type == "recessionUI":
            self.MrkvArray = self.MrkvArray_recessionUI
            self.IncomeDstn = self.IncomeDstn_recessionUI
            self.CondMrkvArrays = self.CondMrkvArrays_recessionUI
        elif shock_type == "recessionTaxCut":
            self.MrkvArray = self.MrkvArray_recessionTaxCut
            self.IncomeDstn = self.IncomeDstn_recessionTaxCut
            self.CondMrkvArrays = self.CondMrkvArrays_recessionTaxCut
        elif shock_type == "recessionCheck":
            self.MrkvArray = self.MrkvArray_recessionCheck
            self.IncomeDstn = self.IncomeDstn_recessionCheck
            self.CondMrkvArrays = self.CondMrkvArrays_recessionCheck
        num_mrkv_states = self.MrkvArray[0].shape[0]
        self.LivPrb = [np.array(self.LivPrb_base*num_mrkv_states)]
        self.PermGroFac =  [np.array(self.PermGroFac_base*num_mrkv_states)]
        self.Rfree = np.array(num_mrkv_states*self.Rfree_base)
        
    def getRfree(self):
        RfreeNow = self.Rfree[self.MrkvNow]*np.ones(self.AgentCount)
        return RfreeNow
    
    def marketAction(self):
        self.simulate(1)
        
    def getCratioNow(self):  # This function exists to be overwritten in StickyE model
        return self.CratioNow*np.ones(self.AgentCount)
    
    def getAggDemandFacNow(self):  
        return self.AggDemandFac*np.ones(self.AgentCount)

    def getShocks(self):
        MarkovConsumerType.getShocks(self)
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow = self.MrkvNow_temp # Make sure real sequence is recorded
        self.update_draw = self.RNG.permutation(np.array(range(self.AgentCount))) # A list permuted integers, low draws will update their aggregate Markov state
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
        # update the idiosyncratic state (employed, unemployed with benefits, unemployed without benefits)
        # but leave the macro state as it is (idiosyncratic state is 'modulo self.num_base_MrkvStates')
        self.MrkvNowPcvd = np.remainder(self.MrkvNow,self.num_base_MrkvStates) + self.num_base_MrkvStates*np.floor_divide(self.MrkvNowPcvd,self.num_base_MrkvStates)
        self.mNrmNow = self.bNrmNow + self.TranShkNow*self.AggDemandFac # Market resources after income accounting for Agg Demand factor (this is for simulation)
        
    def getMacroMarkovStates(self):
        self.MacroMrkvNow = self.EconomyMrkvNow*np.ones(self.AgentCount, dtype=int)
        
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
        MrkvNow = self.num_base_MrkvStates*self.MacroMrkvNow + self.MicroMrkvNow
        self.MrkvNow = MrkvNow.astype(int)
        if (hasattr(self,'Mrkv_univ') and self.Mrkv_univ is not None):
            self.MrkvNow_temp = self.MrkvNow
            self.MrkvNow = self.Mrkv_univ*np.ones(self.AgentCount, dtype=int)
            # ^^ Store the real states but force income shocks to be based on one particular state
            
    def updateMrkvArray(self, shock_type):
        if shock_type=="base":
            self.MacroMrkvArray = np.array([[1.0]])
            self.CondMrkvArrays = makeCondMrkvArrays_base(self.Urate_normal, self.Uspell_normal, self.UBspell_normal)
            self.MrkvArray = makeFullMrkvArray(self.MacroMrkvArray, self.CondMrkvArrays)
        else:
            print("shock_type not recognized")
    
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
                   
    def getControls(self):
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        CratioNow = self.getCratioNow()
        J = self.MrkvArray[0].shape[0]
        
        MrkvBoolArray = np.zeros((J,self.AgentCount), dtype=bool)
        for j in range(J):
            MrkvBoolArray[j,:] = j == self.MrkvNowPcvd # agents choose control based on *perceived* Markov state
        
        for t in range(self.T_cycle):
            right_t = t == self.t_cycle
            for j in range(J):
                these = np.logical_and(right_t, MrkvBoolArray[j,:])
                cNrmNow[these] = self.solution[t].cFunc[j](self.mNrmNow[these], CratioNow[these])
                # Marginal propensity to consume
                MPCnow[these]  = self.solution[t].cFunc[j].derivativeX(self.mNrmNow[these], CratioNow[these])
        self.cNrmNow = cNrmNow
        self.MPCnow  = MPCnow
        self.cLvlNow = cNrmNow*self.pLvlNow
        #self.cLvl_splurgeNow = (1.0-self.Splurge)*self.cLvlNow + self.Splurge*self.pLvlNow*self.TranShkNow
        self.cLvl_splurgeNow = (1.0-self.Splurge)*self.cLvlNow + self.Splurge*self.pLvlNow*self.TranShkNow*self.AggDemandFac   #added last term relaive to Edmund's Version
        
    def reset(self):
        return # do nothing
                    
                
def solveAggConsMarkovALT(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                 MrkvArray,BoroCnstArt,aXtraGrid, Cgrid, CFunc, ADFunc,
                                 num_base_MrkvStates):
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
    Ccount = Cgrid.size
    StateCount = MrkvArray.shape[0]
    
    # Loop through next period's states, assuming we reach each one at a time.
    # Construct EndOfPrdvP_cond functions for each state.
    EndOfPrdvPfunc_cond = []
    BoroCnstNat_cond = []
    for j in range(StateCount):
        # Unpack next period's solution
        vPfuncNext = solution_next.vPfunc[j]
        mNrmMinNext = solution_next.mNrmMin[j]

        # Unpack the income shocks
        ShkPrbsNext = IncomeDstn[j].pmf
        PermShkValsNext = IncomeDstn[j].X[0]
        TranShkValsNext = IncomeDstn[j].X[1]
        ShkCount = ShkPrbsNext.size
        aXtra_tiled = np.tile(np.reshape(aXtraGrid, (1, aCount, 1)), (Ccount, 1, ShkCount))

        # Make tiled versions of the income shocks
        # Dimension order: aNow, Shk
        ShkPrbsNext_tiled = np.tile(np.reshape(ShkPrbsNext, (1, 1, ShkCount)), (Ccount, aCount, 1))
        PermShkValsNext_tiled = np.tile(np.reshape(PermShkValsNext, (1, 1, ShkCount)), (Ccount, aCount, 1))
        TranShkValsNext_tiled_noAD = np.tile(np.reshape(TranShkValsNext, (1, 1, ShkCount)), (Ccount, aCount, 1))
        
        # Calculate aggregate consumption next period
        Cnext_array = np.tile(np.reshape(Cgrid, (Ccount, 1, 1)), (1, aCount, ShkCount)) 

        # Calculate AggDemandFac
        AggState = np.floor(j/num_base_MrkvStates)
        RecState = AggState % 2 == 1
        AggDemandFacnext_array = ADFunc(Cnext_array,RecState)
        TranShkValsNext_tiled = AggDemandFacnext_array*TranShkValsNext_tiled_noAD
        
        # Find the natural borrowing constraint for each value of C in the Cgrid.
        aNrmMin_candidates = PermGroFac[j]*PermShkValsNext_tiled[:, 0, :]/Rfree[j]* \
            (mNrmMinNext(Cnext_array[:, 0, :]) - TranShkValsNext_tiled[:, 0, :])
        aNrmMin_vec = np.max(aNrmMin_candidates, axis=1)
        BoroCnstNat_vec = aNrmMin_vec
        aNrmMin_tiled = np.tile(np.reshape(aNrmMin_vec, (Ccount, 1, 1)), (1, aCount, ShkCount))
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled


        # Calculate market resources next period (and a constant array of capital-to-labor ratio)
        mNrmNext_array = Rfree[j]*aNrmNow_tiled/PermShkValsNext_tiled + TranShkValsNext_tiled

        # Find marginal value next period at every income shock realization and every aggregate market resource gridpoint
        vPnext_array = Rfree[j]*PermShkValsNext_tiled**(-CRRA)*vPfuncNext(mNrmNext_array, Cnext_array)

        # Calculate expectated marginal value at the end of the period at every asset gridpoint
        EndOfPrdvP = DiscFac*np.sum(vPnext_array*ShkPrbsNext_tiled, axis=2)
        
        # Make the conditional end-of-period marginal value function
        BoroCnstNat = LinearInterp(Cgrid, BoroCnstNat_vec)
        EndOfPrdvPnvrs = np.concatenate((np.zeros((Ccount, 1)), EndOfPrdvP**(-1./CRRA)), axis=1)
        EndOfPrdvPnvrsFunc_base = BilinearInterp(np.transpose(EndOfPrdvPnvrs), np.insert(aXtraGrid, 0, 0.0), Cgrid)
        EndOfPrdvPnvrsFunc = VariableLowerBoundFunc2D(EndOfPrdvPnvrsFunc_base, BoroCnstNat)
        EndOfPrdvPfunc_cond.append(MargValueFunc2D(EndOfPrdvPnvrsFunc, CRRA))
        BoroCnstNat_cond.append(BoroCnstNat)
        
    # Prepare some objects that are the same across all current states
    aXtra_tiled = np.tile(np.reshape(aXtraGrid, (1, aCount)), (Ccount, 1))
    cFuncCnst = BilinearInterp(np.array([[0.0, 0.0], [1.0, 1.0]]),
                               np.array([BoroCnstArt, BoroCnstArt+1.0]), np.array([0.0, 1.0]))

    # Now loop through *this* period's discrete states, calculating end-of-period
    # marginal value (weighting across state transitions), then construct consumption
    # and marginal value function for each state.
    cFuncNow = []
    vPfuncNow = []
    mNrmMinNow = []
    for i in range(StateCount):
        # Find natural borrowing constraint for this state by Cratio NOTE THIS CODE IS NOT 100% CHECKED AND SHOULD BE LOOKED OVER
        aNrmMin_candidates = np.zeros((StateCount, Ccount)) + np.nan
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:  # Irrelevant if transition is impossible
                Cnext = CFunc[i][j](Cgrid)
                aNrmMin_candidates[j, :] = BoroCnstNat_cond[j](Cnext)
        aNrmMin_vec = np.nanmax(aNrmMin_candidates, axis=0)
        BoroCnstNat_vec = aNrmMin_vec

        # Make tiled grids of aNrm and Cratio
        aNrmMin_tiled = np.tile(np.reshape(aNrmMin_vec, (Ccount, 1)), (1, aCount))
        aNrmNow_tiled = aNrmMin_tiled + aXtra_tiled

        
        # # Find the minimum allowable market resources
        # if BoroCnstArt is not None:
        #     mNrmMin = np.maximum(BoroCnstArt, aNrmMin)
        # else:
        #     mNrmMin = aNrmMin
        # mNrmMinNow.append(mNrmMin)
        
        # Loop through feasible transitions and calculate end-of-period marginal value
        EndOfPrdvP = np.zeros((Ccount, aCount))
        for j in range(StateCount):
            if MrkvArray[i, j] > 0.:
                Cnext = CFunc[i][j](Cgrid)
                Cnext_tiled = np.tile(np.reshape(Cnext, (Ccount, 1)), (1, aCount))
                temp = EndOfPrdvPfunc_cond[j](aNrmNow_tiled, Cnext_tiled)
                EndOfPrdvP += MrkvArray[i, j]*temp                    
        EndOfPrdvP *= LivPrb[i] # Account for survival out of the current state
        
        # Calculate consumption and the endogenous mNrm gridpoints for this state
        cNrmNow = EndOfPrdvP**(-1./CRRA)
        mNrmNow = aNrmNow_tiled + cNrmNow

        # Loop through the values in Cgrid and make a piecewise linear consumption function for each
        cFuncBaseByC_list = []
        for n in range(Ccount):
            c_temp = np.insert(cNrmNow[n, :], 0, 0.0)  # Add point at bottom
            m_temp = np.insert(mNrmNow[n, :] - BoroCnstNat_vec[n], 0, 0.0)
            cFuncBaseByC_list.append(LinearInterp(m_temp, c_temp))
            # Add the C-specific consumption function to the list
            
        # Construct the unconstrained consumption function by combining the C-specific functions
        BoroCnstNat = LinearInterp(Cgrid, BoroCnstNat_vec)
        cFuncBase = LinearInterpOnInterp1D(cFuncBaseByC_list, Cgrid)
        cFuncUnc = VariableLowerBoundFunc2D(cFuncBase, BoroCnstNat)

        # Combine the constrained consumption function with unconstrained component
        cFuncNow.append(LowerEnvelope2D(cFuncUnc, cFuncCnst))

        # Make the minimum m function as the greater of the natural and artificial constraints
        mNrmMinNow.append(UpperEnvelope(BoroCnstNat, ConstantFunction(BoroCnstArt)))

        # Construct the marginal value function using the envelope condition
        vPfuncNow.append(MargValueFunc2D(cFuncNow[-1], CRRA))

    # Pack up and return the solution
    solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=mNrmMinNow)
    return solution_now


class AggregateDemandEconomy(Market):
    '''
    A class to represent an economy in which productivity responds to aggregate
    consumption
    '''
    def __init__(self,
                 agents=None,
                 **kwds):
        '''
        Make a new instance of AggregateDemandEconomy by filling in attributes
        specific to this kind of market.
        '''
        agents = agents if agents is not None else list()

        Market.__init__(self, agents=agents,
                        sow_vars=['CratioNow', 'AggDemandFac', 'AggDemandFacPrev','EconomyMrkvNow'],
                        reap_vars=['cLvl_splurgeNow'],
                        track_vars=['CratioNow','CratioPrev', 'AggDemandFac', 'AggDemandFacPrev','EconomyMrkvNow'],
                        dyn_vars=['CFunc'],
                        **kwds)
        self.update()


    def millRule(self, cLvl_splurgeNow):
        if self.Shk_idx==0:
            EconomyMrkvNow = 0
        else:
            EconomyMrkvNow = self.EconomyMrkvNow_hist[self.Shk_idx-1]   
        EconomyMrkvNext = self.EconomyMrkvNow_hist[self.Shk_idx]
        if hasattr(self,'base_AggCons'):
            cLvl_all_splurge = np.concatenate([this_cLvl for this_cLvl in cLvl_splurgeNow])      
            AggCons   = np.sum(cLvl_all_splurge)
            self.CratioNow = AggCons/self.base_AggCons[self.Shk_idx] 
            CratioNext = self.CFunc[EconomyMrkvNow*self.num_base_MrkvStates][EconomyMrkvNext*self.num_base_MrkvStates](self.CratioNow)
        else:
            self.CratioNow = 1.0
            CratioNext = 1.0
        self.AggDemandFacPrev = self.AggDemandFac
        self.CratioPrev = self.CratioNow
        RecState = EconomyMrkvNext % 2 == 1
        AggDemandFacNext = self.ADFunc(CratioNext,RecState)
        mill_return = HARKobject()
        mill_return.CratioNow = CratioNext
        mill_return.AggDemandFac = AggDemandFacNext
        mill_return.AggDemandFacPrev = self.AggDemandFacPrev
        mill_return.EconomyMrkvNow = EconomyMrkvNext
        self.Shk_idx += 1
        return mill_return

    def calcDynamics(self):
        return self.calcCFunc()

    def update(self):
        '''
        '''
        self.CratioNow_init = 1.0
        self.AggDemandFac_init = 1.0
        self.AggDemandFacPrev_init = 1.0
        self.ADFunc = lambda C, RecState : C**(RecState*self.ADelasticity)
        #self.ADFunc = lambda C, RecState : C**(self.ADelasticity) #in case AD effects are independent of recession state
        self.EconomyMrkvNow_hist = [0] * self.act_T
        StateCount = self.MrkvArray[0].shape[0]
        CFunc_all = []
        for i in range(StateCount):
            CFunc_i = []
            for j in range(StateCount):
                CFunc_i.append(CRule(self.intercept_prev[i,j], self.slope_prev[i,j]))
            CFunc_all.append(copy(CFunc_i))
        self.CFunc = CFunc_all
        for agent in self.agents:
            agent.getEconomyData(self)

    def reset(self):
        self.Shk_idx = 0
        Market.reset(self)
        #self.EconomyMrkvNow_hist = [0] * self.act_T
        for agent in self.agents:
            agent.initializeSim()
        
    def runExperiment(self, shock_type = "recession", UpdatePrb = 1.0, Splurge = 0.0, EconomyMrkv_init = [0], Full_Output = True):
        # Make the macro markov history
        self.EconomyMrkvNow_hist = [0] * self.act_T
        self.EconomyMrkvNow_hist[0:len(EconomyMrkv_init)] = EconomyMrkv_init
    
        self.CratioNow_init = self.CFunc[0][EconomyMrkv_init[0]*self.num_base_MrkvStates].intercept
        RecState = EconomyMrkv_init[0] % 2 == 1
        self.AggDemandFac_init = self.ADFunc(self.CratioNow_init,RecState)
        
        # Make dictionaries of parameters to give to the agents
        experiment_dict = {
                'use_prestate' : True,
                'shock_type' : shock_type,
                'UpdatePrb' : UpdatePrb
                }
          
        # Begin the experiment by resetting each type's state to the baseline values
        PopCount = 0
        for ThisType in self.agents:
            ThisType.read_shocks = True
            ThisType(**experiment_dict)
            ThisType.updateMrkvArray(shock_type)
            ThisType.solveIfChanged()
            ThisType.initializeSim()
            ThisType.EconomyMrkvNow_hist = self.EconomyMrkvNow_hist
            ThisType.hitWithRecessionShock(shock_type)
            PopCount += ThisType.AgentCount
        self.makeHistory()
        
        
           
        # Extract simulated consumption, labor income, and weight data
        cNrm_all    = np.concatenate([ThisType.history['cNrmNow'] for ThisType in self.agents], axis=1)
        Mrkv_hist   = np.concatenate([ThisType.history['MrkvNow'] for ThisType in self.agents], axis=1)
        pLvl_all    = np.concatenate([ThisType.history['pLvlNow'] for ThisType in self.agents], axis=1)
        TranShk_all = np.concatenate([ThisType.history['TranShkNow'] for ThisType in self.agents], axis=1)
        mNrm_all    = np.concatenate([ThisType.history['mNrmNow'] for ThisType in self.agents], axis=1)
        aNrm_all    = np.concatenate([ThisType.history['aNrmNow'] for ThisType in self.agents], axis=1)
        cLvl_all    = np.concatenate([ThisType.history['cLvlNow'] for ThisType in self.agents], axis=1)
        cLvl_all_splurge = np.concatenate([ThisType.history['cLvl_splurgeNow'] for ThisType in self.agents], axis=1)
        
        IndIncome = pLvl_all*TranShk_all*np.array(self.history['AggDemandFacPrev'])[:,None]
        AggIncome = np.sum(IndIncome,1)
        AggCons   = np.sum(cLvl_all_splurge,1)
        
        # Function calculates the net present value of X, which can be income or consumption
        # Periods defintes the horizon of the NPV measure, R the interest rate at which future income is discounted
        def calculate_NPV(X,Periods,R):
            NPV_discount = np.zeros(Periods)
            for t in range(Periods):
                NPV_discount[t] = 1/(R**t)
            NPV = np.zeros(Periods)
            for t in range(Periods):
                NPV[t] = np.sum(X[0:t+1]*NPV_discount[0:t+1])    
            return NPV

        # calculate NPV
        NPV_AggIncome = calculate_NPV(AggIncome,self.act_T,ThisType.Rfree[0])
        NPV_AggCons   = calculate_NPV(AggCons,self.act_T,ThisType.Rfree[0])
        
        # calculate Cratio_hist
        if hasattr(self,'base_AggCons'):
            Cratio_hist = np.divide(AggCons,self.base_AggCons)
        else:
            Cratio_hist = np.divide(AggCons,AggCons)
        
                
        # Get initial Markov states
        Mrkv_init = np.concatenate([ThisType.history['MrkvNow'][0,:] for ThisType in self.agents])
        
        if Full_Output:
            return_dict = {'cNrm_all' :     cNrm_all,
                           'TranShk_all' :  TranShk_all,
                           'cLvl_all' :     cLvl_all,
                           'pLvl_all' :     pLvl_all,
                           'Mrkv_hist' :    Mrkv_hist,
                           'Mrkv_init' :    Mrkv_init,
                           'mNrm_all' :     mNrm_all,
                           'aNrm_all' :     aNrm_all,
                           'cLvl_all_splurge' : cLvl_all_splurge,
                           'NPV_AggIncome': NPV_AggIncome,
                           'NPV_AggCons':   NPV_AggCons,
                           'AggIncome':     AggIncome,
                           'AggCons':       AggCons,
                           'Cratio_hist' :  Cratio_hist}
        else:
            return_dict = {'NPV_AggIncome': NPV_AggIncome,
                           'NPV_AggCons':   NPV_AggCons,
                           'AggIncome':     AggIncome,
                           'AggCons':       AggCons,
                           'Cratio_hist':   Cratio_hist}    
                
        return return_dict

    def calcCFunc(self):
        StateCount = self.MrkvArray[0].shape[0]
        CFunc_all = []
        for i in range(StateCount):
            CFunc_i = []
            for j in range(StateCount):
                CFunc_i.append(CRule(self.intercept_prev[i,j], self.slope_prev[i,j]))
            CFunc_all.append(copy(CFunc_i))
        self.CFunc = CFunc_all
        
    def switchToCounterfactualMode(self, shock_type):
        '''
        Very small method that swaps in the "big" Markov-state versions of some
        solution attributes, replacing the "small" two-state versions that are used
        only to generate the pre-recession initial distbution of state variables.
        It then prepares this type to create alternate shock histories so it can
        run counterfactual experiments.
        '''       
        # Adjust simulation parameters for the counterfactual experiments
        self.switch_shock_type(shock_type)
        self.act_T = T_sim
        for agent in self.agents:
            agent.getEconomyData(self)
            agent.switchToCounterfactualMode(shock_type)
            
    def switch_shock_type(self, shock_type):
        if shock_type == "base":
            self.MrkvArray = self.MrkvArray_base
        elif shock_type == "recession":
            self.MrkvArray = self.MrkvArray_recession
        elif shock_type == "recessionUI":
            self.MrkvArray = self.MrkvArray_recessionUI
        elif shock_type == "recessionTaxCut":
            self.MrkvArray = self.MrkvArray_recessionTaxCut
        elif shock_type == "recessionCheck":
            self.MrkvArray = self.MrkvArray_recessionCheck
        num_mrkv_states = self.MrkvArray[0].shape[0]
        self.intercept_prev = np.ones((num_mrkv_states,num_mrkv_states ))
        self.slope_prev    = np.zeros((num_mrkv_states,num_mrkv_states ))
        self.calcCFunc()
        for agent in self.agents:
            agent.switch_shock_type(shock_type)
            agent.getEconomyData(self)
            
    def saveState(self):
        for agent in self.agents:
            agent.saveState()
            
    def storeBaseline(self, AggCons):
        self.base_AggCons = copy(AggCons)
        self.stored_solutions = dict()
        self.storeADsolution('baseline')
            
    def storeADsolution(self, name):
        self.stored_solutions[name] = HARKobject()
        self.stored_solutions[name].CFunc = copy(self.CFunc)
        self.stored_solutions[name].ADelasticity = self.ADelasticity
        self.stored_solutions[name].agent_solutions = []
        for i in range(len(self.agents)):
            self.stored_solutions[name].agent_solutions.append(copy(self.agents[i].solution))
                       
    def restoreADsolution(self,name):
        self.CFunc = self.stored_solutions[name].CFunc
        self.ADelasticity = self.stored_solutions[name].ADelasticity
        for i in range(len(self.agents)):
            self.agents[i].solution = self.stored_solutions[name].agent_solutions[i]
            self.agents[i].getEconomyData(self)
        
    def makeIdiosyncraticShockHistories(self):
        for agent in self.agents:
            agent.makeIdiosyncraticShockHistories()
            
    def solve(self):
        for agent in self.agents:
            agent.solve()
            
    def Macro2MicroCFunc(self, MacroCFunc):
        '''
        Converts the aggregate CFunc for Macro transitions to one for micro transitions
        '''
        dim = len(MacroCFunc)
        MicroCFunc = [[CRule(1.0,0.0) for i in range(dim*self.num_base_MrkvStates)] for j in range(dim*self.num_base_MrkvStates)]
        for i in range(dim*self.num_base_MrkvStates):
            for j in range(dim*self.num_base_MrkvStates):
                MicroCFunc[i][j] = MacroCFunc[int(np.floor(i/self.num_base_MrkvStates))][int(np.floor(j/self.num_base_MrkvStates))]
        return MicroCFunc
    
    def CompareCFuncConvergence(self,Old_Cfunc,New_Cfunc):
        dim=len(Old_Cfunc)
        DiffSlopes      = np.zeros((dim,dim))
        DiffIntercepts  = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                DiffSlopes[i,j]     = abs(New_Cfunc[i][j].slope - Old_Cfunc[i][j].slope)
                DiffIntercepts[i,j] = abs(New_Cfunc[i][j].intercept - Old_Cfunc[i][j].intercept)
        Slopes_Diff                         = np.linalg.norm(DiffSlopes)
        [i,j]                               = np.unravel_index(DiffSlopes.argmax(),DiffSlopes.shape)
        FromMrkState_Slopes_Largest_Diff    = int(np.floor(i/self.num_base_MrkvStates))
        ToMrkState_Slopes_Largest_Diff      = int(np.floor(j/self.num_base_MrkvStates))
        
        Intercept_Diff                      = np.linalg.norm(DiffIntercepts)
        [i,j]                               = np.unravel_index(DiffIntercepts.argmax(),DiffIntercepts.shape)
        FromMrkState_Intercept_Largest_Diff = int(np.floor(i/self.num_base_MrkvStates))
        ToMrkState_Intercept_Largest_Diff   = int(np.floor(j/self.num_base_MrkvStates))
        
        Total_Diff          = (Slopes_Diff**2 + Intercept_Diff**2)**0.5
        # print('Diff in Slopes in CFunc: ', Slopes_Diff)
        # print('Largest diff', np.max(DiffSlopes))
        # print('Slope: Largest Diff from Mrk State: ', FromMrkState_Slopes_Largest_Diff)
        # print('Slope: Largest Diff to Mrk State: ', ToMrkState_Slopes_Largest_Diff)
        
        # print('Diff in Intercepts in CFunc: ', Intercept_Diff) 
        # print('Largest diff', np.max(DiffIntercepts))
        # print('Intercept: Largest Diff from Mrk State: ', FromMrkState_Intercept_Largest_Diff)
        # print('Intercept: Largest Diff to Mrk State: ', ToMrkState_Intercept_Largest_Diff)
        print('Total Diff in CFunc: ', Total_Diff)
        return Total_Diff
          

class CRule(HARKobject):
    '''
    A class to represent agent beliefs about aggregate consumption dynamics.
    '''
    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope
        self.distance_criteria = ['slope', 'intercept']

    def __call__(self, Cnow):
        #Cnext = np.exp(self.intercept + self.slope*np.log(Cnow))
        Cnext = self.intercept + self.slope*(Cnow-1.0)        # Not logs!
        return Cnext