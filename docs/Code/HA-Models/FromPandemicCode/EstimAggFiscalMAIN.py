'''
This is the main script for estimating the discount factor distributions.
'''
from time import time
import sys 
import os 
from importlib import reload 
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import namedtuple 
import pickle
import random 
from HARK.distribution import DiscreteDistribution, Uniform
from HARK import multiThreadCommands, multiThreadCommandsFake
from HARK.utilities import getPercentiles, getLorenzShares
from HARK.estimation import minimizeNelderMead

cwd             = os.getcwd()
folders         = cwd.split(os.path.sep)
top_most_folder = folders[-1]
if top_most_folder == 'FromPandemicCode':
    Abs_Path = cwd
    figs_dir = '../../../Figures'
    res_dir = '../Results'
else:
    Abs_Path = cwd + '\\FromPandemicCode'
    figs_dir = '../../Figures'
    res_dir = 'Results'
sys.path.append(Abs_Path)

import EstimParameters as ep
reload(ep)  # Force reload in case the code is running from commandline for different values 

from EstimParameters import init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
     DiscFacCount, CRRA, Splurge, IncUnemp, IncUnempNoBenefits, AgentCountTotal, base_dict, \
     UBspell_normal, data_LorenzPts, data_LorenzPtsAll, data_avgLWPI, data_LWoPI, \
     data_medianLWPI, data_EducShares, data_WealthShares, Rfree_base, \
     GICmaxBetas, GICfactor, minBeta
from AggFiscalModel import AggFiscalType, AggregateDemandEconomy
mystr = lambda x : '{:.2f}'.format(x)
mystr4 = lambda x : '{:.4f}'.format(x)

# -----------------------------------------------------------------------------
def calcEstimStats(Agents):
    '''
    Calculate the average LW/PI-ratio and total LW / total PI for each education
    type. Also calculate the 20th, 40th, 60th, and 80th percentile points of the
    Lorenz curve for (liquid) wealth for all agents. 
    Assumption: Agents is organized by EducType and there are DiscFacCount
    AgentTypes of each EducType. 
    
    Parameters
    ----------
    Agents : [AgentType]
        List of AgentTypes in the economy.
        
    Returns
    -------
    Stats : namedtuple("avgLWPI", "LWoPI", "LorenzPts")
    avgLWPI : [float] 
        The weighted average of LW/PI-ratio for each education type.
    LWoPI : [float]
        Total liquid wealth / total permanent income for each education type. 
    LorenzPts : [float]
        The 20th, 40th, 60th, and 80th percentile points of the Lorenz curve for 
        (liquid) wealth.
    '''

    aLvlAll = np.concatenate([ThisType.aLvlNow for ThisType in Agents])
    numAgents = 0
    for ThisType in Agents: 
        numAgents += ThisType.AgentCount
    weights = np.ones(numAgents) / numAgents      # just using equal weights for now

    # Lorenz points:
    LorenzPts = 100*getLorenzShares(aLvlAll, weights=weights, percentiles = [0.2, 0.4, 0.6, 0.8] )

    avgLWPI = [0]*num_types
    LWoPI = [0]*num_types 
    medianLWPI = [0]*num_types 
    for e in range(num_types):
        aNrmAll_byEd = []
        aNrmAll_byEd = np.concatenate([ThisType.aNrmNow for ThisType in \
                          Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        weights = np.ones(len(aNrmAll_byEd))/len(aNrmAll_byEd)
        avgLWPI[e] = np.dot(aNrmAll_byEd, weights) * 100
        
        aLvlAll_byEd = []
        aLvlAll_byEd = np.concatenate([ThisType.aLvlNow for ThisType in \
                          Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        pLvlAll_byEd = []
        pLvlAll_byEd = np.concatenate([ThisType.pLvlNow for ThisType in \
                          Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        LWoPI[e] = np.dot(aLvlAll_byEd, weights) / np.dot(pLvlAll_byEd, weights) * 100

        medianLWPI[e] = 100*getPercentiles(aNrmAll_byEd,weights=weights,percentiles=[0.5])

    Stats = namedtuple("Stats", ["avgLWPI", "LWoPI", "medianLWPI", "LorenzPts"])

    return Stats(avgLWPI, LWoPI, medianLWPI, LorenzPts) 
# -----------------------------------------------------------------------------
def calcWealthShareByEd(Agents):
    '''
    Calculate the share of total wealth held by each education type. 
    Assumption: Agents is organized by EducType and there are DiscFacCount
    AgentTypes of each EducType. 
    
    Parameters
    ----------
    Agents : [AgentType]
        List of all AgentTypes in the economy. They are assumed to differ in 
        their EducType attribute.

    Returns
    -------
    WealthShares : np.array(float)
        The share of total liquid wealth held by each education type. 
    '''
    aLvlAll = np.concatenate([ThisType.aLvlNow for ThisType in Agents])
    totLiqWealth = np.sum(aLvlAll)
    
    WealthShares = [0]*num_types
    for e in range(num_types):
        aLvlAll_byEd = []
        aLvlAll_byEd = np.concatenate([ThisType.aLvlNow for ThisType in \
                                       Agents[e*DiscFacCount:(e+1)*DiscFacCount]])
        WealthShares[e] = np.sum(aLvlAll_byEd)/totLiqWealth * 100
    
    return np.array(WealthShares)
# -----------------------------------------------------------------------------
def calcLorenzPts(Agents):
    '''
    Calculate the 20th, 40th, 60th, and 80th percentile points of the
    Lorenz curve for (liquid) wealth for the given set of Agents. 

    Parameters
    ----------
    Agents : [AgentType]
        List of AgentTypes.

    Returns
    -------
    LorenzPts : [float]
        The 20th, 40th, 60th, and 80th percentile points of the Lorenz curve for 
        (liquid) wealth.
    '''
    aLvlAll = np.concatenate([ThisType.aLvlNow for ThisType in Agents])
    numAgents = 0
    for ThisType in Agents: 
        numAgents += ThisType.AgentCount
    weights = np.ones(numAgents) / numAgents      # just using equal weights for now
    
    # Lorenz points:
    LorenzPts = 100*getLorenzShares(aLvlAll, weights=weights, percentiles = [0.2, 0.4, 0.6, 0.8] )

    return LorenzPts
# -----------------------------------------------------------------------------
def calcMPCbyEd(Agents):
    '''
    Calculate the average MPC for each education type. 
    Assumption: Agents is organized by EducType and there are DiscFacCount
    AgentTypes of each EducType. 
    
    Parameters
    ----------
    Agents : [AgentType]
        List of all AgentTypes in the economy. They are assumed to differ in 
        their EducType attribute.

    Returns
    -------
    MPCs : namedtuple("MPCsQ", "MPCsA")    
    MPCsQ : [float]
        The average MPC for each education type - Quarterly, ignores splurge.
    MPCsA : [float]
        The average MPC for each education type - Annualized, taking splurge into account. 
        (Only splurge in the first quarter.)
    '''
    MPCsQ = [0]*(num_types+1)
    MPCsA = [0]*(num_types+1)       # Annual MPCs with splurge
    for e in range(num_types):
        MPC_byEd_Q = []
        MPC_byEd_Q = np.concatenate([ThisType.MPCnow for ThisType in \
                                       Agents[e*DiscFacCount:(e+1)*DiscFacCount]])

        MPC_byEd_A = Splurge + (1-Splurge)*MPC_byEd_Q
        for qq in range(3):
            MPC_byEd_A += (1-MPC_byEd_A)*MPC_byEd_Q
        
        MPCsQ[e] = np.mean(MPC_byEd_Q)
        MPCsA[e] = np.mean(MPC_byEd_A)
        
    MPC_all_Q = np.concatenate([ThisType.MPCnow for ThisType in Agents])
    MPC_all_A = Splurge + (1-Splurge)*MPC_all_Q
    for qq in range(3):
        MPC_all_A += (1-MPC_all_A)*MPC_all_Q
    
    MPCsQ[e+1] = np.mean(MPC_all_Q)
    MPCsA[e+1] = np.mean(MPC_all_A)

    MPCs = namedtuple("MPCs", ["MPCsQ", "MPCsA"])
 
    return MPCs(MPCsQ,MPCsA)
 
# -----------------------------------------------------------------------------
def checkDiscFacDistribution(beta, nabla, educ_type, print_mode=False, print_file=False, filename='DefaultResultsFile.txt'):
    '''
    Calculate max and min discount factors in discrete approximation to uniform 
    distribution of discount factors. Also report if most patient agents satisfies 
    the growth impatience condition. 
    
    Parameters
    ----------
    beta : float
        Central value of the discount factor distribution for this education group.
    nabla : float
        Half the width of the discount factor distribution.
    educ_type : int 
        Denotes the education type (either 0, 1 or 2). 
    print_mode : boolean, optional
        If true, results are printed to the screen. The default is False.
    print_file : boolean, optional
        If true, statistics are appended to the file filename. The default is False. 
    filename : str
        Filename for printing calculated statistics. The default is DefaultResultsFile.txt.
    
    Returns
    -------
    dfCheck : namedtuple("betaMin", "betaMax", "GICsatisfied")    
    betaMin : float
        Minimum value in discrete approximation to discount factor distribution.
    betaMax : float
        Maximum value in discrete approximation to discount factor distribution.
    GICsatisfied : boolean
        True if betaMax satisfies the GIC for this education group. 
    '''
    DiscFacDstn = Uniform(beta-nabla, beta+nabla).approx(DiscFacCount)
    betaMin = DiscFacDstn.X[0]
    betaMax = DiscFacDstn.X[DiscFacCount-1]
    GICsatisfied = (betaMax < GICmaxBetas[educ_type])

    if print_mode:
        print('Approximation to beta distribution: betaMin = '+str(round(betaMin,5))
                          +', betaMax = '+str(round(betaMax,5))+'\n')
        print('GIC satisfied = '+str(GICsatisfied)+'\n')
        print('Imposed GIC consistent maximum beta = ' + str(round(GICmaxBetas[educ_type]*GICfactor,5))+'\n\n')
        
    if print_file:
        with open(filename, 'a') as resFile: 
            resFile.write('\tApproximation to beta distribution: betaMin = '+str(round(betaMin,5))
                          +', betaMax = '+str(round(betaMax,5))+'\n')
            resFile.write('\tGIC satisfied = '+str(GICsatisfied)+'\n')
            resFile.write('\tImposed GIC-consistent maximum beta = ' + str(round(GICmaxBetas[educ_type]*GICfactor,5))+'\n\n')
    
    dfCheck = namedtuple("dfCheck", ["betaMin", "betaMax", "GICsatisfied"])
    return dfCheck(betaMin, betaMax, GICsatisfied)    

# =============================================================================
#%% Initialize economy
# Make education types
num_types = 3
# This is not the number of discount factors, but the number of household types

InfHorizonTypeAgg_d = AggFiscalType(**init_dropout)
InfHorizonTypeAgg_d.cycles = 0
InfHorizonTypeAgg_h = AggFiscalType(**init_highschool)
InfHorizonTypeAgg_h.cycles = 0
InfHorizonTypeAgg_c = AggFiscalType(**init_college)
InfHorizonTypeAgg_c.cycles = 0
AggDemandEconomy = AggregateDemandEconomy(**init_ADEconomy)
InfHorizonTypeAgg_d.getEconomyData(AggDemandEconomy)
InfHorizonTypeAgg_h.getEconomyData(AggDemandEconomy)
InfHorizonTypeAgg_c.getEconomyData(AggDemandEconomy)
BaseTypeList = [InfHorizonTypeAgg_d, InfHorizonTypeAgg_h, InfHorizonTypeAgg_c ]
      
# Fill in the Markov income distribution for each base type
# NOTE: THIS ASSUMES NO LIFECYCLE
IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg_d.IncUnemp])])
IncomeDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonTypeAgg_d.IncUnempNoBenefits])])
    
for ThisType in BaseTypeList:
    EmployedIncomeDstn = deepcopy(ThisType.IncomeDstn[0])
    ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0]] + [IncomeDstn_unemp]*UBspell_normal + [IncomeDstn_unemp_nobenefits] 
    ThisType.IncomeDstn_base = ThisType.IncomeDstn
    
# Make the overall list of types
TypeList = []
n = 0
for e in range(num_types):
    for b in range(DiscFacCount):
        DiscFac = DiscFacDstns[e].X[b]
        AgentCount = int(np.floor(AgentCountTotal*data_EducShares[e]*DiscFacDstns[e].pmf[b]))
        ThisType = deepcopy(BaseTypeList[e])
        ThisType.AgentCount = AgentCount
        ThisType.DiscFac = DiscFac
        ThisType.seed = n
        TypeList.append(ThisType)
        n += 1
base_dict['Agents'] = TypeList    

AggDemandEconomy.agents = TypeList
AggDemandEconomy.solve()

AggDemandEconomy.reset()
for agent in AggDemandEconomy.agents:
    agent.initializeSim()
    agent.AggDemandFac = 1.0
    agent.RfreeNow = 1.0
    agent.CaggNow = 1.0

AggDemandEconomy.makeHistory()   
AggDemandEconomy.saveState()   
#AggDemandEconomy.switchToCounterfactualMode("base")
#AggDemandEconomy.makeIdiosyncraticShockHistories()

output_keys = ['NPV_AggIncome', 'NPV_AggCons', 'AggIncome', 'AggCons']


#%% Objective functions
# -----------------------------------------------------------------------------
def betasObjFunc(betas, spreads, target_option=1, print_mode=False, print_file=False, filename='DefaultResultsFile.txt'):
    '''
    Objective function for the estimation of discount factor distributions for the 
    three education groups. The groups can differ in the centering of their discount 
    factor distributions, and in the spread around the central value.
    
    Parameters
    ----------
    betas : [float]
        Central values of the discount factor distributions for each education
        level.
    spreads : [float]
        Half the width of each discount factor distribution. If we want the same spread
        for each education group we simply impose that the spreads are all the same.
        That is done outside this function. 
    target_option : integer
        = 1: Target medianLWPI and LorenzPtsAll 
        = 2: Target avgLWPI and LorenzPts_d, _h and _c
    print_mode : boolean, optional
        If true, statistics for each education level are printed. The default is False.
    print_file : boolean, optional
        If true, statistics are appended to the file filename. The default is False. 
    filename : str
        Filename for printing calculated statistics. The default is DefaultResultsFile.txt.
    
    Returns
    -------
    distance : float
        The distance of the estimation targets between those in the data and those
        produced by the model. 
    '''
    # # Set seed to ensure distance only changes due to different parameters 
    # random.seed(1234)

    beta_d, beta_h, beta_c = betas
    spread_d, spread_h, spread_c = spreads

    # # Overwrite the discount factor distribution for each education level with new values
    dfs_d = Uniform(beta_d-spread_d, beta_d+spread_d).approx(DiscFacCount)
    dfs_h = Uniform(beta_h-spread_h, beta_h+spread_h).approx(DiscFacCount)
    dfs_c = Uniform(beta_c-spread_c, beta_c+spread_c).approx(DiscFacCount)
    dfs = [dfs_d, dfs_h, dfs_c]

    # Check GIC for each type:
    for e in range(num_types):
        for thedf in range(DiscFacCount):
            if dfs[e].X[thedf] > GICmaxBetas[e]*GICfactor: 
                dfs[e].X[thedf] = GICmaxBetas[e]*GICfactor
            elif dfs[e].X[thedf] < minBeta:
                dfs[e].X[thedf] = minBeta

    # Make a new list of types with updated discount factors 
    TypeListNew = []
    n = 0
    for e in range(num_types):
        for b in range(DiscFacCount):
            AgentCount = int(np.floor(AgentCountTotal*data_EducShares[e]*dfs[e].pmf[b]))
            ThisType = deepcopy(BaseTypeList[e])
            ThisType.AgentCount = AgentCount
            ThisType.DiscFac = dfs[e].X[b]
            ThisType.seed = n
            TypeListNew.append(ThisType)
            n += 1
    base_dict['Agents'] = TypeListNew

    AggDemandEconomy.agents = TypeListNew
    AggDemandEconomy.solve()

    AggDemandEconomy.reset()
    for agent in AggDemandEconomy.agents:
        agent.initializeSim()
        agent.AggDemandFac = 1.0
        agent.RfreeNow = 1.0
        agent.CaggNow = 1.0

    AggDemandEconomy.makeHistory()   
    AggDemandEconomy.saveState()   

    # Simulate each type to get a new steady state solution 
    # solve: done in AggDemandEconomy.solve(), initializeSim: done in AggDemandEconomy.reset() 
    # baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()']
    baseline_commands = ['simulate()', 'saveState()']
    multiThreadCommandsFake(TypeListNew, baseline_commands)
    
    Stats = calcEstimStats(TypeListNew)
    
    if target_option == 1:
        sumSquares = 10*np.sum((Stats.medianLWPI-data_medianLWPI)**2)
        sumSquares += np.sum((np.array(Stats.LorenzPts) - data_LorenzPtsAll)**2)
    elif target_option == 2:
        lp_d = calcLorenzPts(TypeListNew[0:DiscFacCount])
        lp_h = calcLorenzPts(TypeListNew[DiscFacCount:2*DiscFacCount])
        lp_c = calcLorenzPts(TypeListNew[2*DiscFacCount:3*DiscFacCount])
        sumSquares = np.sum((np.array(Stats.avgLWPI)-data_avgLWPI)**2)
        sumSquares += np.sum((np.array(lp_d)-data_LorenzPts[0])**2)
        sumSquares += np.sum((np.array(lp_h)-data_LorenzPts[1])**2)
        sumSquares += np.sum((np.array(lp_c)-data_LorenzPts[2])**2)
    
    distance = np.sqrt(sumSquares)

    if print_mode or print_file:
        WealthShares = calcWealthShareByEd(TypeListNew)
        MPCs = calcMPCbyEd(TypeListNew)

    # If not estimating, print stats by education level
    if print_mode:
        print('Dropouts: beta = ', mystr(beta_d), ' spread = ', mystr(spread_d))
        print('Highschool: beta = ', mystr(beta_h), ' spread = ', mystr(spread_h))
        print('College: beta = ', mystr(beta_c), ' spread = ', mystr(spread_c))
        print('Median LW/PI-ratios: D = ' + mystr(Stats.medianLWPI[0][0]) + ' H = ' + mystr(Stats.medianLWPI[1][0]) \
              + ' C = ' + mystr(Stats.medianLWPI[2][0])) 
        print('Lorenz shares - all:')
        print(Stats.LorenzPts)
        if target_option == 2:
            print('Lorenz shares - Dropouts:')
            print(lp_d)
            print('Lorenz shares - Highschool:')
            print(lp_h)
            print('Lorenz shares - College:')
            print(lp_c) 
        
        print('Distance = ' + mystr(distance))
        print('Average LW/PI-ratios: D = ' + mystr(Stats.avgLWPI[0]) + ' H = ' + mystr(Stats.avgLWPI[1]) \
              + ' C = ' + mystr(Stats.avgLWPI[2])) 
        print('Total LW/Total PI: D = ' + mystr(Stats.LWoPI[0]) + ' H = ' + mystr(Stats.LWoPI[1]) \
              + ' C = ' + mystr(Stats.LWoPI[2]))
        print('Wealth Shares: D = ' + mystr(WealthShares[0]) + \
              ' H = ' + mystr(WealthShares[1]) + ' C = ' + mystr(WealthShares[2]))
        print('Average MPCs (incl. splurge) = ['+str(round(MPCs.MPCsA[0],3))+', '
                      +str(round(MPCs.MPCsA[1],3))+', '+str(round(MPCs.MPCsA[2],3))+', '
                      +str(round(MPCs.MPCsA[3],3))+']\n')

    if print_file:
        with open(filename, 'a') as resFile: 
            resFile.write('Population calculations:\n')
            resFile.write('\tMedian LW/PI-ratios = ['+mystr(Stats.medianLWPI[0][0])+', '+ 
                          mystr(Stats.medianLWPI[1][0])+', '+mystr(Stats.medianLWPI[2][0])+']\n')
            resFile.write('\tLorenz Points = ['+str(round(Stats.LorenzPts[0],4))+', '
                          +str(round(Stats.LorenzPts[1],4))+', '+str(round(Stats.LorenzPts[2],4))+', '
                          +str(round(Stats.LorenzPts[3],4))+']\n')
            resFile.write('\tWealth shares = ['+str(round(WealthShares[0],3))+', '
                          +str(round(WealthShares[1],3))+', '+str(round(WealthShares[2],3))+']\n')
            resFile.write('\tAverage MPCs (incl. splurge) = ['+str(round(MPCs.MPCsA[0],3))+', '
                          +str(round(MPCs.MPCsA[1],3))+', '+str(round(MPCs.MPCsA[2],3))+', '
                          +str(round(MPCs.MPCsA[3],3))+']\n')
        
    return distance 
# -----------------------------------------------------------------------------
def betasObjFuncEduc(beta, spread, educ_type=2, print_mode=False, print_file=False, filename='DefaultResultsFile.txt'):
    '''
    Objective function for the estimation of a discount factor distribution for
    a single education group.
    
    Parameters
    ----------
    beta : float
        Central value of the discount factor distribution.
    spread : float
        Half the width of the discount factor distribution.
    educ_type : integer
        The education type to estimate a discount factor distribution for.     
        Targets are avgLWPI[educ_type] and LorenzPts[educ_type]
    print_mode : boolean, optional
        If true, statistics are printed. The default is False.
    print_file : boolean, optional
        If true, statistics are appended to the file filename. The default is False. 
    filename : str
        Filename for printing calculated statistics. The default is DefaultResultsFile.txt.
    
    Returns
    -------
    distance : float
        The distance of the estimation targets between those in the data and those
        produced by the model. 
    '''
    # # Set seed to ensure distance only changes due to different parameters 
    # random.seed(1234)

    dfs = Uniform(beta-spread, beta+spread).approx(DiscFacCount)
    
    # Check GIC:
    for thedf in range(DiscFacCount):
        if dfs.X[thedf] > GICmaxBetas[educ_type]*GICfactor:
            dfs.X[thedf] = GICmaxBetas[educ_type]*GICfactor
        elif dfs.X[thedf] < minBeta:
            dfs.X[thedf] = minBeta

    # Make a new list of types with updated discount factors for the given educ type
    TypeListNewEduc = []
    n = 0
    for b in range(DiscFacCount):
        AgentCount = int(np.floor(AgentCountTotal*data_EducShares[educ_type]*dfs.pmf[b]))
        ThisType = deepcopy(BaseTypeList[educ_type])
        ThisType.AgentCount = AgentCount
        ThisType.DiscFac = dfs.X[b]
        ThisType.seed = n
        TypeListNewEduc.append(ThisType)
        n += 1
    TypeListAll = AggDemandEconomy.agents
    TypeListAll[educ_type*DiscFacCount:(educ_type+1)*DiscFacCount] = TypeListNewEduc
            
    base_dict['Agents'] = TypeListAll
    AggDemandEconomy.agents = TypeListAll
    AggDemandEconomy.solve()

    AggDemandEconomy.reset()
    for agent in AggDemandEconomy.agents:
        agent.initializeSim()
        agent.AggDemandFac = 1.0
        agent.RfreeNow = 1.0
        agent.CaggNow = 1.0

    AggDemandEconomy.makeHistory()   
    AggDemandEconomy.saveState()   

    # Simulate each type to get a new steady state solution 
    # solve: done in AggDemandEconomy.solve(), initializeSim: done in AggDemandEconomy.reset() 
    # baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()']
    baseline_commands = ['simulate()', 'saveState()']
    multiThreadCommandsFake(TypeListAll, baseline_commands)
    
    Stats = calcEstimStats(TypeListAll)
    
    sumSquares = np.sum((Stats.medianLWPI[educ_type]-data_medianLWPI[educ_type])**2)
    lp = calcLorenzPts(TypeListNewEduc)
    sumSquares += np.sum((np.array(lp) - data_LorenzPts[educ_type])**2)
#    sumSquares = np.sum((Stats.avgLWPI[educ_type]-data_avgLWPI[educ_type])**2)
   
    distance = np.sqrt(sumSquares)

    # If not estimating, print stats by education level
    if print_mode:
        print('Median LW/PI-ratio for group e = ' + mystr(educ_type) + ' is: ' \
              + mystr(Stats.medianLWPI[educ_type][0]))
        if educ_type == 0:
            print('Lorenz shares - Dropouts:')
        elif educ_type == 1:
            print('Lorenz shares - Highschool:')
        elif educ_type == 2:
            print('Lorenz shares - College:')
        print(lp)
        print('Distance = ' + mystr(distance))
        print('Non-targeted moments:')
        print('Average LW/PI-ratios for group e = ' + mystr(educ_type) + ' is: ' \
              + mystr(Stats.avgLWPI[educ_type]))
        print('Lorenz shares - all:')
        print(Stats.LorenzPts)
    
    if print_file:
        with open(filename, 'a') as resFile: 
            resFile.write('Education group = '+mystr(educ_type)+': beta = '+mystr4(beta)+
                          ', nabla = '+mystr4(spread)+'\n')
            resFile.write('\tMedian LW/PI-ratio = '+mystr(Stats.medianLWPI[educ_type][0])+'\n')
            resFile.write('\tLorenz Points = ['+str(round(lp[0],4))+', '+str(round(lp[1],4))+', '
                          +str(round(lp[2],4))+', '+str(round(lp[3],4))+']\n')
        
    return distance 
# -----------------------------------------------------------------------------
#%% Estimate discount factor distributions separately for each education type

if IncUnemp == 0.7 and IncUnempNoBenefits == 0.5:
    # Baseline unemployment system: 
    print('Estimating for CRRA = '+str(round(CRRA,1))+' and R = ' + str(round(Rfree_base[0],3))+':\n')
    df_resFileStr = res_dir+'/DiscFacEstim_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'.txt'
else:
    print('Estimating for an alternativ unemployment system with IncUnemp = '+str(round(IncUnemp,2))+
          ' and IncUnempNoBenefits = ' + str(round(IncUnempNoBenefits,2))+':\n')
    df_resFileStr = res_dir+'/DiscFacEstim_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'_altBenefits.txt'
    
for edType in [0,1,2]:
    f_temp = lambda x : betasObjFuncEduc(x[0],x[1], educ_type=edType)
    if edType == 0:
        initValues = [0.69, 0.54]       # Dropouts
    elif edType == 1:
        initValues = [0.90, 0.1]      # HighSchool
    elif edType == 2:
        initValues = [0.97, 0.015]     # College
    else:
        initValues = [0.90,0.02]
        
    opt_params = minimizeNelderMead(f_temp, initValues, verbose=True)
    print('Finished estimating for education type = '+str(edType)+'. Optimal beta and spread are:')
    print('Beta = ' + mystr4(opt_params[0]) +'  Nabla = ' + mystr4(opt_params[1]))

    if edType == 0:
        mode = 'w'      # Overwrite old file...
    else:
        mode = 'a'      # ...but append all results in same file 
    with open(df_resFileStr, mode) as f: 
        outStr = repr({'EducationGroup' : edType, 'beta' : opt_params[0], 'nabla' : opt_params[1]})
        f.write(outStr+'\n')
        f.close()

#%% Read in estimates and calculate all results:
if IncUnemp == 0.7 and IncUnempNoBenefits == 0.5:
    # Baseline unemployment system: 
    print('Calculating all results for CRRA = '+str(round(CRRA,1))+' and R = ' + str(round(Rfree_base[0],3))+':\n')
    df_resFileStr = res_dir+'/DiscFacEstim_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'.txt'
    ar_resFileStr = res_dir+'/AllResults_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'.txt'
else:
    print('Calculating all results for an alternativ unemployment system with IncUnemp = '+str(round(IncUnemp,2))+
          ' and IncUnempNoBenefits = ' + str(round(IncUnempNoBenefits,2))+':\n')
    df_resFileStr = res_dir+'/DiscFacEstim_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'_altBenefits.txt'
    ar_resFileStr = res_dir+'/AllResults_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'_altBenefits.txt'

with open(ar_resFileStr, 'w') as resFile: 
    resFile.write('Results for CRRA = '+str(CRRA)+' and R = '+str(round(Rfree_base[0],3))+'\n\n')
    
# Calculate results by education group    
myEstim = [[],[],[]]
betFile = open(df_resFileStr, 'r')
readStr = betFile.readline().strip()
while readStr != '' :
    dictload = eval(readStr)
    edType = dictload['EducationGroup']
    beta = dictload['beta']
    nabla = dictload['nabla']
    myEstim[edType] = [beta,nabla]
    betasObjFuncEduc(beta, nabla, educ_type = edType, print_mode=True, print_file=True, filename=ar_resFileStr)
    checkDiscFacDistribution(beta, nabla, edType, print_mode=True, print_file=True, filename=ar_resFileStr)
    readStr = betFile.readline().strip()
betFile.close()

# Also calculate results for the whole population 
betasObjFunc([myEstim[0][0], myEstim[1][0], myEstim[2][0]], \
             [myEstim[0][1], myEstim[1][1], myEstim[2][1]], \
             target_option = 1, print_mode=True, print_file=True, filename=ar_resFileStr)



#%% 
run_additional_analysis = False
if run_additional_analysis:
    #%% Read in estimates and save resulting discount factor distributions:
    myEstim = [[],[],[]]
    if IncUnemp == 0.7 and IncUnempNoBenefits == 0.5:
        # Baseline unemployment system: 
        betFile = open(res_dir+'/DiscFacEstim_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'.txt', 'r')
    else:
        betFile = open(res_dir+'/DiscFacEstim_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'_altBenefits.txt', 'r')
    readStr = betFile.readline().strip()
    while readStr != '' :
        dictload = eval(readStr)
        edType = dictload['EducationGroup']
        beta = dictload['beta']
        nabla = dictload['nabla']
        myEstim[edType] = [beta,nabla]
        readStr = betFile.readline().strip()
    betFile.close()

    if IncUnemp == 0.7 and IncUnempNoBenefits == 0.5:
        # Baseline unemployment system: 
        outFileStr = res_dir+'/DiscFacDistributions_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'.txt'
    else:
        outFileStr = res_dir+'/DiscFacDistributions_CRRA_'+str(CRRA)+'_R_'+str(Rfree_base[0])+'_altBenefits.txt'
    outFile = open(outFileStr, 'w')
    
    for e in [0,1,2]:
        dfs = Uniform(myEstim[e][0]-myEstim[e][1], myEstim[e][0]+myEstim[e][1]).approx(DiscFacCount)
        
        # Check GIC:
        for thedf in range(DiscFacCount):
            if dfs.X[thedf] > GICmaxBetas[e]*GICfactor:
                dfs.X[thedf] = GICmaxBetas[e]*GICfactor
            elif dfs.X[thedf] < minBeta:
                dfs.X[thedf] = minBeta
        theDFs = np.round(dfs.X,4)
        outStr = repr({'EducationGroup' : e, 'betaDistr' : theDFs.tolist()})
        outFile.write(outStr+'\n')
    outFile.close()
    

    #%% Plot of MPCs
    mpcs = calcMPCbyEd(AggDemandEconomy.agents)
    
    plt.plot(range(len(mpcs[0])), np.sort(mpcs[0]))
    plt.xlabel('Agents')
    plt.ylabel('MPCs')
    plt.title('Dropout')
    plt.show()
    
    plt.plot(range(len(mpcs[1])), np.sort(mpcs[1]))
    plt.xlabel('Agents')
    plt.ylabel('MPCs')
    plt.title('Highschool')
    plt.show()
    
    plt.plot(range(len(mpcs[2])), np.sort(mpcs[2]))
    plt.xlabel('Agents')
    plt.ylabel('MPCs')
    plt.title('College')
    plt.show()


      








