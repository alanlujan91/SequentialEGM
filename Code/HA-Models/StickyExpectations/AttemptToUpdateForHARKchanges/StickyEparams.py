'''
This module holds calibrated parameter dictionaries for a small open economy
sticky expectations model with a tax cut experiment embedded in the Markov 
set up
'''
from __future__ import division
from builtins import range
#from past.utils import old_div
import os
import numpy as np
from copy import copy
from HARK.distribution import Lognormal, MeanOneLogNormal, Uniform, combineIndepDstns, DiscreteDistribution

# Choose directory paths relative to the StickyE files
# See: https://stackoverflow.com/questions/918154/relative-paths-in-python
my_file_path = os.path.dirname(os.path.abspath(__file__))

calibration_dir = os.path.join(my_file_path, "./Calibration/") # Absolute directory for primitive parameter files
tables_dir = os.path.join(my_file_path, "./Tables/")       # Absolute directory for saving tex tables
results_dir = os.path.join(my_file_path, "./Results/")         # Absolute directory for saving output files
figures_dir = os.path.join(my_file_path, "./Figures/")     # Absolute directory for saving figures

def importParam(param_name):
    return float(np.max(np.genfromtxt(calibration_dir + param_name + '.txt')))

# Import primitive parameters from calibrations folder
CRRA = importParam('CRRA')                   # Coefficient of relative risk aversion
DeprFacAnn = importParam('DeprFacAnn')       # Annual depreciation factor
CapShare = importParam('CapShare')           # Capital's share in production function
KYratioSS = importParam('KYratioSS')         # Steady state capital to output ratio (PF-DSGE)
UpdatePrb = importParam('UpdatePrb')         # Probability that each agent observes the aggregate productivity state each period (in sticky version)
UnempPrb = importParam('UnempPrb')           # Unemployment probability
DiePrb = importParam('DiePrb')               # Quarterly mortality probability
TranShkVarAnn = importParam('TranShkVarAnn') # Annual variance of idiosyncratic transitory shocks
PermShkVarAnn = importParam('PermShkVarAnn') # Annual variance of idiosyncratic permanent shocks
TranShkAggVar = importParam('TranShkAggVar') # Variance of aggregate transitory shocks
PermShkAggVar = importParam('PermShkAggVar') # Variance of aggregate permanent shocks

TranShkAggVar = 0.000000000001 # Variance of aggregate transitory shocks
PermShkAggVar = 0.000000000001 # Variance of aggregate permanent shocks

DiscFacSOE = importParam('betaSOE')          # Discount factor, SOE model

# Calculate parameters based on the primitive parameters
DeprFac = 1. - DeprFacAnn**0.25                  # Quarterly depreciation rate
KSS = KtYratioSS = KYratioSS**(1./(1.-CapShare)) # Steady state Capital to labor productivity
wRteSS = (1.-CapShare)*KSS**CapShare             # Steady state wage rate
rFreeSS = CapShare*KSS**(CapShare-1.)            # Steady state interest rate
RfreeSS = 1. - DeprFac + rFreeSS                 # Steady state return factor
LivPrb = 1. - DiePrb                             # Quarterly survival probability
DiscFacDSGE = RfreeSS**(-1)                      # Discount factor, HA-DSGE and RA models
TranShkVar = TranShkVarAnn*4.                    # Variance of idiosyncratic transitory shocks
#PermShkVar = old_div(PermShkVarAnn,4.)           # Variance of idiosyncratic permanent shocks
PermShkVar = PermShkVarAnn/4.0

## Choose basic simulation parameters
#periods_to_sim = 21010 # Total number of periods to simulate; this might be increased by DSGEmarkov model
#ignore_periods = 1000  # Number of simulated periods to ignore (in order to ensure we are near steady state)
#interval_size = 200    # Number of periods in each subsample interval
#AgentCount = 20000     # Total number of agents to simulate in the economy
max_t_between_updates = None # Maximum number of periods an agent will go between updating (can be None)

TypeCount = 11
DiscFacMeanSOE = 0.93286
DiscFacSpread = 0.0641
IncUnemp = 0.3

periods_to_sim = 1200
AgentCount = 5000
StateCount = 4
TypeCount = 2
income_increase = 1.02
expected_tax_cut_duration = 8 #expected to last 2 years

prob_end = 1.0/expected_tax_cut_duration
MrkvArray = np.array([ [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0-prob_end, prob_end], [0.0, 0.0, 1.0-prob_end, prob_end], [1.0, 0.0, 0.0, 0.0]])
PermGroFacSet = np.array([1.0, income_increase, 1.0, 1.0/income_increase])

DiscFacSetSOE  = Uniform(bot=DiscFacMeanSOE-DiscFacSpread,
                                top=DiscFacMeanSOE+DiscFacSpread
                                ).approx(N=TypeCount).X

###############################################################################

# Define parameters for the small open economy version of the model
init_SOE_consumer = { 'CRRA': CRRA,
                      'DiscFac': DiscFacMeanSOE,
                      'LivPrb': [LivPrb],
                      'PermGroFac': [1.0],
                      'AgentCount': AgentCount // TypeCount, # Spread agents evenly among types
                      'aXtraMin': 0.00001,
                      'aXtraMax': 40.0,
                      'aXtraNestFac': 3,
                      'aXtraCount': 48,
                      'aXtraExtra': [None],
                      'PermShkStd': [np.sqrt(PermShkVar)],
                      'PermShkCount': 7,
                      'TranShkStd': [np.sqrt(TranShkVar)],
                      'TranShkCount': 7,
                      'UnempPrb': UnempPrb,
                      'UnempPrbRet': 0.0,
                      'IncUnemp': IncUnemp,
                      'IncUnempRet': 0.0,
                      'BoroCnstArt':0.0,
                      'tax_rate':0.0,
                      'T_retire':0,
                      'MgridBase': np.array([0.5,1.5]),
                      'aNrmInitMean' : np.log(0.00001),
                      'aNrmInitStd' : 0.0,
                      'pLvlInitMean' : 0.0,
                      'pLvlInitStd' : 0.0,
                      'UpdatePrb' : UpdatePrb,
                      'T_age' : None,
                      'T_cycle' : 1,
                      'cycles' : 0,
                      'T_sim' : periods_to_sim,
                       'max_t_between_updates' : max_t_between_updates
                    }

# Define market parameters for the small open economy
init_SOE_market = {  'PermShkAggCount': 5,
                     'TranShkAggCount': 5,
                     'PermShkAggStd': np.sqrt(PermShkAggVar),
                     'TranShkAggStd': np.sqrt(TranShkAggVar),
                     'PermGroFacAgg': 1.0,
                     'DeprFac': DeprFac,
                     'CapShare': CapShare,
                     'Rfree': RfreeSS,
                     'wRte': wRteSS,
                     'act_T': periods_to_sim,
                     }

###############################################################################

# Define parameters for the small open Markov economy version of the model
init_SOE_mrkv_consumer = copy(init_SOE_consumer)
init_SOE_mrkv_consumer['MrkvArray'] = MrkvArray

# Define market parameters for the small open Markov economy
init_SOE_mrkv_market = copy(init_SOE_market)
init_SOE_mrkv_market['MrkvArray'] = MrkvArray
init_SOE_mrkv_market['PermShkAggStd'] = StateCount*[init_SOE_market['PermShkAggStd']]
init_SOE_mrkv_market['TranShkAggStd'] = StateCount*[init_SOE_market['TranShkAggStd']]
init_SOE_mrkv_market['PermGroFacAgg'] = PermGroFacSet
init_SOE_mrkv_market['MrkvNow_init'] = 0 # Start at 0 and stay there unless forced out by MIT shock
init_SOE_mrkv_market['loops_max'] = 1

