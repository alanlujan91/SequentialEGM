import os
import numpy as np
from cstwMPC_Adjusted import cstwMPCagent 
# in cstwMPC I made changes to cstwMPC.py in the HARK library to adjust for the fact,
# that TransShkDstn is an object with .X and .pmf rather than an array


# Set basic parameters for the lifecycle micro model
Rfree = 1.04**(0.25)          # Quarterly interest factor
working_T = 41*4              # Number of working periods
retired_T = 55*4              # Number of retired periods
T_cycle = working_T+retired_T # Total number of periods
CRRA = 1.0                    # Coefficient of relative risk aversion
UnempPrb = 0.07               # Probability of unemployment while working
UnempPrbRet = 0.0005          # Probabulity of "unemployment" while retired
IncUnemp = 0.15               # Unemployment benefit replacement rate
IncUnempRet = 0.0             # Ditto when retired
BoroCnstArt = 0.0             # Artificial borrowing constraint

# Set grid sizes
PermShkCount = 5              # Number of points in permanent income shock grid
TranShkCount = 5              # Number of points in transitory income shock grid
aXtraMin = 0.00001            # Minimum end-of-period assets in grid
aXtraMax = 20                 # Maximum end-of-period assets in grid
aXtraCount = 20               # Number of points in assets grid
aXtraNestFac = 3              # Number of times to 'exponentially nest' when constructing assets grid
CubicBool = False             # Whether to use cubic spline interpolation
vFuncBool = False             # Whether to calculate the value function during solution

# Set simulation parameters
T_sim_PY = 1200               # Number of periods to simulate (idiosyncratic shocks model, perpetual youth)
T_age = T_cycle + 1           # Don't let simulated agents survive beyond this age
pLvlInitStd = 0.4             # Standard deviation of initial permanent income
aNrmInitMean = np.log(0.5)    # log initial wealth/income mean
aNrmInitStd  = 0.5            # log initial wealth/income standard deviation

# Set indiividual parameters for the infinite horizon model
IndL = 10.0/9.0               # Labor supply per individual (constant)
PermGroFac_i = [1.000**0.25]  # Permanent income growth factor (no perm growth)
DiscFac_i = 0.97              # Default intertemporal discount factor
LivPrb_i = [1.0 - 1.0/160.0]  # Survival probability
PermShkStd_i = [(0.01*4/11)**0.5] # Standard deviation of permanent shocks to income
TranShkStd_i = [(0.01*4)**0.5]    # Standard deviation of transitory shocks to income

init_infinite = {"CRRA":CRRA,
                "Rfree":1.01/LivPrb_i[0],
                "PermGroFac":PermGroFac_i,
                "PermGroFacAgg":1.0,
                "BoroCnstArt":BoroCnstArt,
                "CubicBool":CubicBool,
                "vFuncBool":vFuncBool,
                "PermShkStd":PermShkStd_i,
                "PermShkCount":PermShkCount,
                "TranShkStd":TranShkStd_i,
                "TranShkCount":TranShkCount,
                "UnempPrb":UnempPrb,
                "IncUnemp":IncUnemp,
                "UnempPrbRet":None,
                "IncUnempRet":None,
                "aXtraMin":aXtraMin,
                "aXtraMax":aXtraMax,
                "aXtraCount":aXtraCount,
                "aXtraExtra":[None],
                "aXtraNestFac":aXtraNestFac,
                "LivPrb":LivPrb_i,
                "DiscFac":DiscFac_i, # dummy value, will be overwritten
                "cycles":0,
                "T_cycle":1,
                "T_retire":0,
                'T_sim':T_sim_PY,
                'T_age': 400,
                'IndL': IndL,
                'aNrmInitMean':np.log(0.00001),
                'aNrmInitStd':0.0,
                'pLvlInitMean':0.0,
                'pLvlInitStd':0.0,
                'AgentCount':0, # will be overwritten by parameter distributor
                }



# Make AgentTypes for estimation
PerpetualYouthType = cstwMPCagent(**init_infinite)
PerpetualYouthType.solve()

