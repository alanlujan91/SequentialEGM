# Import python tools
import sys
import os
import numpy as np
from copy import deepcopy

# Import needed tools from HARK
from HARK.ConsumptionSaving.ConsIndShockModel import *
from HARK.cstwMPC.SetupParamsCSTW import init_infinite

# Set standard HARK parameter values
base_params = deepcopy(init_infinite)
base_params['LivPrb']       = [0.995]                       #from stickyE paper
base_params['Rfree']        = 1.015/base_params['LivPrb'][0]#from stickyE paper
base_params['PermShkStd']   = [0.003**0.5]                  #from stickyE paper
base_params['TranShkStd']   = [0.120**0.5]                  #from stickyE paper
base_params['T_age']        = 400           # Kill off agents if they manage to achieve T_kill working years
base_params['AgentCount']   = 1             # Number of agents per instance of IndShockConsType
base_params['pLvlInitMean'] = np.log(23.72) # From Table 1, in thousands of USD (Q-I: where is this from?)

# T_sim needs to be long enough to reach the ergodic distribution
base_params['T_sim'] = 1000

#%% Test 1

CheckType = IndShockConsumerType(**base_params)
CheckType(DiscFac = 0.96)   # Check only for center Disc Fac
CheckType.track_vars = ['aNrmNow','mNrmNow','cNrmNow','pLvlNow','PermShkNow','TranShkNow']
CheckType.solve()
CheckType.initializeSim()
CheckType.simulate()
CheckType.unpackcFunc()

# Simulating manually from start onwards
c = np.zeros((CheckType.T_sim,CheckType.AgentCount))
a = np.zeros((CheckType.T_sim,CheckType.AgentCount))
a[0,:] = CheckType.aNrmNow_hist[0,:]

AlwaysStartFromCorrect_a = True

for period in range(1,CheckType.T_sim):
    if AlwaysStartFromCorrect_a:
        if CheckType.TranShkNow_hist[period] != 1:
            m_adj = CheckType.aNrmNow_hist[period-1,:]*base_params['Rfree']/CheckType.PermShkNow_hist[period] + CheckType.TranShkNow_hist[period]
        else:
            m_adj = np.exp(base_params['aNrmInitMean'])*base_params['Rfree']/CheckType.PermShkNow_hist[period] + CheckType.TranShkNow_hist[period]
    else:
        m_adj = a[period-1,:]*base_params['Rfree']/CheckType.PermShkNow_hist[period] + CheckType.TranShkNow_hist[period]
        
    c[period,:] = CheckType.cFunc[0](m_adj)
    a[period,:] = m_adj - c[period,:]

# I am only checking for the first agent
Diff_in_c = abs(c[:,0]-CheckType.cNrmNow_hist[:,0])
# This difference should be zero
y= np.where(Diff_in_c>1e-10)
# For these periods it's not zero
print('C_Diff is not zero for indices: ', y,'\n')    
print('TransShokNow is exactly 1 for indices', np.where(CheckType.TranShkNow_hist==1),'\n')
    



    


   