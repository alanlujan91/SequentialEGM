'''
This file has major functions that are used by GiveItAwayMAIN.py
'''
import warnings
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from HARK import multiThreadCommands, multiThreadCommandsFake

mystr = lambda x : '{:.2f}'.format(x)
mystr2 = lambda x : '{:.3f}'.format(x)

def runExperiment(agents,RecessionShock = False,TaxCutShock = False, \
                  ExtendedUIShock =False, UpdatePrb = 1.0, Splurge = 0.0):
    '''
    Conduct an experiment in which the recession hits and/or fiscal policy is initiated.
    
    Parameters
    ----------
    agents : [AgentType]
        List of agent types in the economy.
    RecessionShock : bool
        Indicator for whether the recession actually hits.
        
    Returns
    -------
    TBD
    '''
    T = agents[0].T_sim
    
    # Make dictionaries of parameters to give to the agents
    experiment_dict = {
            'use_prestate' : True,
            'RecessionShock' : RecessionShock,
            'TaxCutShock' : TaxCutShock,
            'ExtendedUIShock' : ExtendedUIShock,
            'UpdatePrb' : UpdatePrb
            }
      
    # Begin the experiment by resetting each type's state to the baseline values
    PopCount = 0
    for ThisType in agents:
        ThisType.read_shocks = True
        ThisType(**experiment_dict)
        PopCount += ThisType.AgentCount
        
    # Update the perceived and actual Markov arrays, solve and re-draw shocks if
    # warranted, then impose the recession shock, and finally
    # simulate the model for three years.
    experiment_commands = ['updateMrkvArray()', 'solveIfChanged()',
                           'makeShocksIfChanged()', 'initializeSim()',
                           'hitWithRecessionShock()',
                           'simulate()']
    multiThreadCommandsFake(agents, experiment_commands)
    
    #print(agents[0].history['pLvlNow'][35:40,0])
    #print(agents[0].history['TranShkNow'][35:40,0])
    
    # Extract simulated consumption, labor income, and weight data
    cNrm_all = np.concatenate([ThisType.history['cNrmNow'] for ThisType in agents], axis=1)
    Mrkv_hist = np.concatenate([ThisType.history['MrkvNow'] for ThisType in agents], axis=1)
    pLvl_all = np.concatenate([ThisType.history['pLvlNow'] for ThisType in agents], axis=1)
    TranShk_all = np.concatenate([ThisType.history['TranShkNow'] for ThisType in agents], axis=1)
    mNrm_all = np.concatenate([ThisType.history['mNrmNow'] for ThisType in agents], axis=1)
    aNrm_all = np.concatenate([ThisType.history['aNrmNow'] for ThisType in agents], axis=1)
    cLvl_all = cNrm_all*pLvl_all
    # Calculate Splurge results (agents splurge on some of their income, and follow model for the rest)
    cLvl_all_splurge = (1.0-Splurge)*cLvl_all + Splurge*pLvl_all*TranShk_all
    
    IndIncome = pLvl_all*TranShk_all
    AggIncome = np.sum(IndIncome,1)
    AggCons   = np.sum(cLvl_all_splurge,1)
    
    #print(AggIncome[35:40])
   
    
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
    NPV_AggIncome = calculate_NPV(AggIncome,T,ThisType.Rfree[0])
    NPV_AggCons   = calculate_NPV(AggCons,T,ThisType.Rfree[0])
    
    
    # Get initial Markov states
    Mrkv_init = np.concatenate([ThisType.history['MrkvNow'][0,:] for ThisType in agents])
    return_dict = {'cNrm_all' : cNrm_all,
                   'TranShk_all' : TranShk_all,
                   'cLvl_all' : cLvl_all,
                   'pLvl_all' : pLvl_all,
                   'Mrkv_hist' : Mrkv_hist,
                   'Mrkv_init' : Mrkv_init,
                   'mNrm_all' : mNrm_all,
                   'aNrm_all' : aNrm_all,
                   'cLvl_all_splurge' : cLvl_all_splurge,
                   'NPV_AggIncome': NPV_AggIncome,
                   'NPV_AggCons': NPV_AggCons,
                   'AggIncome': AggIncome,
                   'AggCons': AggCons}
    return return_dict

