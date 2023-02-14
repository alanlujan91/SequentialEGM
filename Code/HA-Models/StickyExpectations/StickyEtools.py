"""
This module holds some data tools used in the cAndCwithStickyE project.
"""
from __future__ import division
from __future__ import absolute_import

from builtins import str
from builtins import range
#from past.utils import old_div

import os
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.sandbox.regression.gmm as smsrg
import matplotlib.pyplot as plt
from copy import deepcopy
import subprocess
from HARK.utilities import CRRAutility
from HARK.interpolation import LinearInterp
from HARK.distribution import Uniform
from StickyEparams import results_dir, tables_dir, figures_dir, UpdatePrb, PermShkAggVar
UpdatePrbBase = UpdatePrb
PermShkAggVarBase = PermShkAggVar
   
    
def runParkerExperiment(BaseEconomy,BonusSize,T_before_bonus,T_after_bonus,verbose):
    '''
    Run an experiment in which a universal tax refund ("bonus") is announced T periods ahead
    of when households will actually get their checks.  However, each household only "notices"
    this upcoming bonus check when they update to the latest macroeconomic news (or when it
    actually arrives in their bank account).  Households are able to borrow against the bonus,
    so they perceive it as a (discounted) transitory shock when they notice it.
    
    Results are saved to text files in the Results directory.
    
    Parameters
    ----------
    BaseEconomy : StickySmallOpenMarkovEconomy
        Baseline economy (used in the "main" part of the work) to be replicated for the experiment.
    BonusSize : float
        Size of the "bonus" check relative to average quarterly earnings.
    T_before_bonus : int
        Number of periods in advance the "bonus" checks are announced.
    T_after_bonus : int
        Number of periods after the arrival of the "bonus" to simulate and report.
    verbose : bool
        Whether to print the results to screen.
        
    Returns
    -------
    None
    '''
    # Calculate the absolute level of the bonus based on its relative size and aggregate productivity
    pLvlBase = BaseEconomy.agents[0].PlvlAggNow # Average permanent income level
    BonusLvl = BonusSize*BaseEconomy.wRte*pLvlBase # Bonus check of 5% of average quarterly income
    T_sim_parker = T_before_bonus + T_after_bonus + 1
    
    # Make four copies of the economy: frictionless vs sticky, bonus vs none
    StickyNoneEconomy = deepcopy(BaseEconomy)
    StickyBonusEconomy = deepcopy(BaseEconomy)
    FrictionlessNoneEconomy = deepcopy(BaseEconomy)
    FrictionlessBonusEconomy = deepcopy(BaseEconomy)
    for agent in StickyNoneEconomy.agents:
        agent(UpdatePrb = UpdatePrb)
        agent(parker_experiment = False, BonusLvl = 0.0, t_until_bonus = 0)
    for agent in StickyBonusEconomy.agents:
        agent(UpdatePrb = UpdatePrb)
        agent(parker_experiment = True, BonusLvl = BonusLvl, t_until_bonus = T_before_bonus)
    for agent in FrictionlessNoneEconomy.agents:
        agent(UpdatePrb = 1.0)
        agent(parker_experiment = False, BonusLvl = 0.0, t_until_bonus = 0)
    for agent in FrictionlessBonusEconomy.agents:
        agent(UpdatePrb = 1.0)
        agent(parker_experiment = True, BonusLvl = BonusLvl, t_until_bonus = T_before_bonus)
        
    # Run the "Parker experiment" for the four economies and collect mean consumption change data
    cLvl_StickyNone = StickyNoneEconomy.runParkerExperiment(T_sim_parker)
    cLvl_StickyBonus = StickyBonusEconomy.runParkerExperiment(T_sim_parker)
    cLvl_FrictionlessNone = FrictionlessNoneEconomy.runParkerExperiment(T_sim_parker)
    cLvl_FrictionlessBonus = FrictionlessBonusEconomy.runParkerExperiment(T_sim_parker)
    
#    # Format and save the results of the "Parker experiment"
#    policy_text = 'B' + str(T_before_bonus) + 'A' + str(T_after_bonus) + 'S' + str(int(100*BonusSize))
#    parker_results_sticky = makeParkerExperimentText(BonusLvl,T_before_bonus,cLvl_StickyNone,cLvl_StickyBonus,True,'ParkerResults' + policy_text + 'S')
#    parker_results_frictionless = makeParkerExperimentText(BonusLvl,T_before_bonus,cLvl_FrictionlessNone,cLvl_FrictionlessBonus,False,'ParkerResults' + policy_text + 'F')
#    if verbose:
#        print(parker_results_sticky + '\n')
#        print(parker_results_frictionless + '\n')
        
    return cLvl_StickyNone, cLvl_StickyBonus, cLvl_FrictionlessNone, cLvl_FrictionlessBonus
    

def makeParkerExperimentText(BonusLvl,T_ahead,cLvlNone_hist,cLvlBonus_hist,sticky_bool,out_filename=None):
    '''
    Makes a string of text to describe the results of the "Parker experiment", in which
    a "bonus" payment is announced T_ahead periods in advance, but only households who
    update to the latest macroeconomic news notice this.  Advance knowledge of the bonus
    allows households to borrow against it.
    
    Parameters
    ----------
    BonusLvl : float
        Absolute level of the bonus check that households will receive.
    T_ahead : int
        Number of periods ahead of bonus arrival that the policy is announced.
    cLvlNone_hist : np.array
        Array with average consumption levels in a world with no bonus announcement.
    cLvlBonus_hist : np.array
        Array with average consumption levels in a world with the bonus policy.
    sticky_bool : bool
        Whether results are for sticky (True) or frictionless agents.
    out_filename : str
        Name of txt file in which to save the output, if not None.
        
    Returns
    -------
    output : str
        String describing the period by periods results of the Parker experiment.
    '''
    before_text_P = ' periods before the bonus arrives, '
    before_text_S = ' period before the bonus arrives, '
    after_text_P = ' periods after the bonus arrives, '
    after_text_S = ' period after the bonus arrives, '
    arrival_text =  'In the period that the bonus arrives, '
    con_text1 = 'households consume '
    con_text2 = ' of the bonus amount.'
    Number_text = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten']
    number_text = ['zero','one','two','three','four','five','six','seven','eight','nine','ten']
    if sticky_bool:
        agent_text = 'sticky households '
    else:
        agent_text = 'frictionless households '
    if T_ahead !=1:
        period_text = ' periods'
    else:
        period_text = ' period'
    
    output = 'The bonus check is announced to ' + agent_text + number_text[T_ahead] + period_text + ' in advance.\n'
    T_total = cLvlNone_hist.size
    
    for t in range(-T_ahead,T_total-T_ahead):
        s = t+T_ahead
        abs_t = np.abs(t)
        if abs_t != 0:
            text_now = Number_text[abs_t]
            if t > 0:
                if abs_t > 1:
                    temp = after_text_P
                else:
                    temp = after_text_S
            else:
                if abs_t > 1:
                    temp = before_text_P
                else:
                    temp = before_text_S
            text_now += temp
        else:
            text_now = arrival_text
        text_now += con_text1
        text_now += '{:.1%}'.format((cLvlBonus_hist[s] - cLvlNone_hist[s])/BonusLvl)
        text_now += con_text2
        text_now += '\n'
        output += text_now
        
    if out_filename is not None:
        with open(results_dir + out_filename + '.txt','w') as f:
            f.write(output)
            f.close()
    
    return output


    
def runTaxCutExperiment(BaseEconomy,T_after,num_agg_sims = 20):
    '''
    Run an experiment in which a tax cut increases labor income by a fixed percentage
    (which needs to be set up in a four state Markov process) for a stochastic 
    period. Each household only notices that the tax cut has been introduced 
    when they update, similarly they only notice it has gone when they update.
    
    Results are saved to text files in the Results directory.
    
    Parameters
    ----------
    BaseEconomy : StickySmallOpenMarkovEconomy
        Baseline economy (used in the "main" part of the work) to be replicated for the experiment.
    T_after : int
        Number of periods after the start of the tax cut to simulate and report.
    num_agg_sims : int
        Number of different aggregate markov simulations to run
        
    Returns
    -------
    None
    '''
    cLvl_StickyTaxCut = np.zeros(T_after)
    cLvl_StickyNone = np.zeros(T_after)
    cLvl_FrictionlessTaxCut = np.zeros(T_after)
    cLvl_FrictionlessNone = np.zeros(T_after)
    for agent in BaseEconomy.agents:
        agent.pLvlNow_start = agent.pLvlNow
        agent.pLvlTrue_start = agent.pLvlTrue
        agent.pLvlErrNow_start = agent.pLvlErrNow
        agent.aLvlNow_start = agent.aLvlNow
        agent.aNrmNow_start = agent.aNrmNow
        agent.mLvlTrueNow_start = agent.mLvlTrueNow
        agent.mNrmNow_start = agent.mNrmNow
    for i in range(4):
    # Make four copies of the economy: frictionless vs sticky, bonus vs none
        Economy = deepcopy(BaseEconomy)
        if i==0 or i==1: #sticky economies
            for agent in Economy.agents:
                agent(UpdatePrb = UpdatePrb)
        if i==2 or i==3: #frictionless economies
            for agent in Economy.agents:
                agent(UpdatePrb = 1.0)
        for n in range(num_agg_sims):
            Economy.Shk_idx = 0
            for agent in Economy.agents:
                agent.pLvlNow = agent.pLvlNow_start
                agent.pLvlTrue = agent.pLvlTrue_start
                agent.pLvlErrNow = agent.pLvlErrNow_start
                agent.aLvlNow = agent.aLvlNow_start
                agent.aNrmNow = agent.aNrmNow_start
                agent.mLvlTrueNow = agent.mLvlTrueNow_start
                agent.mNrmNow = agent.mNrmNow_start
            if i==0 or i==2: # no tax cut economies
                Economy.MrkvNow_init = 0
                Economy.MrkvNow_hist[:] = 0
                Economy.makeAggShkHist_fixMrkv()
            if i==1 or i==3: # tax cut economies
                Economy.MrkvNow_init = 1
                Economy.MrkvNow_hist[:] = 0   
                # Initialize the Markov history and set up transitions
                MrkvNow_hist = np.zeros(Economy.act_T, dtype=int)
                cutoffs = np.cumsum(Economy.MrkvArray, axis=1)
                MrkvNow = Economy.MrkvNow_init
                t = 0
                draws = Uniform().draw(N=T_after, seed=75+n)
                for t in range(draws.size):  # Add act_T_orig more periods
                    MrkvNow_hist[t] = MrkvNow
                    MrkvNow = np.searchsorted(cutoffs[MrkvNow, :], draws[t])
                Economy.MrkvNow_hist = MrkvNow_hist

                Economy.makeAggShkHist_fixMrkv()
            for n in range(len(Economy.agents)):
                Economy.agents[n].getEconomyData(Economy) # Have the consumers inherit relevant objects from the economy
    
            this_cLvl = Economy.runTaxCutExperiment(T_after)

            if i==0:
                cLvl_StickyNone += this_cLvl/num_agg_sims
            if i==1:
                cLvl_StickyTaxCut += this_cLvl/num_agg_sims
            if i==2:
                cLvl_FrictionlessNone += this_cLvl/num_agg_sims
            if i==3:
                cLvl_FrictionlessTaxCut += this_cLvl/num_agg_sims
            
        
    return cLvl_StickyTaxCut, cLvl_StickyNone, cLvl_FrictionlessTaxCut, cLvl_FrictionlessNone


