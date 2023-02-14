'''
Runs Fiscal Experiments:
    1) Parker style one off payment, arriving in 2 quarters
    2) Bush style tax cut on wages for a duration of about 2 years
'''

from builtins import str
from builtins import range
import numpy as np
from time import time
from copy import deepcopy
from StickyEmodel import StickyEmarkovConsumerType, StickySmallOpenMarkovEconomy
import matplotlib.pyplot as plt
import StickyEparams as Params
from StickyEtools import runParkerExperiment, runTaxCutExperiment

mystr = lambda number : "{:.3f}".format(number)
results_dir = Params.results_dir

# Run models and save output if this module is called from main
if __name__ == '__main__':

    ###############################################################################
    ########## SMALL OPEN ECONOMY WITH MACROECONOMIC MARKOV STATE##################
    ###############################################################################

    run_models = True
    run_parker = True
    verbose_main = True
    save_data = True
    if run_models:
        TypeCount = Params.TypeCount
        IncUnemp = Params.IncUnemp
        DiscFacSetSOE = Params.DiscFacSetSOE
        
        # Make consumer types to inhabit the small open Markov economy
        init_dict = deepcopy(Params.init_SOE_mrkv_consumer)
        init_dict['IncUnemp'] = IncUnemp
        init_dict['AgentCount'] = Params.AgentCount // TypeCount
        StickySOEmarkovBaseType = StickyEmarkovConsumerType(**init_dict)
        StickySOEmarkovBaseType.IncomeDstn[0] = Params.StateCount*[StickySOEmarkovBaseType.IncomeDstn[0]]
        StickySOEmarkovBaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','pLvlNow','t_age','TranShkNow','MrkvNowPcvd','MrkvNow']
        StickySOEmarkovConsumers = []
        for n in range(TypeCount):
            StickySOEmarkovConsumers.append(deepcopy(StickySOEmarkovBaseType))
            StickySOEmarkovConsumers[-1].seed = n
            StickySOEmarkovConsumers[-1].DiscFac = DiscFacSetSOE[n]

        # Make a small open economy for the agents
        StickySOmarkovEconomy = StickySmallOpenMarkovEconomy(agents=StickySOEmarkovConsumers, **Params.init_SOE_mrkv_market)
        StickySOmarkovEconomy.track_vars += ['TranShkAggNow','wRteNow']
        StickySOmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
        for n in range(TypeCount):
            StickySOEmarkovConsumers[n].getEconomyData(StickySOmarkovEconomy) # Have the consumers inherit relevant objects from the economy

        # Solve the small open Markov model
        t_start = time()
        print('Now solving the SOE model; this will take a few minutes.')
        StickySOmarkovEconomy.solveAgents()
        t_end = time()
        print('Solving the small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

        # Plot the consumption function in each Markov state
        my_new_title = 'Consumption function for one type in the small open Markov economy:'
        m = np.linspace(0.,20.,500)
        M = np.ones_like(m)
        c = np.zeros((Params.StateCount,m.size))
        for i in range(Params.StateCount):
            c[i,:] = StickySOEmarkovConsumers[0].solution[0].cFunc[i](m,M)
            plt.plot(m,c[i,:])
        plt.title(my_new_title)
        plt.xlim([0.,20.])
        plt.ylim([0.,None])
        if verbose_main:
            print(my_new_title)
            plt.show()
        plt.close()

        # Simulate the sticky small open Markov economy
        t_start = time()
        for agent in StickySOmarkovEconomy.agents:
            agent(UpdatePrb = Params.UpdatePrb)
        StickySOmarkovEconomy.makeHistory()
        t_end = time()
        print('Simulating the sticky small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

        run_parker = True
        run_tax_cut = True
        if run_parker or run_tax_cut:
            # First, clear the simulation histories for all of the types to free up memory space;
            # this allows the economy to be copied without blowing up the computer.
            for agent in StickySOmarkovEconomy.agents:
                delattr(agent,'history')
                agent.track_vars = [] # Don't need to track any simulated variables
            # The market is at the end of its pre-generated simulated shock history, so it needs to be
            # reset back to the beginning
            StickySOmarkovEconomy.Shk_idx = 0
                
            #       Run the "Parker experiment"
            if run_parker:
                t_start = time()               
                # Run Parker experiments for different lead times for the policy
                cLvl_StickyNone_parker, cLvl_StickyBonus, cLvl_FrictionlessNone_parker, cLvl_FrictionlessBonus = runParkerExperiment(StickySOmarkovEconomy,0.05,1,30,True) # One quarter ahead
                t_end = time()
                print('Running the "Parker experiment" took ' + str(t_end-t_start) + ' seconds.')
                plt.plot(cLvl_StickyBonus/cLvl_StickyNone_parker, label="Sticky")
                plt.plot(cLvl_FrictionlessBonus/cLvl_FrictionlessNone_parker, label="Frictionless")
                plt.legend(loc="upper right")
                plt.show()
    #       Run the "Tax Cut experiment"
            if run_tax_cut:
                num_agg_sims =100
                T_after = 40
                t_start = time()
                cLvl_StickyTaxCut, cLvl_StickyNone, cLvl_FrictionlessTaxCut, cLvl_FrictionlessNone = runTaxCutExperiment(StickySOmarkovEconomy,T_after=T_after,num_agg_sims=num_agg_sims) 
                t_end = time()
                print('Running the "Tax Cut experiment" took ' + str(t_end-t_start) + ' seconds.')
                
                plt.plot(cLvl_StickyTaxCut/cLvl_StickyNone, label="Sticky")
                plt.plot(cLvl_FrictionlessTaxCut/cLvl_FrictionlessNone, label="Frictionless")
                plt.legend(loc="upper right")
                plt.show()
                
    #       Make pretty plot
            if run_parker and run_tax_cut:
                plt.plot(cLvl_StickyBonus/cLvl_StickyNone_parker, label="Stimulus Payments")
                plt.plot(cLvl_StickyTaxCut[1:]/cLvl_StickyNone[1:], label="Income Tax Cut")
                plt.legend(loc="upper right")
                plt.xlabel('Quarter',fontsize=12)
                plt.ylabel('Percentage Consumption Increase',fontsize=12)
                plt.title('Fiscal Experiments')
                plt.savefig('./Figures/FiscalExperiments.png')
                plt.show()
