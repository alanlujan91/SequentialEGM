'''
This module runs the exercises and regressions for the cAndCwithStickyE paper.
User can choose which among the three models are actually run.  Descriptive
statistics and regression results are both output to screen and saved in a log
file in the results directory.  TeX code for tables in the paper are saved in
the /Tables directory.  See StickyEparams for calibrated model parameters.

This module can only be run if the following boolean variables have been defined
in the global scope: do_SOE, do_DSGE, do_RA, run_models, calc_micro_stats,
make_tables, make_emp_table, make_histogram, save_data, run_ucost_vs_pi,
run_value_vs_aggvar, run_alt_beliefs, run_parker, verbose_main, and use_stata.
These booleans are set when running any of the do_XXX.py files in the root directory.
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
import os
import numpy as np
import csv
from time import time
from copy import deepcopy
import subprocess
from StickyEmodel import StickyEmarkovConsumerType, StickyEmarkovRepAgent,\
        StickyCobbDouglasMarkovEconomy, StickySmallOpenMarkovEconomy
from HARK.utilities import plotFuncs
import matplotlib.pyplot as plt
import StickyEparams as Params
from StickyEtools import makeStickyEdataFile, runStickyEregressions, makeResultsTable,\
                  runStickyEregressionsInStata, makeParameterTable, makeEquilibriumTable,\
                  makeMicroRegressionTable, extractSampleMicroData, makeuCostVsPiFig, \
                  makeValueVsAggShkVarFig, makeValueVsPiFig, runStickyEregressionLagged, \
                  runParkerExperiment
# Import module to check RAM size:
import warnings
import psutil

# Check RAM size and warn user if less than 32G:
memory_specs = psutil.virtual_memory()
actual_RAM_size_GB  = int(round(memory_specs.total / (2.**30)))
desired_RAM_size_GB = 32
warning_message_1 = """\n\nWARNING: You have selected to run SOE and/or DSGE, with RAM < """+ str(desired_RAM_size_GB) +"""GB. This may fill up your memory and crash your computer. Please run this code on a machine with >= """+ str(desired_RAM_size_GB) +"""G of RAM. Relevant values:
    
    do_SOE      = """ + str(do_SOE)  + """
    do_DSGE     = """ + str(do_DSGE) + """
    
    actual_RAM_size_GB  = ~""" + str(actual_RAM_size_GB) + """ GB
    desired_RAM_size_GB =  """ + str(desired_RAM_size_GB) + """ GB

You may want to interrupt this code to avoid experiencing an out-of-memory computer crash.\n\n"""
warning_message_2 = """\n\nWARNING: You have selected to run both SOE and DSGE, with RAM < """+ str(2*desired_RAM_size_GB) +"""GB but > """ + str(desired_RAM_size_GB) + 'GB.\n'
warning_message_2 += 'You have enough memory to hold the full microeconomic results of one model in memory, but not both simultaneously.\n'
warning_message_2 += 'To prevent an out-of-memory crash, the simulated micro results from the SOE model have been deleted.\n'
warning_message_2 += 'If you want to work with the simulated micro data in the SOE model, you should run that model without\n'
warning_message_2 += 'running the DSGE model on the same execution run, or use a computer with at least 64GB of memory.'

if (do_SOE or do_DSGE) and actual_RAM_size_GB < desired_RAM_size_GB:
    warnings.warn(warning_message_1)

ignore_periods = Params.ignore_periods # Number of simulated periods to ignore as a "burn-in" phase
interval_size = Params.interval_size   # Number of periods in each non-overlapping subsample
total_periods = Params.periods_to_sim  # Total number of periods in simulation
interval_count = (total_periods-ignore_periods) // interval_size # Number of intervals in the macro regressions
periods_to_sim_micro = Params.periods_to_sim_micro # To save memory, micro regressions are run on a smaller sample
AgentCount_micro = Params.AgentCount_micro # To save memory, micro regressions are run on a smaller sample
my_counts = [interval_size,interval_count]
long_counts = [interval_size*interval_count,1]
mystr = lambda number : "{:.3f}".format(number)
results_dir = Params.results_dir
empirical_dir = Params.empirical_dir

# Define the function to run macroeconomic regressions, depending on whether Stata is used
if use_stata:
    runRegressions = lambda a,b,c,d,e : runStickyEregressionsInStata(a,b,c,d,e,stata_exe)
else:
    runRegressions = lambda a,b,c,d,e : runStickyEregressions(a,b,c,d,e)



# Run models and save output if this module is called from main
if __name__ == '__main__':

    ###############################################################################
    ########## SMALL OPEN ECONOMY WITH MACROECONOMIC MARKOV STATE##################
    ###############################################################################

    if do_SOE:
        if run_models:
            # Choose parameter values depending on whether or not the Parker experiment
            # is being run right now.  The main results use a single discount factor.
            if not run_parker:
                TypeCount = Params.TypeCount
                IncUnemp = Params.IncUnemp
                DiscFacSetSOE = Params.DiscFacSetSOE
            else:
                TypeCount = Params.TypeCount_parker
                IncUnemp = Params.IncUnemp_parker
                DiscFacSetSOE = Params.DiscFacSetSOE_parker
            
            # Make consumer types to inhabit the small open Markov economy
            init_dict = deepcopy(Params.init_SOE_mrkv_consumer)
            init_dict['IncUnemp'] = IncUnemp
            init_dict['AgentCount'] = Params.AgentCount // TypeCount
            StickySOEmarkovBaseType = StickyEmarkovConsumerType(**init_dict)
            StickySOEmarkovBaseType.IncomeDstn[0] = Params.StateCount*[StickySOEmarkovBaseType.IncomeDstn[0]]
            StickySOEmarkovBaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
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

            # Make results for the sticky small open Markov economy
            desc = 'Results for the sticky small open Markov economy with update probability ' + mystr(Params.UpdatePrb)
            name = 'SOEmarkovSticky'
            makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            if calc_micro_stats:
                sticky_SOEmarkov_micro_data = extractSampleMicroData(StickySOmarkovEconomy, np.minimum(StickySOmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickySOmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
            DeltaLogC_stdev = np.genfromtxt(results_dir + 'SOEmarkovStickyResults.csv', delimiter=',')[3] # For use in frictionless spec

            # Simulate the frictionless small open Markov economy
            t_start = time()
            for agent in StickySOmarkovEconomy.agents:
                agent(UpdatePrb = 1.0)
            StickySOmarkovEconomy.makeHistory()
            t_end = time()
            print('Simulating the frictionless small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            # Make results for the frictionless small open Markov economy
            desc = 'Results for the frictionless small open Markov economy (update probability 1.0)'
            name = 'SOEmarkovFrictionless'
            makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats,meas_err_base=DeltaLogC_stdev)
            if calc_micro_stats:
                frictionless_SOEmarkov_micro_data = extractSampleMicroData(StickySOmarkovEconomy, np.minimum(StickySOmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickySOmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
                makeMicroRegressionTable('CGrowCross', [frictionless_SOEmarkov_micro_data,sticky_SOEmarkov_micro_data])
                
            # If the alternate belief structure exercise was requested, re-solve the model with the alternate beliefs
            if run_alt_beliefs:
                # Save the original consumption function for comparison to the "alternate belief" cFunc
                cFunc_original_list = []
                for n in range(TypeCount):
                    cFunc_original_list.append(deepcopy(StickySOEmarkovConsumers[n].solution[0].cFunc))
                
                for n in range(TypeCount):
                    StickySOEmarkovConsumers[n].installAltShockBeliefs()
                        
                t_start = time()
                StickySOmarkovEconomy.solveAgents()
                t_end = time()
                print('Re-solving the small open Markov economy with alternate beliefs took ' + mystr(t_end-t_start) + ' seconds.')
                
                t_start = time()
                for agent in StickySOmarkovEconomy.agents:
                    agent(UpdatePrb = Params.UpdatePrb)
                StickySOmarkovEconomy.makeHistory()
                t_end = time()
                print('Simulating the alternate belief small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
                
                desc = 'Results for the alternate belief small open Markov economy with update probability ' + mystr(Params.UpdatePrb)
                name = 'SOEmarkovStickyAlt'
                makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
                
                # Calculate the difference in consumption between baseline and alternate beliefs in each state for each household
                AgentCount = np.sum([StickySOEmarkovConsumers[n].AgentCount for n in range(TypeCount)])
                cAlt = np.zeros((AgentCount,Params.StateCount)) + np.nan
                cBase = np.zeros((AgentCount,Params.StateCount)) + np.nan
                for j in range(Params.StateCount):
                    start = 0
                    for n in range(TypeCount): 
                        end = start + StickySOEmarkovConsumers[n].AgentCount
                        mNrm_temp = StickySOEmarkovConsumers[n].mNrmNow
                        Mnrm_temp = 0.5*np.ones_like(mNrm_temp)
                        cAlt[start:end,j] = StickySOEmarkovConsumers[n].solution[0].cFunc[j](mNrm_temp,Mnrm_temp)
                        cBase[start:end,j] = cFunc_original_list[n][j](mNrm_temp,Mnrm_temp)
                        start = end
                cAltVsBasePctDiff = np.log(cAlt/cBase)
                my_tol = 0.0005
                AltBeliefResultString = ''
                AltBeliefResultString += 'The mean difference between baseline and alternate belief consumption functions is ' + '{:.4%}'.format(np.mean(cAltVsBasePctDiff)) + '.\n'
                AltBeliefResultString += 'The stdev of differences between baseline and alternate belief consumption functions is ' + '{:.4%}'.format(np.std(cAltVsBasePctDiff)) + '.\n'
                AltBeliefResultString += 'The highest percentage difference between baseline and alternate belief consumption functions is ' + '{:.4%}'.format(np.max(cAltVsBasePctDiff)) + '.\n'
                AltBeliefResultString += 'The lowest percentage difference between baseline and alternate belief consumption functions is ' + '{:.4%}'.format(np.min(cAltVsBasePctDiff)) + '.\n'
                AltBeliefResultString += 'The fraction of absolute differences less than ' + '{:.2%}'.format(my_tol) + ' is ' + '{:.2%}'.format(np.mean(np.abs(cAltVsBasePctDiff) < my_tol)) + '.\n'
                with open(Params.results_dir + 'AltBeliefStats.txt','w') as f:
                    f.write(AltBeliefResultString)
                    f.close()
                print(AltBeliefResultString)
                
            if run_ucost_vs_pi:
                # Find the birth value and cost of stickiness as it varies with updating probability
                UpdatePrbVec = np.linspace(0.025,1.0,40)
                CRRA = StickySOmarkovEconomy.agents[0].CRRA
                vBirth_F = np.genfromtxt(results_dir + 'SOEmarkovFrictionlessBirthValue.csv', delimiter=',')
                uCostVec = np.zeros_like(UpdatePrbVec)
                vVec = np.zeros_like(UpdatePrbVec)
                for j in range(UpdatePrbVec.size):
                    for agent in StickySOmarkovEconomy.agents:
                        agent(UpdatePrb = UpdatePrbVec[j])
                    StickySOmarkovEconomy.makeHistory()
                    makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description='trash',filename='TEMP',save_data=False,calc_micro_stats=True)
                    vBirth_S = np.genfromtxt(results_dir + 'TEMPBirthValue.csv', delimiter=',')
                    uCost = np.mean(1. - (vBirth_S/vBirth_F)**(1./(1.-CRRA)))
                    uCostVec[j] = uCost
                    vVec[j] = np.mean(vBirth_S)
                    print('Found that uCost=' + str(uCost) + ' for Pi=' + str(UpdatePrbVec[j]))
                with open(results_dir + 'SOEuCostbyUpdatePrb.csv','w') as f:
                    my_writer = csv.writer(f, delimiter = ',', lineterminator = '\n')
                    my_writer.writerow(UpdatePrbVec)
                    my_writer.writerow(uCostVec)
                    f.close()
                with open(results_dir + 'SOEvVecByUpdatePrb.csv','w') as f:
                    my_writer = csv.writer(f, delimiter = ',', lineterminator = '\n')
                    my_writer.writerow(UpdatePrbVec)
                    my_writer.writerow(vVec)
                    f.close()
                os.remove(results_dir + 'TEMPResults.csv')
                os.remove(results_dir + 'TEMPBirthValue.csv')

            if run_value_vs_aggvar:
                # Find value as it varies with updating probability
                PermShkAggVarBase = np.linspace(0.5,1.5,40)
                PermShkAggVarVec = PermShkAggVarBase*Params.PermShkAggVar
                vVec = np.zeros_like(PermShkAggVarVec)
                for j in range(PermShkAggVarVec.size):
                    StickySOmarkovEconomy.PermShkAggStd = Params.StateCount*[np.sqrt(PermShkAggVarVec[j])]
                    StickySOmarkovEconomy.makeAggShkDstn()
                    StickySOmarkovEconomy.makeAggShkHist()
                    for agent in StickySOmarkovEconomy.agents:
                        agent(UpdatePrb = 1.0)
                        agent.getEconomyData(StickySOmarkovEconomy)
                    StickySOmarkovEconomy.solveAgents()
                    StickySOmarkovEconomy.makeHistory()
                    makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description='trash',filename='TEMP',save_data=False,calc_micro_stats=True)
                    vBirth_S = np.genfromtxt(results_dir + 'TEMPBirthValue.csv', delimiter=',')
                    v = np.mean(vBirth_S)
                    vVec[j] = v
                    print('Found that v=' + str(v) + ' for PermShkAggVar=' + str(PermShkAggVarVec[j]))
                with open(results_dir + 'SOEvVecByPermShkAggVar.csv','w') as f:
                    my_writer = csv.writer(f, delimiter = ',', lineterminator = '\n')
                    my_writer.writerow(PermShkAggVarVec)
                    my_writer.writerow(vVec)
                    f.close()
                os.remove(results_dir + 'TEMPResults.csv')
                os.remove(results_dir + 'TEMPBirthValue.csv')

        # Process the coefficients, standard errors, etc into LaTeX tables
        if make_tables:
            t_start = time()
            frictionless_panel = runRegressions('SOEmarkovFrictionlessData',interval_size,False,False,True)
            frictionless_me_panel = runRegressions('SOEmarkovFrictionlessData',interval_size,True,False,True)
            frictionless_long_panel = runRegressions('SOEmarkovFrictionlessData',interval_size*interval_count,True,False,True)
            sticky_panel = runRegressions('SOEmarkovStickyData',interval_size,False,True,True)
            sticky_me_panel = runRegressions('SOEmarkovStickyData',interval_size,True,True,True)
            sticky_alt_panel = runRegressions('SOEmarkovStickyAltData',interval_size,True,True,True)
            sticky_long_panel = runRegressions('SOEmarkovStickyData',interval_size*interval_count,True,True,True)
            if not run_parker:
                makeResultsTable('Aggregate Consumption Dynamics in SOE Model',[frictionless_me_panel,sticky_me_panel],my_counts,'SOEmrkvSimReg','tPESOEsim')
                makeResultsTable('Aggregate Consumption Dynamics in SOE Model (Alternate Beliefs)',[frictionless_me_panel,sticky_alt_panel],my_counts,'SOEmrkvSimRegAlt','tPESOEsimAlt')
                makeResultsTable('Aggregate Consumption Dynamics in SOE Model',[frictionless_panel,sticky_panel],my_counts,'SOEmrkvSimRegNoMeasErr','tPESOEsimNoMeasErr')
                makeResultsTable('Aggregate Consumption Dynamics in SOE Model',[frictionless_long_panel,sticky_long_panel],long_counts,'SOEmrkvSimRegLong','tSOEsimLong')
                makeResultsTable(None,[frictionless_me_panel],my_counts,'SOEmrkvSimRegF','tPESOEsimF')
                makeResultsTable(None,[sticky_me_panel],my_counts,'SOEmrkvSimRegS','tPESOEsimS')
            else:
                makeResultsTable('Aggregate Consumption Dynamics in SOE Model (Parker Experiment)',[frictionless_me_panel,sticky_me_panel],my_counts,'SOEmrkvSimRegHetero','tPESOEsimHetero')
            t_end = time()
            print('Running time series regressions for the small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Extra code to see how long the smoothness dynamic lasts
            lag_coeffs_array = runStickyEregressionLagged('SOEmarkovStickyData',interval_size,False,True,True)
            lag_coeffs_array_me = runStickyEregressionLagged('SOEmarkovStickyData',interval_size,True,True,True)
            
        # Run the "Parker experiment"
        if run_parker and run_models:
            t_start = time()
            
            # First, clear the simulation histories for all of the types to free up memory space;
            # this allows the economy to be copied without blowing up the computer.
            attr_list = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
            for agent in StickySOmarkovEconomy.agents:
                for attr in attr_list:
                    delattr(agent,attr+'_hist')
                agent.track_vars = [] # Don't need to track any simulated variables
                
            # The market is at the end of its pre-generated simulated shock history, so it needs to be
            # reset back to an earlier shock index that has the same Markov state as the current one.
            MrkvNow = StickySOmarkovEconomy.MrkvNow
            Shk_idx_reset = np.where(StickySOmarkovEconomy.MrkvNow_hist == MrkvNow)[0][0]
            StickySOmarkovEconomy.Shk_idx = Shk_idx_reset
            
            # Run Parker experiments for different lead times for the policy
            runParkerExperiment(StickySOmarkovEconomy,0.05,1,4,True) # One quarter ahead
            runParkerExperiment(StickySOmarkovEconomy,0.05,2,4,True) # Two quarters ahead
            runParkerExperiment(StickySOmarkovEconomy,0.05,3,4,True) # Three quarters ahead
            
            t_end = time()
            print('Running the "Parker experiment" took ' + str(t_end-t_start) + ' seconds.')
        

    ###############################################################################
    ########## COBB-DOUGLAS ECONOMY WITH MACROECONOMIC MARKOV STATE ###############
    ###############################################################################

    if do_DSGE:
        if run_models:
            if do_SOE and (actual_RAM_size_GB < 2*desired_RAM_size_GB):
                warnings.warn(warning_message_2) # Tell the user that micro data from SOE model is being deleted for memory reasons
                attr_list = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
                for agent in StickySOmarkovEconomy.agents:
                    for attr in attr_list:
                        delattr(agent,attr+'_hist')
                
            # Make consumers who will live in the Cobb-Douglas Markov economy
            StickyDSGEmarkovBaseType = StickyEmarkovConsumerType(**Params.init_DSGE_mrkv_consumer)
            StickyDSGEmarkovBaseType.IncomeDstn[0] = Params.StateCount*[StickyDSGEmarkovBaseType.IncomeDstn[0]]
            StickyDSGEmarkovBaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
            StickyDSGEmarkovConsumers = []
            for n in range(Params.TypeCount):
                StickyDSGEmarkovConsumers.append(deepcopy(StickyDSGEmarkovBaseType))
                StickyDSGEmarkovConsumers[-1].seed = n
                StickyDSGEmarkovConsumers[-1].DiscFac = Params.DiscFacSetDSGE[n]

            # Make a Cobb-Douglas economy for the agents
            StickyDSGEmarkovEconomy = StickyCobbDouglasMarkovEconomy(agents = StickyDSGEmarkovConsumers,**Params.init_DSGE_mrkv_market)
            StickyDSGEmarkovEconomy.track_vars += ['RfreeNow','wRteNow','TranShkAggNow']
            StickyDSGEmarkovEconomy.overwrite_hist = False
            StickyDSGEmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
            for n in range(Params.TypeCount):
                StickyDSGEmarkovConsumers[n].getEconomyData(StickyDSGEmarkovEconomy) # Have the consumers inherit relevant objects from the economy
                StickyDSGEmarkovConsumers[n](UpdatePrb = Params.UpdatePrb)

            # Solve the sticky heterogeneous agent DSGE model
            print('Now solving the HA-DSGE model; this could take a few hours!')
            t_start = time()
            StickyDSGEmarkovEconomy.solve()
            t_end = time()
            print('Solving the sticky Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            print('Displaying the consumption functions for the Cobb-Douglas Markov economy would be too much.')

            # Make results for the sticky Cobb-Douglas Markov economy
            desc = 'Results for the sticky Cobb-Douglas Markov economy with update probability ' + mystr(Params.UpdatePrb)
            name = 'DSGEmarkovSticky'
            makeStickyEdataFile(StickyDSGEmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            DeltaLogC_stdev = np.genfromtxt(results_dir + 'DSGEmarkovStickyResults.csv', delimiter=',')[3] # For use in frictionless spec
            if calc_micro_stats:
                sticky_DSGEmarkov_micro_data = extractSampleMicroData(StickyDSGEmarkovEconomy, np.minimum(StickyDSGEmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickyDSGEmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)

            # Store the histories of MaggNow, wRteNow, and Rfree now in _overwrite attributes
            StickyDSGEmarkovEconomy.MaggNow_overwrite = deepcopy(StickyDSGEmarkovEconomy.MaggNow_hist)
            StickyDSGEmarkovEconomy.wRteNow_overwrite = deepcopy(StickyDSGEmarkovEconomy.wRteNow_hist)
            StickyDSGEmarkovEconomy.RfreeNow_overwrite = deepcopy(StickyDSGEmarkovEconomy.RfreeNow_hist)

            # Calculate the lifetime value of being frictionless when all other agents are sticky
            if calc_micro_stats:
                StickyDSGEmarkovEconomy.overwrite_hist = True # History will be overwritten by sticky outcomes
                for agent in StickyDSGEmarkovEconomy.agents:
                    agent(UpdatePrb = 1.0) # Make agents frictionless
                StickyDSGEmarkovEconomy.makeHistory() # Simulate a history one more time

                # Save the birth value file in a temporary file and delete the other generated results files
                makeStickyEdataFile(StickyDSGEmarkovEconomy,ignore_periods,description=desc,filename=name+'TEMP',save_data=False,calc_micro_stats=calc_micro_stats)
                os.remove(results_dir + name + 'TEMP' + 'Results.csv')
                sticky_name = name

            # Solve the frictionless heterogeneous agent DSGE model
            StickyDSGEmarkovEconomy.overwrite_hist = False
            for agent in StickyDSGEmarkovEconomy.agents:
                agent(UpdatePrb = 1.0)
            t_start = time()
            StickyDSGEmarkovEconomy.solve()
            t_end = time()
            print('Solving the frictionless Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            print('Displaying the consumption functions for the Cobb-Douglas Markov economy would be too much.')

            # Make results for the frictionless Cobb-Douglas Markov economy
            desc = 'Results for the frictionless Cobb-Douglas Markov economy (update probability 1.0)'
            name = 'DSGEmarkovFrictionless'
            makeStickyEdataFile(StickyDSGEmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats,meas_err_base=DeltaLogC_stdev)
            if calc_micro_stats:
                os.remove(results_dir + name + 'BirthValue.csv') # Delete the frictionless birth value file
                os.rename(results_dir + sticky_name + 'TEMPBirthValue.csv',results_dir + name + 'BirthValue.csv') # Replace just deleted file with "alternate" value calculation
                frictionless_DSGEmarkov_micro_data = extractSampleMicroData(StickyDSGEmarkovEconomy, np.minimum(StickyDSGEmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickyDSGEmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
                makeMicroRegressionTable('CGrowCrossDSGE', [frictionless_DSGEmarkov_micro_data,sticky_DSGEmarkov_micro_data])

        # Process the coefficients, standard errors, etc into LaTeX tables
        if make_tables:
            t_start = time()
            frictionless_panel = runRegressions('DSGEmarkovFrictionlessData',interval_size,False,False,True)
            frictionless_me_panel = runRegressions('DSGEmarkovFrictionlessData',interval_size,True,False,True)
            frictionless_long_panel = runRegressions('DSGEmarkovFrictionlessData',interval_size*interval_count,True,False,True)
            sticky_panel = runRegressions('DSGEmarkovStickyData',interval_size,False,True,True)
            sticky_me_panel = runRegressions('DSGEmarkovStickyData',interval_size,True,True,True)
            sticky_long_panel = runRegressions('DSGEmarkovStickyData',interval_size*interval_count,True,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in HA-DSGE Model',[frictionless_me_panel,sticky_me_panel],my_counts,'DSGEmrkvSimReg','tDSGEsim')
            makeResultsTable('Aggregate Consumption Dynamics in HA-DSGE Model',[frictionless_panel,sticky_panel],my_counts,'DSGEmrkvSimRegNoMeasErr','tDSGEsimNoMeasErr')
            makeResultsTable('Aggregate Consumption Dynamics in HA-DSGE Model',[frictionless_long_panel,sticky_long_panel],long_counts,'DSGEmrkvSimRegLong','tDSGEsimLong')
            makeResultsTable(None,[frictionless_me_panel],my_counts,'DSGEmrkvSimRegF','tDSGEsimF')
            makeResultsTable(None,[sticky_me_panel],my_counts,'DSGEmrkvSimRegS','tDSGEsimS')
            t_end = time()
            print('Running time series regressions for the Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')



    ###############################################################################
    ########### REPRESENTATIVE AGENT ECONOMY WITH MARKOV STATE ####################
    ###############################################################################

    if do_RA:
        if run_models:
            # Make a representative agent consumer, then solve and simulate the model
            StickyRAmarkovConsumer = StickyEmarkovRepAgent(**Params.init_RA_mrkv_consumer)
            StickyRAmarkovConsumer.IncomeDstn[0] = Params.StateCount*[StickyRAmarkovConsumer.IncomeDstn[0]]
            StickyRAmarkovConsumer.track_vars = ['cLvlNow','yNrmTrue','aLvlNow','pLvlTrue','TranShkNow','MrkvNow']

            # Solve the representative agent Markov economy
            t_start = time()
            StickyRAmarkovConsumer.solve()
            t_end = time()
            print('Solving the representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            if verbose_main:
                print('Consumption functions for the Markov representative agent:')
                plotFuncs(StickyRAmarkovConsumer.solution[0].cFunc,0,50)

            # Simulate the sticky representative agent Markov economy
            t_start = time()
            StickyRAmarkovConsumer(UpdatePrb = Params.UpdatePrb)
            StickyRAmarkovConsumer.initializeSim()
            StickyRAmarkovConsumer.simulate()
            t_end = time()
            print('Simulating the sticky representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            # Make results for the sticky representative agent economy
            desc = 'Results for the sticky representative agent Markov economy'
            name = 'RAmarkovSticky'
            makeStickyEdataFile(StickyRAmarkovConsumer,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            DeltaLogC_stdev = np.genfromtxt(results_dir + 'RAmarkovStickyResults.csv', delimiter=',')[3] # For use in frictionless spec

            # Simulate the frictionless representative agent Markov economy
            t_start = time()
            StickyRAmarkovConsumer(UpdatePrb = 1.0)
            StickyRAmarkovConsumer.initializeSim()
            StickyRAmarkovConsumer.simulate()
            t_end = time()
            print('Simulating the frictionless representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            # Make results for the frictionless representative agent economy
            desc = 'Results for the frictionless representative agent Markov economy (update probability 1.0)'
            name = 'RAmarkovFrictionless'
            makeStickyEdataFile(StickyRAmarkovConsumer,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats,meas_err_base=DeltaLogC_stdev)


        if make_tables:
            # Process the coefficients, standard errors, etc into LaTeX tables
            t_start = time()
            frictionless_panel = runRegressions('RAmarkovFrictionlessData',interval_size,False,False,True)
            frictionless_me_panel = runRegressions('RAmarkovFrictionlessData',interval_size,True,False,True)
            frictionless_long_panel = runRegressions('RAmarkovFrictionlessData',interval_size*interval_count,True,False,True)
            sticky_panel = runRegressions('RAmarkovStickyData',interval_size,False,True,True)
            sticky_me_panel = runRegressions('RAmarkovStickyData',interval_size,True,True,True)
            sticky_long_panel = runRegressions('RAmarkovStickyData',interval_size*interval_count,True,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in RA Model',[frictionless_me_panel,sticky_me_panel],my_counts,'RepAgentMrkvSimReg','tRAsim')
            makeResultsTable('Aggregate Consumption Dynamics in RA Model',[frictionless_panel,sticky_panel],my_counts,'RepAgentMrkvSimRegNoMeasErr','tRAsimNoMeasErr')
            makeResultsTable('Aggregate Consumption Dynamics in RA Model',[frictionless_long_panel,sticky_long_panel],long_counts,'RepAgentMrkvSimRegLong','tRAsimLong')
            t_end = time()
            print('Running time series regressions for the representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

    ###############################################################################
    ########### MAKE OTHER TABLES AND FIGURES #####################################
    ###############################################################################
    if make_tables:
        makeEquilibriumTable('Eqbm', ['SOEmarkovFrictionless','SOEmarkovSticky','DSGEmarkovFrictionless','DSGEmarkovSticky'],Params.init_SOE_consumer['CRRA'])
        makeParameterTable('Calibration', Params)

    if run_ucost_vs_pi:
        makeuCostVsPiFig('SOEuCostbyUpdatePrb', show=verbose_main)
        makeValueVsPiFig('SOEvVecByUpdatePrb', show=verbose_main)

    if run_value_vs_aggvar:
        makeValueVsAggShkVarFig('SOEvVecByPermShkAggVar', show=verbose_main)

    if make_emp_table and use_stata:
        # Define the command to run the Stata do file
        base_path = os.path.abspath('../../')
        cmd = [stata_exe, "do", empirical_dir + "_usConsDynEmp.do", base_path]
        # Run Stata do-file
        stata_status = subprocess.call(cmd,shell = 'true')
        if stata_status!=0:
            raise ValueError('Stata code could not run. Check stata_exe in USER_OPTIONS.py')

    if make_histogram and use_stata:
        # Define the command to run the Stata do file
        base_path = os.path.abspath('../../')
        cmd = [stata_exe, "do", empirical_dir + "metaAnalysis/habitsHistogram.do", base_path]
        # Run Stata do-file
        stata_status = subprocess.call(cmd,shell = 'true')
        if stata_status!=0:
            raise ValueError('Stata code could not run. Check stata_exe in USER_OPTIONS.py')


