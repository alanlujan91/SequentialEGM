'''
This is the main script for the paper
'''
#$$$$$$$$$$ represents places in the code that need to be adjusted when the markov state space is changed
from Parameters import T_sim, init_infhorizon, DiscFacDstns,\
     AgentCountTotal, TypeShares, base_dict, recession_changes, sticky_e_changes,\
     UI_changes, recession_UI_changes, TaxCut_changes, recession_TaxCut_changes,\
     figs_dir
from FiscalModel import FiscalType
from FiscalTools import runExperiment
from HARK import multiThreadCommands, multiThreadCommandsFake
from HARK.distribution import DiscreteDistribution
from time import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#%%
if __name__ == '__main__':
    
    mystr = lambda x : '{:.2f}'.format(x)
    t_start = time()

    # Make baseline types - for now only one type, might add more
    num_types = 1
    InfHorizonType = FiscalType(**init_infhorizon)
    InfHorizonType.cycles = 0
    BaseTypeList = [InfHorizonType]
    
    # Fill in the Markov income distribution for each base type
    #$$$$$$$$$$
    # NOTE: THIS ASSUMES NO LIFECYCLE
    IncomeDstn_unemp = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonType.IncUnemp])])
    IncomeDstn_unemp_nobenefits = DiscreteDistribution(np.array([1.0]), [np.array([1.0]), np.array([InfHorizonType.IncUnempNoBenefits])])
    IncomeDstn_big = []
    for ThisType in BaseTypeList:
        IncomeDstn_taxcut = deepcopy(ThisType.IncomeDstn[0])
        IncomeDstn_taxcut.X[1] = IncomeDstn_taxcut.X[1]*ThisType.TaxCutIncFactor
        IncomeDstn_big.append([ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, extended UI
                               ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, extended UI
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,    # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # recession, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp,   # normal, payroll tax cut
                               IncomeDstn_taxcut,      IncomeDstn_unemp_nobenefits, IncomeDstn_unemp])  # recession, payroll tax cut
        #NEED TO DO THIS in loop rather than hardcoded using TaxCutPeriods*2
        ThisType.IncomeDstn[0] = [ThisType.IncomeDstn[0], IncomeDstn_unemp_nobenefits, IncomeDstn_unemp]
        ThisType.IncomeDstn_big = IncomeDstn_big #Comment income distributions for each state
            
    # Make the overall list of types
    TypeList = []
    n = 0
    for b in range(DiscFacDstns[0].X.size):
        for e in range(num_types):
            DiscFac = DiscFacDstns[e].X[b]
            AgentCount = int(np.floor(AgentCountTotal*TypeShares[e]*DiscFacDstns[e].pmf[b]))
            ThisType = deepcopy(BaseTypeList[e])
            ThisType.AgentCount = AgentCount
            ThisType.DiscFac = DiscFac
            ThisType.seed = n
            TypeList.append(ThisType)
            n += 1
    base_dict['agents'] = TypeList
       
    # Solve and simulate each type to get to the initial distribution of states
    # and then prepare for new counterfactual simulations
    t0 = time()
    baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()',
                         'switchToCounterfactualMode()', 'makeAlternateShockHistories()']
    multiThreadCommandsFake(TypeList, baseline_commands)
    t1 = time()
    print('Making the baseline distribution of states and preparing to run counterfactual simulations took ' + mystr(t1-t0) + ' seconds.')


    # Run the baseline consumption level
    t0 = time()
    base_results = runExperiment(**base_dict)
    t1 = time()
    print('Calculating baseline consumption took ' + mystr(t1-t0) + ' seconds.')

    
    
    #%%
    
    # Run the extended UI consumption level
    t0 = time()
    UI_dict = base_dict.copy()
    UI_dict.update(**UI_changes)
    UI_results = runExperiment(**UI_dict)
    # UI_dict.update(**sticky_e_changes)
    # UI_results_sticky = runExperiment(**UI_dict)
    t1 = time()
    print('Calculating extended UI consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # Run the payroll tax cut consumption level
    t0 = time()
    TaxCut_dict = base_dict.copy()
    TaxCut_dict.update(**TaxCut_changes)
    TaxCut_results = runExperiment(**TaxCut_dict)
    # TaxCut_dict.update(**sticky_e_changes)
    # TaxCut_results_sticky = runExperiment(**TaxCut_dict)
    t1 = time()
    print('Calculating payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # Run the recession consumption level
    t0 = time()
    recession_dict = base_dict.copy()
    recession_dict.update(**recession_changes)
    recession_results = runExperiment(**recession_dict)
    # recession_dict.update(**sticky_e_changes)
    # recession_results_sticky = runExperiment(**recession_dict)
    t1 = time()
    print('Calculating recession consumption took ' + mystr(t1-t0) + ' seconds.')
    
    # Run the recession and extended UI consumption level
    t0 = time()
    recession_UI_dict = base_dict.copy()
    recession_UI_dict.update(**recession_UI_changes)
    recession_UI_results = runExperiment(**recession_UI_dict)
    # recession_UI_dict.update(**sticky_e_changes)
    # recession_UI_results_sticky = runExperiment(**recession_UI_dict)
    t1 = time()
    print('Calculating recession and extended UI consumption took ' + mystr(t1-t0) + ' seconds.')
      
    # Run the recession and payroll tax cut consumption level
    t0 = time()
    recession_TaxCut_dict = base_dict.copy()
    recession_TaxCut_dict.update(**recession_TaxCut_changes)
    recession_TaxCut_results = runExperiment(**recession_TaxCut_dict)
    # recession_TaxCut_dict.update(**sticky_e_changes)
    # recession_TaxCut_results_sticky = runExperiment(**recession_TaxCut_dict)
    t1 = time()
    print('Calculating recession and payroll tax cut consumption took ' + mystr(t1-t0) + ' seconds.')
 
    t_end = time()
    print('Doing everything took ' + mystr(t_end-t_start) + ' seconds in total.')
    
    #%% Fiscal expenditure effectiveness
    
    to_plot1 = 'NPV_AggCons'
    to_plot2 = 'NPV_AggIncome'
    to_plot3 = 'AggCons'
    to_plot4 = 'AggIncome'
    
    add_plot_text = ''
    
    
    NPV_AddCons = UI_results[to_plot1]-base_results[to_plot1]
    NPV_AddInc  = UI_results[to_plot2]-base_results[to_plot2]  
    AddCons     = UI_results[to_plot3]-base_results[to_plot3]
    AddInc      = UI_results[to_plot4]-base_results[to_plot4] 
    plt.plot(AddInc)
    plt.plot(AddCons)
    plt.legend(['Fiscal policy expenditure, UI extension','Additional consumption, UI extension'])
    plt.savefig(figs_dir +'UI_cut' + add_plot_text +'.pdf')
    plt.show()
    Stimulus_UI    = AddCons/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy

    
    NPV_AddCons = recession_UI_results[to_plot1]-recession_results[to_plot1]
    NPV_AddInc  = recession_UI_results[to_plot2]-recession_results[to_plot2]  
    AddCons     = recession_UI_results[to_plot3]-recession_results[to_plot3]
    AddInc      = recession_UI_results[to_plot4]-recession_results[to_plot4] 
    plt.plot(AddInc)
    plt.plot(AddCons)
    plt.legend(['Fiscal policy expenditure, UI extension during recession','Additional consumption, UI extension during recession'])
    plt.savefig(figs_dir +'UI_cut_rec' + add_plot_text +'.pdf')
    plt.show()
    Stimulus_UI_rec    = AddCons/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy

    
    NPV_AddCons = TaxCut_results[to_plot1]-base_results[to_plot1]
    NPV_AddInc  = TaxCut_results[to_plot2]-base_results[to_plot2]  
    AddCons     = TaxCut_results[to_plot3]-base_results[to_plot3]
    AddInc      = TaxCut_results[to_plot4]-base_results[to_plot4] 
    plt.plot(AddInc)
    plt.plot(AddCons)
    plt.legend(['Fiscal policy expenditure, tax cut','Additional consumption, tax cut'])
    plt.savefig(figs_dir +'tax_cut' + add_plot_text +'.pdf')
    plt.show()
    Stimulus_taxcut    = AddCons/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy

    
    NPV_AddCons = recession_TaxCut_results[to_plot1]-recession_results[to_plot1]
    NPV_AddInc  = recession_TaxCut_results[to_plot2]-recession_results[to_plot2]  
    AddCons     = recession_TaxCut_results[to_plot3]-recession_results[to_plot3]
    AddInc      = recession_TaxCut_results[to_plot4]-recession_results[to_plot4] 
    plt.plot(AddInc)
    plt.plot(AddCons)
    plt.legend(['Fiscal policy expenditure, tax cut during recession','Additional consumption, tax cut during recession'])
    plt.savefig(figs_dir +'tax_cut_rec' + add_plot_text +'.pdf')
    plt.show()
    Stimulus_taxcut_rec    = AddCons/NPV_AddInc[-1]  #divide by total cumulative NPV of the policy
  
    
    # Compare stimulus effects across policy interventions
    plt.plot(Stimulus_UI)
    plt.plot(Stimulus_UI_rec)
    plt.plot(Stimulus_taxcut)
    plt.plot(Stimulus_taxcut_rec)
    plt.title('Stimulated consumption per period relative to NPV of policy intervention')
    plt.legend(['UI','recession_UI','TaxCut','recession_TaxCut'])
    plt.savefig(figs_dir +'stimulated-consumption' + add_plot_text +'.pdf')
    plt.show()
 
    
    #%%
    
    # to_plot = 'cLvl_all_splurge'
    # plt.plot(np.mean(base_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_results[to_plot],axis=1))
    # plt.plot(np.mean(UI_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_UI_results[to_plot],axis=1))
    # plt.plot(np.mean(TaxCut_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_TaxCut_results[to_plot],axis=1))
    # plt.legend(['base','recession','UI','recession_UI','TaxCut','recession_TaxCut'])
    # plt.title(to_plot)
    # plt.savefig(figs_dir +'ScenarioPaths_'+to_plot+'.pdf')
    # plt.show()
    
    # plt.plot(np.mean(UI_results[to_plot]-base_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_UI_results[to_plot]-recession_results[to_plot],axis=1))
    # plt.plot(np.mean(TaxCut_results[to_plot]-base_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_TaxCut_results[to_plot]-recession_results[to_plot],axis=1))
    # plt.legend(['UI','recession_UI','TaxCut','recession_TaxCut'])
    # plt.title(to_plot + 'Policy vs no policy')
    # plt.savefig(figs_dir +'PolicyVsNoPolicy_'+to_plot+'.pdf')
    # plt.show()
    
    # # sticky vs frictionless
    # plt.plot(np.mean(recession_results_sticky[to_plot]-recession_results[to_plot],axis=1))
    # plt.plot(np.mean(UI_results_sticky[to_plot]-UI_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_UI_results_sticky[to_plot]-recession_UI_results[to_plot],axis=1))
    # plt.plot(np.mean(TaxCut_results_sticky[to_plot]-TaxCut_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_TaxCut_results_sticky[to_plot]-recession_TaxCut_results[to_plot],axis=1))
    # plt.legend(['recession','UI','recession_UI','TaxCut','recession_TaxCut'])
    # plt.title(to_plot + ' Sticky vs Frictionless')
    # plt.savefig(figs_dir +'StickyVsFrictionless_'+to_plot+'.pdf')
    # plt.show()
    
    # # sticky vs frictionless
    # plt.plot(np.mean(base_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_results_sticky[to_plot],axis=1))
    # plt.plot(np.mean(recession_UI_results_sticky[to_plot],axis=1))
    # plt.plot(np.mean(recession_TaxCut_results_sticky[to_plot],axis=1))
    # plt.legend(['Baseline','recession frictionless','recession sticky','recession_UI sticky','recession_TaxCut sticky'])
    # plt.title(to_plot + ' Sticky vs Frictionless')
    # plt.savefig(figs_dir +'StickyVsFrictionless2_'+to_plot+'.pdf')
    # plt.show()
    
    # # Splurge vs no-splurge
    # to_plot = 'cLvl_all'
    # to_plot2 = 'cLvl_all_splurge'
    # plt.plot(np.mean(base_results[to_plot2]-base_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_results[to_plot2]-recession_results[to_plot],axis=1))
    # plt.plot(np.mean(UI_results[to_plot2]-UI_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_UI_results[to_plot2]-recession_UI_results[to_plot],axis=1))
    # plt.plot(np.mean(TaxCut_results[to_plot2]-TaxCut_results[to_plot],axis=1))
    # plt.plot(np.mean(recession_TaxCut_results[to_plot2]-recession_TaxCut_results[to_plot],axis=1))
    # plt.legend(['base','recession','UI','recession_UI','TaxCut','recession_TaxCut'])
    # plt.title(to_plot)
    # plt.savefig(figs_dir +'consumption_splurge_vs_not.pdf')
    # plt.show()
