def Simulate(Run_Dict,figs_dir,Parametrization='Baseline'):
    
    
    from AggFiscalModel import AggFiscalType, AggregateDemandEconomy
    from HARK.distribution import DiscreteDistribution
    from time import time
    import numpy as np
    from copy import deepcopy
    from OtherFunctions import saveAsPickleUnderVarName,  saveAsPickle
    import os
    
    from Parameters import returnParameters 
    
    [init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
    DiscFacCount, AgentCountTotal, base_dict, num_max_iterations_solvingAD,\
    convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
    data_EducShares, max_recession_duration, num_experiment_periods,\
    recession_changes, UI_changes, recession_UI_changes,\
    TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes] = returnParameters(Parametrization=Parametrization,OutputFor='_Main.py')
              
    
    mystr = lambda x : '{:.2f}'.format(x)
    
    ## Which experiments to run / plots to show
    Run_Baseline            = Run_Dict['Run_Baseline']
    Run_Recession           = Run_Dict['Run_Recession ']
    Run_Check_Recession     = Run_Dict['Run_Check_Recession'] 
    Run_UB_Ext_Recession    = Run_Dict['Run_UB_Ext_Recession']
    Run_TaxCut_Recession    = Run_Dict['Run_TaxCut_Recession']
    Run_Check               = Run_Dict['Run_Check'] 
    Run_UB_Ext              = Run_Dict['Run_UB_Ext'] 
    Run_TaxCut              = Run_Dict['Run_TaxCut']
    Run_AD                  = Run_Dict['Run_AD ']
    Run_1stRoundAD          = Run_Dict['Run_1stRoundAD']
    Run_NonAD               = Run_Dict['Run_NonAD'] 
    
    
    try:
        os.mkdir(figs_dir)
    except OSError:
        print ("Creation of the directory %s failed" % figs_dir)
    else:
        print ("Successfully created the directory %s " % figs_dir)
    
    
    #%% 
    
 
    # Setting up AggDemandEconmy
    
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
        
        IncomeDstn_recession = [ThisType.IncomeDstn[0]*(2*(num_experiment_periods+1))] # for normal, rec, recovery  
        ThisType.IncomeDstn_recession = IncomeDstn_recession
        ThisType.IncomeDstn_recessionUI = IncomeDstn_recession
        
        EmployedIncomeDstn.X[1] = EmployedIncomeDstn.X[1]*ThisType.TaxCutIncFactor
        TaxCutStatesIncomeDstn = [EmployedIncomeDstn] + [IncomeDstn_unemp]*UBspell_normal + [IncomeDstn_unemp_nobenefits] 
        IncomeDstn_recessionTaxCut = deepcopy(IncomeDstn_recession)
        # Tax states are 2,3 (q1) 4,5 (q2) ... 16,17 (q8)
        for i in range(2*num_base_MrkvStates,18*num_base_MrkvStates,1):
            IncomeDstn_recessionTaxCut[0][i] =  TaxCutStatesIncomeDstn[np.mod(i,4)]
        ThisType.IncomeDstn_recessionTaxCut = IncomeDstn_recessionTaxCut
        
        ThisType.IncomeDstn_recessionCheck = deepcopy(IncomeDstn_recession)
    

        
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
    #base_dict['Agents'] = TypeList    
    
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
    AggDemandEconomy.switchToCounterfactualMode("base")
    AggDemandEconomy.makeIdiosyncraticShockHistories()
    
    output_keys = ['NPV_AggIncome', 'NPV_AggCons', 'AggIncome', 'AggCons']
    
    
    base_dict_agg = deepcopy(base_dict)
    
    Rspell = AggDemandEconomy.agents[0].Rspell #NOTE - this should come from the market, not the agent
    R_persist = 1.-1./Rspell
    recession_prob_array = np.array([R_persist**t*(1-R_persist) for t in range(max_recession_duration)])
    recession_prob_array[-1] = 1.0 - np.sum(recession_prob_array[:-1])
   
    x = np.zeros(len(AggDemandEconomy.agents))
    for i in range(len(AggDemandEconomy.agents)):
        x[i] = AggDemandEconomy.agents[i].AgentCount
       
        
    if Run_Baseline:   
        # Run the baseline consumption level
        t0 = time()
        base_results = AggDemandEconomy.runExperiment(**base_dict_agg, Full_Output = 'ForWelfare')
        saveAsPickleUnderVarName(base_results,figs_dir,locals())
        AggDemandEconomy.storeBaseline(base_results['AggCons'])     
        t1 = time()
        print('Calculating agg consumption took ' + mystr(t1-t0) + ' seconds.')
        
        
    #%%         
                 
        
    def runExperimentsAllRecessions(dict_changes,AggDemandEconomy):
        
        t0 = time()
        dictt = base_dict_agg.copy()
        dictt.update(**dict_changes)
        all_results = []
        avg_results = dict()
        #  running recession with diferent lengths up to max_recession_duration then averaging the result
        for t in range(max_recession_duration):
            dictt['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
            dictt['EconomyMrkv_init'][0:t+1] = np.array(dictt['EconomyMrkv_init'][0:t+1]) +1
            print(dictt['EconomyMrkv_init'])
            this_result = AggDemandEconomy.runExperiment(**dictt, Full_Output = 'ForWelfare')
            all_results += [this_result]
        for key in output_keys:
            avg_results[key] = np.sum(np.array([all_results[t][key]*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0)   
        t1 = time()
        print('Calculating took ' + mystr(t1-t0) + ' seconds.') 
        return [avg_results,all_results]
    
    def runExperimentsNoRecessions(dict_changes,AggDemandEconomy):
        
        t0 = time()
        dictt = base_dict_agg.copy()
        dictt.update(**dict_changes)
        dictt['EconomyMrkv_init'] = list(np.arange(1,AggDemandEconomy.num_experiment_periods+1)*2) + [0]*20 
        results = AggDemandEconomy.runExperiment(**dictt, Full_Output = True)
        t1 = time()
        print('Calculating took ' + mystr(t1-t0) + ' seconds.') 
        return results
    
    def Run_FullRoutineNoRecessions(shock_type):
        AggDemandEconomy_Routine = deepcopy(AggDemandEconomy)
        #Run non-recession outcomes
        if shock_type == 'Check':
            changes = Check_changes    
        elif shock_type == 'UI':
            changes = UI_changes    
        elif shock_type == 'TaxCut':     
            changes = TaxCut_changes
        if shock_type=='Check' or shock_type=='UI' or shock_type=='TaxCut':   
            print('Calculating no recession effects for shock_type: ', shock_type)
            AggDemandEconomy_Routine.switch_shock_type(shock_type)
            AggDemandEconomy_Routine.solve()
            results = runExperimentsNoRecessions(changes,AggDemandEconomy_Routine)
            saveAsPickle(shock_type + '_results',results,figs_dir)
            
    def Run_FullRoutine(shock_type):
        AggDemandEconomy_Routine = deepcopy(AggDemandEconomy)
        
        if shock_type == 'recession':
            changes = recession_changes
        elif shock_type == 'recessionCheck':
            changes = recession_Check_changes    
        elif shock_type == 'recessionUI':
            changes = recession_UI_changes    
        elif shock_type == 'recessionTaxCut':     
            changes = recession_TaxCut_changes
            
            
        if Run_NonAD:   
            print('Calculating no AD effects for shock_type: ', shock_type)
            AggDemandEconomy_Routine.switch_shock_type(shock_type)
            AggDemandEconomy_Routine.solve()
            [results,all_results] = runExperimentsAllRecessions(changes,AggDemandEconomy_Routine)
            saveAsPickle(shock_type + '_results',results,figs_dir)
            saveAsPickle(shock_type + '_all_results',all_results,figs_dir)
        
        if Run_AD:
            # Solving recession under Agg Multiplier   
            t0 = time()
            AggDemandEconomy_Routine.switch_shock_type(shock_type)
            if shock_type == 'recession':
                AggDemandEconomy_Routine.solveAD_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = shock_type)
            elif shock_type == 'recessionCheck':
                AggDemandEconomy_Routine.solveAD_Check_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = shock_type)
            elif shock_type == 'recessionUI':
                AggDemandEconomy_Routine.solveAD_UIExtension_Recession(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = shock_type)
            elif shock_type == 'recessionTaxCut':         
                AggDemandEconomy_Routine.solveAD_Recession_TaxCut(num_max_iterations=num_max_iterations_solvingAD,convergence_cutoff=convergence_tol_solvingAD, name = shock_type)
            t1 = time()
            print('Solving took ' + mystr(t1-t0) + ' seconds for shock_type: ', shock_type)
            
            print('Calculating AD effects for shock_type: ', shock_type)
            AggDemandEconomy_Routine.switch_shock_type(shock_type)
            AggDemandEconomy_Routine.restoreADsolution(name = shock_type)
            [results_AD,all_results_AD] = runExperimentsAllRecessions(changes,AggDemandEconomy_Routine)
            saveAsPickle(shock_type + '_results_AD',results_AD,figs_dir)
            saveAsPickle(shock_type + '_all_results_AD',all_results_AD,figs_dir)
        
        if Run_1stRoundAD:
            # Solving recession under Agg Multiplier   
            t0 = time()
            AggDemandEconomy_Routine.switch_shock_type(shock_type)
            
            if shock_type == 'recession':
                AggDemandEconomy_Routine.solveAD_Recession(num_max_iterations=1,convergence_cutoff=convergence_tol_solvingAD, name = shock_type + '1stRoundAD')
            elif shock_type == 'recessionCheck':
                AggDemandEconomy_Routine.solveAD_Check_Recession(num_max_iterations=1,convergence_cutoff=convergence_tol_solvingAD, name = shock_type + '1stRoundAD')
            elif shock_type == 'recessionUI':
                AggDemandEconomy_Routine.solveAD_UIExtension_Recession(num_max_iterations=1,convergence_cutoff=convergence_tol_solvingAD, name = shock_type + '1stRoundAD')
            elif shock_type == 'recessionTaxCut':         
                AggDemandEconomy_Routine.solveAD_Recession_TaxCut(num_max_iterations=1,convergence_cutoff=convergence_tol_solvingAD, name = shock_type + '1stRoundAD')
            t1 = time()
            print('Solving took ' + mystr(t1-t0) + ' seconds for 1st round AD for shock_type: ', shock_type)
           
            print('Calculating 1st round AD effects for shock_type: ', shock_type)
            AggDemandEconomy_Routine.switch_shock_type(shock_type)
            AggDemandEconomy_Routine.restoreADsolution(name = shock_type + '1stRoundAD')
            [results_firstRoundAD,all_results_firstRoundAD] = runExperimentsAllRecessions(changes,AggDemandEconomy_Routine)
            saveAsPickle(shock_type + '_results_firstRoundAD',results_firstRoundAD,figs_dir)
            saveAsPickle(shock_type + '_all_results_firstRoundAD',all_results_firstRoundAD,figs_dir)
    

         
        
    #%% Simulation
        
    if Run_Recession: 
        Run_FullRoutine('recession')
    if Run_Check_Recession:
        Run_FullRoutine('recessionCheck')
    if Run_UB_Ext_Recession:
        Run_FullRoutine('recessionUI') 
    if Run_TaxCut_Recession:
        Run_FullRoutine('recessionTaxCut')
        
    if Run_Check:
        Run_FullRoutineNoRecessions('Check')
    if Run_UB_Ext:
        Run_FullRoutineNoRecessions('UI') 
    if Run_TaxCut:
        Run_FullRoutineNoRecessions('TaxCut')
  
    
 