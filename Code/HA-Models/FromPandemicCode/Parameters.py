def returnParameters(Parametrization='Baseline',OutputFor='_Main.py'):
    
    import os
    
    # for output
    cwd             = os.getcwd()
    folders         = cwd.split(os.path.sep)
    top_most_folder = folders[-1]
    if top_most_folder == 'FromPandemicCode':
        Abs_Path_Results = "".join([x + "//" for x in folders[0:-1]],)
        Abs_Path = cwd
    else:
        Abs_Path_Results = cwd
        Abs_Path = cwd + '\\FromPandemicCode'

    import numpy as np
    from HARK.distribution import Uniform
    from OtherFunctions import loadPickle, getSimulationDiff
    

    
    from EstimParameters import data_EducShares, Urate_normal_d, Urate_normal_h, Urate_normal_c,\
    Uspell_normal, UBspell_normal, PopGroFac, PermGroFacAgg, IncUnemp,\
    pLvlInitMean_d, pLvlInitMean_h, pLvlInitMean_c,\
    pLvlInitStd_d, pLvlInitStd_h, pLvlInitStd_c,\
    PermGroFac_base_d, PermGroFac_base_h, PermGroFac_base_c,\
    TranShkStd, PermShkStd, LivPrb_base, num_types
    

    CRRA = 2.0
    Rfree_base = [1.01]
    Rspell = 6            # Expected length of recession, in quarters. If R_shared = True, must be an integer
    betas_txt_location = Abs_Path_Results+'/Results/DiscFacEstim_CRRA_2.0_R_1.01.txt' 
    Splurge_txt_location = Abs_Path_Results+'/Target_AggMPCX_LiquWealth/Result_CRRA_2.0.txt'
    IncUnempNoBenefits = 0.5   # Unemployment income when benefits run out (proportion of permanent income)
    ADelasticity = 0.3      
    
    # make changes according to robustness run
    if Parametrization == 'ADElas' or Parametrization == 'ADElas_PVSame':
        ADelasticity = 0.5
    elif Parametrization == 'CRRA1' or Parametrization == 'CRRA1_PVSame':
        CRRA = 1.0
        betas_txt_location = Abs_Path_Results+'/Results/DiscFacEstim_CRRA_1.0_R_1.01.txt' 
        Splurge_txt_location = Abs_Path_Results+'/Target_AggMPCX_LiquWealth/Result_CRRA_1.0.txt'  
    elif Parametrization == 'CRRA3' or Parametrization == 'CRRA3_PVSame':
        CRRA = 3.0
        betas_txt_location = Abs_Path_Results+'/Results/DiscFacEstim_CRRA_3.0_R_1.01.txt'
        Splurge_txt_location = Abs_Path_Results+'/Target_AggMPCX_LiquWealth/Result_CRRA_3.0.txt'  
    elif Parametrization == 'Rfree_1005' or Parametrization == 'Rfree_1005_PVSame':
        Rfree_base = [1.005]
        betas_txt_location = Abs_Path_Results+'/Results/DiscFacEstim_CRRA_2.0_R_1.005.txt'  
    elif Parametrization == 'Rfree_1015' or Parametrization == 'Rfree_1015_PVSame':
        Rfree_base = [1.015]
        betas_txt_location = Abs_Path_Results+'/Results/DiscFacEstim_CRRA_2.0_R_1.015.txt'
    elif Parametrization == 'Rspell_4' or Parametrization == 'Rspell_4_PVSame':
        Rspell = 4
    elif Parametrization == 'LowerUBnoB' or Parametrization == 'LowerUBnoB_PVSame':
        betas_txt_location = Abs_Path_Results+'/Results/DiscFacEstim_CRRA_2.0_R_1.01_altBenefits.txt'
        IncUnempNoBenefits = 0.15
        IncUnemp = 0.3
    
    myEstim = [[],[],[]]
    f = open(betas_txt_location, 'r')
    readStr = f.readline().strip()
    while readStr != '' :
        dictload = eval(readStr)
        edType = dictload['EducationGroup']
        beta = dictload['beta']
        nabla = dictload['nabla']
        myEstim[edType] = [beta,nabla]
        readStr = f.readline().strip()
    f.close()
    
    
    f = open(Splurge_txt_location, 'r')
    if f.mode=='r':
        contents= f.read()
    dictload= eval(contents)
    Splurge = dictload['splurge']
    
    
    if Parametrization == 'CRRA2_PVSame':
        figs_dir_FullRun = Abs_Path+'/Figures/CRRA2/'
    elif Parametrization == 'ADElas_PVSame':
        figs_dir_FullRun = Abs_Path+'/Figures/ADElas/'           
    elif Parametrization == 'CRRA1_PVSame':
        figs_dir_FullRun = Abs_Path+'/Figures/CRRA1/'
    elif Parametrization == 'CRRA3_PVSame':
        figs_dir_FullRun = Abs_Path+'/Figures/CRRA3/'
    elif Parametrization == 'Rfree_1005_PVSame':
        figs_dir_FullRun = Abs_Path+'/Figures/Rfree_1005/'
    elif Parametrization == 'Rfree_1015_PVSame':
        figs_dir_FullRun = Abs_Path+'/Figures/Rfree_1015/'
    elif Parametrization == 'Rspell_4_PVSame':
        figs_dir_FullRun = Abs_Path+'/Figures/Rspell_4/'
    elif Parametrization == 'LowerUBnoB_PVSame':
        figs_dir_FullRun = Abs_Path+'/Figures/LowerUBnoB/'


        
    if Parametrization.find('PVSame')>0:
        base_results                    = loadPickle('base_results',figs_dir_FullRun,locals())
        check_results                   = loadPickle('Check_results',figs_dir_FullRun,locals())
        UI_results                      = loadPickle('UI_results',figs_dir_FullRun,locals())
        TaxCut_results                  = loadPickle('TaxCut_results',figs_dir_FullRun,locals())
        NPV_AddInc_Check                = getSimulationDiff(base_results,check_results,'NPV_AggIncome') 
        NPV_AddInc_UI                   = getSimulationDiff(base_results,UI_results,'NPV_AggIncome') # Policy expenditure
        NPV_AddInc_TaxCut               = getSimulationDiff(base_results,TaxCut_results,'NPV_AggIncome')
        
        TaxCutAdjFactor = NPV_AddInc_UI[-1]/NPV_AddInc_TaxCut[-1]
        CheckAdjFactor =  NPV_AddInc_UI[-1]/NPV_AddInc_Check[-1]
    else:
        TaxCutAdjFactor = 1
        CheckAdjFactor = 1

    
    if not OutputFor == '_Model.py':
        print('CRRA = ',CRRA)
        print('Rfree_base = ',Rfree_base)
        print('Rspell = ',Rspell)
        
        print('Splurge = ',Splurge)
        print('beta/nablas = ',myEstim)
        
        print('TaxCutAdjFactor = ',TaxCutAdjFactor)
        print('CheckAdjFactor = ',CheckAdjFactor)
        
        print('IncUnempNoBenefits = ', IncUnempNoBenefits)
        print('IncUnemp = ', IncUnemp)
        print('ADelasticity = ', ADelasticity)

       
    # # Targets in the estimation of the discount factor distributions for each 
    # # education level. 
    # # From SCF 2004: [20,40,60,80]-percentiles of the Lorenz curve for liquid wealth
    # data_LorenzPts_d = np.array([0, 0.01, 0.60, 3.58])    # \
    # data_LorenzPts_h = np.array([0.06, 0.63, 2.98, 11.6]) # -> units: % 
    # data_LorenzPts_c = np.array([0.15, 0.92, 3.27, 10.3]) # /
    # data_LorenzPts = [data_LorenzPts_d, data_LorenzPts_h, data_LorenzPts_c]
    # data_LorenzPtsAll = np.array([0.03, 0.35, 1.84, 7.42])
    # # From SCF 2004: Average liquid wealth to permanent income ratio 
    # data_avgLWPI = np.array([15.7, 47.7, 111])*4 # weighted average of fractions in percent
    # # From SCF 2004: Total LW over total PI by education group
    # data_LWoPI = np.array([28.1, 59.6, 162])*4 # units: %
    # # From SCF 2004: Weighted median of liquid wealth to permanent income ratio
    # data_medianLWPI = np.array([1.16, 7.55, 28.2])*4 # weighted median of fractions in percent
    
    

    

    # Parameters concerning the distribution of discount factors
    DiscFacMeanD = myEstim[0][0]  # Mean intertemporal discount factor for dropout types
    DiscFacMeanH = myEstim[1][0]  # Mean intertemporal discount factor for high school types
    DiscFacMeanC = myEstim[2][0]  # Mean intertemporal discount factor for college types
    # DiscFacInit = [DiscFacMeanD, DiscFacMeanH, DiscFacMeanC]
    DiscFacSpreadD = myEstim[0][1]
    DiscFacSpreadH = myEstim[1][1]
    DiscFacSpreadC = myEstim[2][1] 
    
    # Define the distribution of the discount factor for each eduation level
    DiscFacCount = 7
    DiscFacDstnD = Uniform(DiscFacMeanD-DiscFacSpreadD, DiscFacMeanD+DiscFacSpreadD).approx(DiscFacCount)
    DiscFacDstnH = Uniform(DiscFacMeanH-DiscFacSpreadH, DiscFacMeanH+DiscFacSpreadH).approx(DiscFacCount)
    DiscFacDstnC = Uniform(DiscFacMeanC-DiscFacSpreadC, DiscFacMeanC+DiscFacSpreadC).approx(DiscFacCount)
    DiscFacDstns = [DiscFacDstnD, DiscFacDstnH, DiscFacDstnC]
    
    # Calculate max beta values for each education group where GIC holds with equality: 
    GICmaxBetas = [(PermGroFac_base_d[0]**CRRA)/Rfree_base[0], (PermGroFac_base_h[0]**CRRA)/Rfree_base[0], 
                       (PermGroFac_base_c[0]**CRRA)/Rfree_base[0]]
    GICfactor = 0.9975
    minBeta = 0.01
    
    for e in range(num_types):
        for thedf in range(DiscFacCount):
            if DiscFacDstns[e].X[thedf] > GICmaxBetas[e]*GICfactor: 
                DiscFacDstns[e].X[thedf] = GICmaxBetas[e]*GICfactor
            elif DiscFacDstns[e].X[thedf] < minBeta:
                DiscFacDstns[e].X[thedf] = minBeta

  
    # Recession
    Urate_recession_d = 2 * Urate_normal_d # Unemployment rate in recession
    Urate_recession_h = 2 * Urate_normal_h 
    Urate_recession_c = 2 * Urate_normal_c
    
    Uspell_recession = 4         # Average duration of unemployment spell in recession, in quarters
    R_shared = False             # Indicator for whether the recession shared (True) or idiosyncratic (False)
    # UI extension
    UBspell_extended = 5         # Average duration of unemployment benefits when extended and assuming policy remains in place, in quarters
    PolicyUBspell = 2            # Average duration that policy of extended unemployment benefits is in place
    # Tax Cut parameter
    PolicyTaxCutspell = 2        # Average duration that policy of payroll tax cuts
    TaxCutIncFactor = 1 + 0.02*TaxCutAdjFactor
    TaxCutPeriods = 8            # Deterministic duTrueration of tax cut 
    TaxCutContinuationProb_Rec = 0.5   # Probability that tax cut is continued after tax cut periods run out, when recession in q8
    TaxCutContinuationProb_Bas = 0.0   # Probability that tax cut is continued after tax cut periods run out, when baseline in q8
    #Check experiment parameter
    CheckStimLvl = 1200/1000 * CheckAdjFactor
    CheckStimLvl_PLvl_Cutoff_start = 100/4/1 #100 k yearly income #At this Level of permanent inc, Stimulus beings to fall linearly
    CheckStimLvl_PLvl_Cutoff_end = 150/4/1 #150k yearly income #At this Level of permanent inc, Stimulus is zero
    
    
    # UpdatePrb = 0.25    # probability of updating macro state (when sticky expectations is on)
 
  

    # Parameters concerning grid sizes: assets, permanent income shocks, transitory income shocks
    aXtraMin = 0.001        # Lowest non-zero end-of-period assets above minimum gridpoint
    aXtraMax = 40           # Highest non-zero end-of-period assets above minimum gridpoint
    aXtraCount = 48         # Base number of end-of-period assets above minimum gridpoints
    aXtraExtra = [0.002,0.003] # Additional gridpoints to "force" into the grid
    aXtraNestFac = 3        # Exponential nesting factor for aXtraGrid (how dense is grid near zero)
    PermShkCount = 7        # Number of points in equiprobable discrete approximation to permanent shock distribution
    TranShkCount = 7        # Number of points in equiprobable discrete approximation to transitory shock distribution
    T_age = 200             # Kill off agents who have worked for 50 years
    
    
    # Size of simulations
    AgentCountTotal = 10000     # Total simulated population
    T_sim = 40                  # Number of quarters to simulate in counterfactuals
    
    # Basic lifecycle length parameters (don't touch these)
    T_cycle = 1
    
    # Define grid of aggregate assets to labor
    CgridBase = np.array([0.8, 1.0, 1.2])  
    
    num_base_MrkvStates = 2 + UBspell_normal #employed, unemployed with 2 quarters benefits, unemployed with 1 quarter benefit, unemployed no benefits
    num_experiment_periods = 20
    max_recession_duration = 21
    
    def makeMacroMrkvArray_recession(Rspell, num_experiment_periods):
        R_persist = 1.-1./Rspell
        recession_transitions = np.array([[1.0, 0.0],[1-R_persist, R_persist]])
        MacroMrkvArray = np.zeros((2*(num_experiment_periods+1), 2*(num_experiment_periods+1)))
        MacroMrkvArray[0:2,0:2] = recession_transitions
        for i in np.array(range(num_experiment_periods-1))+1:
            MacroMrkvArray[2*i:2*i+2, 2*i+2:2*i+4] = recession_transitions
        MacroMrkvArray[2*num_experiment_periods:2*num_experiment_periods+2, 0:2] = recession_transitions 
        return MacroMrkvArray
    
    def small_MrkvArray(e,u,ub,transition_ub=True):
        small_MrkvArray = np.zeros((ub+2, ub+2))
        small_MrkvArray[0,0] = e
        small_MrkvArray[0,1] = 1-e
        for i in np.array(range(ub))+1:
            if transition_ub:
                small_MrkvArray[i,i+1] = u
            else:
                small_MrkvArray[i,i] = u
            small_MrkvArray[i,0] = 1-u
        small_MrkvArray[ub+1,ub+1] = u
        small_MrkvArray[ub+1,0] = 1-u
        return small_MrkvArray 
    
    def makeCondMrkvArrays_base(Urate_normal, Uspell_normal, UBspell_normal):
        U_persist_normal = 1.-1./Uspell_normal
        E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
        MrkvArray_normal         = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBspell_normal)
        CondMrkvArrays = [MrkvArray_normal]
        return CondMrkvArrays
    
    def makeCondMrkvArrays_recession(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, num_experiment_periods):
        U_persist_normal = 1.-1./Uspell_normal
        E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
        U_persist_recession = 1.-1./Uspell_recession
        E_persist_recession = 1.-Urate_recession*(1.-U_persist_recession)/(1.-Urate_recession)
        MrkvArray_normal         = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBspell_normal)
        MrkvArray_recession      = small_MrkvArray(E_persist_recession, U_persist_recession, UBspell_normal)
        CondMrkvArrays = [MrkvArray_normal, MrkvArray_recession]*(num_experiment_periods+1)
        return CondMrkvArrays
    
    def makeCondMrkvArrays_recessionUI(Urate_normal, Uspell_normal, UBspell_normal, Urate_recession, Uspell_recession, num_experiment_periods, ExtraUBperiods):
        U_persist_normal = 1.-1./Uspell_normal
        E_persist_normal = 1.-Urate_normal*(1.-U_persist_normal)/(1.-Urate_normal)
        U_persist_recession = 1.-1./Uspell_recession
        E_persist_recession = 1.-Urate_recession*(1.-U_persist_recession)/(1.-Urate_recession)
        MrkvArray_normal         = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBspell_normal)
        MrkvArray_recession      = small_MrkvArray(E_persist_recession, U_persist_recession, UBspell_normal)
        MrkvArray_normalUI       = small_MrkvArray(E_persist_normal,    U_persist_normal,    UBspell_normal, transition_ub=False)
        MrkvArray_recessionUI    = small_MrkvArray(E_persist_recession, U_persist_recession, UBspell_normal, transition_ub=False)
        CondMrkvArrays = [MrkvArray_normal, MrkvArray_recession] + [MrkvArray_normalUI, MrkvArray_recessionUI]*ExtraUBperiods + [MrkvArray_normal, MrkvArray_recession]*(num_experiment_periods-ExtraUBperiods)
        return CondMrkvArrays
    
    
    def makeFullMrkvArray(MacroMrkvArray, CondMrkvArrays):
        for i in range(MacroMrkvArray.shape[0]):
            this_row = MacroMrkvArray[i,0]*CondMrkvArrays[0]
            for j in range(MacroMrkvArray.shape[0]-1):
                this_row = np.concatenate((this_row, MacroMrkvArray[i,j+1]*CondMrkvArrays[j+1]),axis=1)
            if i==0:
                FullMrkv = this_row
            else:
                FullMrkv = np.concatenate((FullMrkv, this_row), axis=0)
        return [FullMrkv]
    
    MacroMrkvArray_base = np.array([[1.0]])
    CondMrkvArrays_base_d = makeCondMrkvArrays_base(Urate_normal_d, Uspell_normal, UBspell_normal)
    CondMrkvArrays_base_h = makeCondMrkvArrays_base(Urate_normal_h, Uspell_normal, UBspell_normal)
    CondMrkvArrays_base_c = makeCondMrkvArrays_base(Urate_normal_c, Uspell_normal, UBspell_normal)
    MrkvArray_base_d = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_d)
    MrkvArray_base_h = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_h)
    MrkvArray_base_c = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_c)
    
    MacroMrkvArray_recession = makeMacroMrkvArray_recession(Rspell, num_experiment_periods)
    CondMrkvArrays_recession_d = makeCondMrkvArrays_recession(Urate_normal_d, Uspell_normal, UBspell_normal, Urate_recession_d, Uspell_recession, num_experiment_periods)
    CondMrkvArrays_recession_h = makeCondMrkvArrays_recession(Urate_normal_h, Uspell_normal, UBspell_normal, Urate_recession_h, Uspell_recession, num_experiment_periods)
    CondMrkvArrays_recession_c = makeCondMrkvArrays_recession(Urate_normal_c, Uspell_normal, UBspell_normal, Urate_recession_c, Uspell_recession, num_experiment_periods)
    MrkvArray_recession_d = makeFullMrkvArray(MacroMrkvArray_recession, CondMrkvArrays_recession_d)
    MrkvArray_recession_h = makeFullMrkvArray(MacroMrkvArray_recession, CondMrkvArrays_recession_h)
    MrkvArray_recession_c = makeFullMrkvArray(MacroMrkvArray_recession, CondMrkvArrays_recession_c)
    
    MacroMrkvArray_recessionCheck = MacroMrkvArray_recession
    CondMrkvArrays_recessionCheck_d = CondMrkvArrays_recession_d
    CondMrkvArrays_recessionCheck_h = CondMrkvArrays_recession_h
    CondMrkvArrays_recessionCheck_c = CondMrkvArrays_recession_c
    MrkvArray_recessionCheck_d = makeFullMrkvArray(MacroMrkvArray_recessionCheck, CondMrkvArrays_recessionCheck_d)
    MrkvArray_recessionCheck_h = makeFullMrkvArray(MacroMrkvArray_recessionCheck, CondMrkvArrays_recessionCheck_h)
    MrkvArray_recessionCheck_c = makeFullMrkvArray(MacroMrkvArray_recessionCheck, CondMrkvArrays_recessionCheck_c)
    
    MacroMrkvArray_recessionTaxCut = MacroMrkvArray_recession
    CondMrkvArrays_recessionTaxCut_d = CondMrkvArrays_recession_d
    CondMrkvArrays_recessionTaxCut_h = CondMrkvArrays_recession_h
    CondMrkvArrays_recessionTaxCut_c = CondMrkvArrays_recession_c
    MrkvArray_recessionTaxCut_d = makeFullMrkvArray(MacroMrkvArray_recessionTaxCut, CondMrkvArrays_recessionTaxCut_d)
    MrkvArray_recessionTaxCut_h = makeFullMrkvArray(MacroMrkvArray_recessionTaxCut, CondMrkvArrays_recessionTaxCut_h)
    MrkvArray_recessionTaxCut_c = makeFullMrkvArray(MacroMrkvArray_recessionTaxCut, CondMrkvArrays_recessionTaxCut_c)
    
    MacroMrkvArray_recessionUI = MacroMrkvArray_recession
    CondMrkvArrays_recessionUI_d = makeCondMrkvArrays_recessionUI(Urate_normal_d, Uspell_normal, UBspell_normal, Urate_recession_d, Uspell_recession, num_experiment_periods, UBspell_extended-UBspell_normal)
    CondMrkvArrays_recessionUI_h = makeCondMrkvArrays_recessionUI(Urate_normal_h, Uspell_normal, UBspell_normal, Urate_recession_h, Uspell_recession, num_experiment_periods, UBspell_extended-UBspell_normal)
    CondMrkvArrays_recessionUI_c = makeCondMrkvArrays_recessionUI(Urate_normal_c, Uspell_normal, UBspell_normal, Urate_recession_c, Uspell_recession, num_experiment_periods, UBspell_extended-UBspell_normal)
    MrkvArray_recessionUI_d    = makeFullMrkvArray(MacroMrkvArray_recessionUI, CondMrkvArrays_recessionUI_d)
    MrkvArray_recessionUI_h    = makeFullMrkvArray(MacroMrkvArray_recessionUI, CondMrkvArrays_recessionUI_h)
    MrkvArray_recessionUI_c    = makeFullMrkvArray(MacroMrkvArray_recessionUI, CondMrkvArrays_recessionUI_c)
    
    
    
  


    # find intial distribution of states for each education type
    vals_d, vecs_d = np.linalg.eig(np.transpose(MrkvArray_base_d[0])) 
    dist_d = np.abs(np.abs(vals_d) - 1.)
    idx_d = np.argmin(dist_d)
    init_mrkv_dist_d = vecs_d[:,idx_d].real/np.sum(vecs_d[:,idx_d].real)
    
    vals_h, vecs_h = np.linalg.eig(np.transpose(MrkvArray_base_h[0])) 
    dist_h = np.abs(np.abs(vals_h) - 1.)
    idx_h = np.argmin(dist_h)
    init_mrkv_dist_h = vecs_h[:,idx_h].real/np.sum(vecs_h[:,idx_h].real)
    
    vals_c, vecs_c = np.linalg.eig(np.transpose(MrkvArray_base_c[0])) 
    dist_c = np.abs(np.abs(vals_c) - 1.)
    idx_c = np.argmin(dist_c)
    init_mrkv_dist_c = vecs_c[:,idx_c].real/np.sum(vecs_c[:,idx_c].real)
    
    
    
    # Define a parameter dictionary for dropout type
    init_dropout = {"cycles": 0, # 00This will be overwritten at type construction
                    "T_cycle": T_cycle,
                    'T_sim': 400, #Simhulate up to age 400
                    'T_age': T_age,
                    'AgentCount': 1000, #number overwritten later
                    "PermGroFacAgg": PermGroFacAgg,
                    "PopGroFac": PopGroFac,
                    "CRRA": CRRA,
                    "DiscFac": 0.98, # This will be overwritten at type construction
                    "Rfree_base" : Rfree_base,
                    "PermGroFac_base": PermGroFac_base_d,
                    "LivPrb_base": LivPrb_base,
                    "MrkvArray_recession" : MrkvArray_recession_d,
                    "MacroMrkvArray_recession" : MacroMrkvArray_recession,
                    "CondMrkvArrays_recession" : CondMrkvArrays_recession_d,
                    "MrkvArray_recessionUI" : MrkvArray_recessionUI_d,
                    "MacroMrkvArray_recessionUI" : MacroMrkvArray_recessionUI,
                    "CondMrkvArrays_recessionUI" : CondMrkvArrays_recessionUI_d,
                    "MrkvArray_recessionTaxCut" : MrkvArray_recessionTaxCut_d,
                    "MacroMrkvArray_recessionTaxCut" : MacroMrkvArray_recessionTaxCut,
                    "CondMrkvArrays_recessionTaxCut" : CondMrkvArrays_recessionTaxCut_d,
                    "MrkvArray_recessionCheck" : MrkvArray_recessionCheck_d,
                    "MacroMrkvArray_recessionCheck" : MacroMrkvArray_recessionCheck,
                    "CondMrkvArrays_recessionCheck" : CondMrkvArrays_recessionCheck_d,
                    "Rfree" : np.array(num_base_MrkvStates*Rfree_base),
                    "PermGroFac": [np.array(PermGroFac_base_d*num_base_MrkvStates)],
                    "LivPrb": [np.array(LivPrb_base*num_base_MrkvStates)],
                    "MrkvArray_base" : MrkvArray_base_d, 
                    "MacroMrkvArray_base" : MacroMrkvArray_base,
                    "CondMrkvArrays_base" : CondMrkvArrays_base_d,
                    "MrkvArray" : MrkvArray_base_d, 
                    "MacroMrkvArray" : MacroMrkvArray_base,
                    "CondMrkvArrays" : CondMrkvArrays_base_d,
                    "BoroCnstArt": 0.0,
                    "PermShkStd": PermShkStd,
                    "PermShkCount": PermShkCount,
                    "TranShkStd": TranShkStd,
                    "TranShkCount": TranShkCount,
                    "UnempPrb": 0.0, # Unemployment is modeled as a Markov state
                    "IncUnemp": IncUnemp,
                    "IncUnempNoBenefits": IncUnempNoBenefits,
                    "aXtraMin": aXtraMin,
                    "aXtraMax": aXtraMax,
                    "aXtraCount": aXtraCount,
                    "aXtraExtra": aXtraExtra,
                    "aXtraNestFac": aXtraNestFac,
                    "CubicBool": False,
                    "vFuncBool": False,
                    'aNrmInitMean': np.log(0.00001), # Initial assets are zero
                    'aNrmInitStd': 0.0,
                    'pLvlInitMean': pLvlInitMean_d,
                    'pLvlInitStd': pLvlInitStd_d,
                    "MrkvPrbsInit" : np.array(list(init_mrkv_dist_d)),
                    'Urate_normal' : Urate_normal_d,
                    'Uspell_normal' : Uspell_normal,
                    'UBspell_normal' : UBspell_normal,
                    'num_base_MrkvStates' : num_base_MrkvStates,
                    'Urate_recession' : Urate_recession_d,
                    'Uspell_recession' : Uspell_recession,
                    'num_experiment_periods' : num_experiment_periods,
                    'Rspell' : Rspell,
                    'R_shared' : R_shared,
                    'UBspell_extended' : UBspell_extended,
                    'PolicyUBspell' : PolicyUBspell,
                    'PolicyTaxCutspell' : PolicyTaxCutspell,
                    'TaxCutIncFactor' : TaxCutIncFactor,
                    'TaxCutPeriods' : TaxCutPeriods,
                    'TaxCutContinuationProb_Rec' : TaxCutContinuationProb_Rec,
                    'TaxCutContinuationProb_Bas' : TaxCutContinuationProb_Bas,
                    'CheckStimLvl' : CheckStimLvl,
                    'CheckStimLvl_PLvl_Cutoff_start' : CheckStimLvl_PLvl_Cutoff_start,
                    'CheckStimLvl_PLvl_Cutoff_end' : CheckStimLvl_PLvl_Cutoff_end,
                    'UpdatePrb' : 1.0,
                    'Splurge' : Splurge,
                    'track_vars' : [],
                    'EducType': 0
                    }
    
    adj_highschool = {
        "PermGroFac_base": PermGroFac_base_h,
        "PermGroFac": [np.array(PermGroFac_base_h*num_base_MrkvStates)],
        "MrkvArray_base" : MrkvArray_base_h, 
        "CondMrkvArrays_base" : CondMrkvArrays_base_h,
        "MrkvArray" : MrkvArray_base_h, 
        "CondMrkvArrays" : CondMrkvArrays_base_h,
        "MrkvArray_recession" : MrkvArray_recession_h,
        "CondMrkvArrays_recession" : CondMrkvArrays_recession_h,
        "MrkvArray_recessionUI" : MrkvArray_recessionUI_h,
        "CondMrkvArrays_recessionUI" : CondMrkvArrays_recessionUI_h,
        "MrkvArray_recessionTaxCut" : MrkvArray_recessionTaxCut_h,
        "CondMrkvArrays_recessionTaxCut" : CondMrkvArrays_recessionTaxCut_h,
        "MrkvArray_recessionCheck" : MrkvArray_recessionCheck_h,
        "CondMrkvArrays_recessionCheck" : CondMrkvArrays_recessionCheck_h,  
        'pLvlInitMean': pLvlInitMean_h,
        'pLvlInitStd': pLvlInitStd_h,
        "MrkvPrbsInit" : np.array(list(init_mrkv_dist_h)),
        'Urate_normal' : Urate_normal_h,
        'Urate_recession' : Urate_recession_h,
        'EducType' : 1}
    init_highschool = init_dropout.copy()
    init_highschool.update(adj_highschool)
    
    adj_college = {
        "PermGroFac_base": PermGroFac_base_c,
        "PermGroFac": [np.array(PermGroFac_base_c*num_base_MrkvStates)],
        "MrkvArray_base" : MrkvArray_base_c, 
        "CondMrkvArrays_base" : CondMrkvArrays_base_c,
        "MrkvArray" : MrkvArray_base_c, 
        "CondMrkvArrays" : CondMrkvArrays_base_c,
        "MrkvArray_recession" : MrkvArray_recession_c,
        "CondMrkvArrays_recession" : CondMrkvArrays_recession_c,
        "MrkvArray_recessionUI" : MrkvArray_recessionUI_c,
        "CondMrkvArrays_recessionUI" : CondMrkvArrays_recessionUI_c,
        "MrkvArray_recessionTaxCut" : MrkvArray_recessionTaxCut_c,
        "CondMrkvArrays_recessionTaxCut" : CondMrkvArrays_recessionTaxCut_c,
        "MrkvArray_recessionCheck" : MrkvArray_recessionCheck_c,
        "CondMrkvArrays_recessionCheck" : CondMrkvArrays_recessionCheck_c, 
        'pLvlInitMean': pLvlInitMean_c,
        'pLvlInitStd': pLvlInitStd_c,
        "MrkvPrbsInit" : np.array(list(init_mrkv_dist_c)),
        'Urate_normal' : Urate_normal_c,
        'Urate_recession' : Urate_recession_c,
        'EducType' : 2}
    init_college = init_dropout.copy()
    init_college.update(adj_college)
    
        
    # Population share of each type (at present only one type)    
    # TypeShares = [1.0]
    
    # Define a dictionary to represent the baseline scenario
    base_dict = {'shock_type' : "base",
                 'UpdatePrb' : 1.0,
                 'Splurge' : Splurge
                 }
    # Define a dictionary to mutate baseline for the recession
    recession_changes = {
                 'shock_type' : "recession",
                 }
    UI_changes = {
                 'shock_type' : "UI",
                 }
    recession_UI_changes = {
                 'shock_type' : "recessionUI",
                 }
    TaxCut_changes = {
                 'shock_type' : "TaxCut",
                 }
    recession_TaxCut_changes = {
                 'shock_type' : "recessionTaxCut",
                 }
    Check_changes = {
                 'shock_type' : "Check",
                 }
    recession_Check_changes = {
                 'shock_type' : "recessionCheck",
                 }
    # sticky_e_changes = {
    #              'UpdatePrb' : UpdatePrb
    #              }
    # frictionless_changes = {
    #              'UpdatePrb' : 1.0
    #              }
    
    
        
    # Parameters for AggregateDemandEconomy economy
    intercept_prev = np.ones((num_base_MrkvStates,num_base_MrkvStates ))    # Intercept of aggregate savings function
    slope_prev = np.zeros((num_base_MrkvStates,num_base_MrkvStates ))       # Slope of aggregate savings function
                                                    # Elasticity of productivity to consumption
    
    num_max_iterations_solvingAD = 15
    convergence_tol_solvingAD = 1E-4
    Cfunc_iter_stepsize       = 1
    
    # Make a dictionary to specify a Cobb-Douglas economy
    init_ADEconomy = {'intercept_prev': intercept_prev,
                         'slope_prev': slope_prev,
                         'ADelasticity' : 0.0,
                         'demand_ADelasticity' : ADelasticity,
                         'Cfunc_iter_stepsize' : Cfunc_iter_stepsize,
                         'MrkvArray' : MrkvArray_base_h,
                         'MrkvArray_recession' : MrkvArray_recession_h,
                         'MrkvArray_recessionUI' : MrkvArray_recessionUI_h,
                         'MrkvArray_recessionTaxCut' : MrkvArray_recessionTaxCut_h,
                         'MrkvArray_recessionCheck' : MrkvArray_recessionCheck_h,
                         'num_base_MrkvStates' : num_base_MrkvStates,
                         'num_experiment_periods' : num_experiment_periods,
                         "MrkvArray_base" : MrkvArray_base_h, 
                         'CgridBase' : CgridBase,
                         'EconomyMrkvNow_init': 0,
                         'act_T' : 400,
                         'TaxCutContinuationProb_Rec' : TaxCutContinuationProb_Rec,
                         'TaxCutContinuationProb_Bas' : TaxCutContinuationProb_Bas
                         }
    
    if OutputFor=='_Main.py':
    
        return [init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
                 DiscFacCount, AgentCountTotal, base_dict, num_max_iterations_solvingAD,\
                 convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates, \
                 data_EducShares, max_recession_duration, num_experiment_periods,\
                 recession_changes, UI_changes, recession_UI_changes,\
                 TaxCut_changes, recession_TaxCut_changes, Check_changes, recession_Check_changes]
            
    elif OutputFor=='_Model.py':

        return [makeMacroMrkvArray_recession, makeCondMrkvArrays_recession, makeFullMrkvArray, T_sim, \
                 makeCondMrkvArrays_base, makeCondMrkvArrays_recessionUI]            
            
    elif OutputFor=='_Output_Results.py':

        return [max_recession_duration, Rspell, Rfree_base] 