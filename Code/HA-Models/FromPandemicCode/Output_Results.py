import os
from Welfare import Welfare_Results
from HARK.utilities import make_figs

def Output_Results(saved_results_dir,fig_dir,table_dir,Parametrization='Baseline'):

    # Make folders for output   
    try:
        os.mkdir(fig_dir)
    except OSError:
        print ("Creation of the directory %s failed" % fig_dir)
    else:
        print ("Successfully created the directores %s " % fig_dir)

    try:
        os.mkdir(table_dir)
    except OSError:
        print ("Creation of the directory %s failed" % table_dir)
    else:
        print ("Successfully created the directory %s " % table_dir)

    
    #3. add welfare function to this
    #4. think about way how to create overall robustness table
    
    
    from Parameters import returnParameters
    import numpy as np
    import matplotlib.pyplot as plt
    from OtherFunctions import getSimulationDiff, getSimulationPercentDiff, getNPVMultiplier, loadPickle
    mystr = lambda x : '{:.2f}'.format(x)
    
    
    [max_recession_duration, Rspell, Rfree_base]  = returnParameters(Parametrization=Parametrization,OutputFor='_Output_Results.py')
    
    
    Plot_1stRoundAd         = False
            
    max_T = 16
    x_axis = np.arange(1,max_T+1)
    
    folder_AD           = saved_results_dir 
    folder_base         = saved_results_dir
    folder_noAD         = saved_results_dir
    folder_firstroundAD = saved_results_dir
    
    
    base_results                            = loadPickle('base_results',folder_base,locals())
    
    recession_results                       = loadPickle('recession_results',folder_noAD,locals())
    recession_results_AD                    = loadPickle('recession_results_AD',folder_AD,locals())
    recession_results_firstRoundAD          = loadPickle('recession_results_firstRoundAD',folder_firstroundAD,locals())
    
    recession_UI_results                    = loadPickle('recessionUI_results',folder_noAD,locals())       
    recession_UI_results_AD                 = loadPickle('recessionUI_results_AD',folder_AD,locals())
    recession_UI_results_firstRoundAD       = loadPickle('recessionUI_results_firstRoundAD',folder_firstroundAD,locals())
    
    recession_Check_results                 = loadPickle('recessionCheck_results',folder_noAD,locals())       
    recession_Check_results_AD              = loadPickle('recessionCheck_results_AD',folder_AD,locals())
    recession_Check_results_firstRoundAD    = loadPickle('recessionCheck_results_firstRoundAD',folder_firstroundAD,locals())
    
    recession_TaxCut_results                = loadPickle('recessionTaxCut_results',folder_noAD,locals())
    recession_TaxCut_results_AD             = loadPickle('recessionTaxCut_results_AD',folder_AD,locals())
    recession_TaxCut_results_firstRoundAD   = loadPickle('recessionTaxCut_results_firstRoundAD',folder_firstroundAD,locals())
    
    if type(recession_results_firstRoundAD) == int:
        Mltp_1stRoundAd         = False
    else:
        Mltp_1stRoundAd         = True
          
    
    #%% IRFs for income and consumption for three policies
    # Tax cut        
    
    
    AddCons_Rec_TaxCut_RelRec               = getSimulationPercentDiff(recession_results,               recession_TaxCut_results,'AggCons')
    AddCons_Rec_TaxCut_AD_RelRec            = getSimulationPercentDiff(recession_results_AD,            recession_TaxCut_results_AD,'AggCons')
    
    AddInc_Rec_TaxCut_RelRec                = getSimulationPercentDiff(recession_results,               recession_TaxCut_results,'AggIncome')
    AddInc_Rec_TaxCut_AD_RelRec             = getSimulationPercentDiff(recession_results_AD,            recession_TaxCut_results_AD,'AggIncome')
    
    if Plot_1stRoundAd:
        AddCons_Rec_TaxCut_firstRoundAD_RelRec  = getSimulationPercentDiff(recession_results_firstRoundAD,  recession_TaxCut_results_firstRoundAD,'AggCons')
        AddInc_Rec_TaxCut_firstRoundAD_RelRec  = getSimulationPercentDiff(recession_results_firstRoundAD,   recession_TaxCut_results_firstRoundAD,'AggIncome')
       
    
    plt.figure()
    #plt.title('Recession + tax cut', size=30)
    plt.plot(x_axis,AddInc_Rec_TaxCut_RelRec[0:max_T],              color='blue',linestyle='-')
    plt.plot(x_axis,AddInc_Rec_TaxCut_AD_RelRec[0:max_T],           color='blue',linestyle='--')
    if Plot_1stRoundAd:
        plt.plot(x_axis,AddInc_Rec_TaxCut_firstRoundAD_RelRec[0:max_T], color='blue',linestyle=':')
    plt.plot(x_axis,AddCons_Rec_TaxCut_RelRec[0:max_T],             color='red',linestyle='-')
    plt.plot(x_axis,AddCons_Rec_TaxCut_AD_RelRec[0:max_T],          color='red',linestyle='--') 
    
    plt.legend(['Income','Income (AD effects)', \
                'Consumption','Consumption (AD effects)'],loc='best')
    
    if Plot_1stRoundAd:
        plt.plot(x_axis,AddCons_Rec_TaxCut_firstRoundAD_RelRec[0:max_T],color='red',linestyle=':')
        plt.legend(['Income','Income (AD effects)','Inc, 1st round AD effects', \
                    'Consumption','Consumption (AD effects)','Cons, 1st round AD effects'])
            
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('quarter')
    plt.ylabel('% difference relative to recession')
    #plt.savefig(fig_dir +'recession_taxcut_relrecession.pdf')
    make_figs('recession_taxcut_relrecession', True , False, target_dir=fig_dir)
    plt.show()   
    
    
    
    #UI extension
    AddCons_UI_Ext_Rec_RelRec               = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggCons')
    AddInc_UI_Ext_Rec_RelRec                = getSimulationPercentDiff(recession_results,    recession_UI_results,'AggIncome')
    
    AddCons_UI_Ext_Rec_RelRec_AD            = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggCons')
    AddInc_UI_Ext_Rec_RelRec_AD             = getSimulationPercentDiff(recession_results_AD,    recession_UI_results_AD,'AggIncome')
     
    if Plot_1stRoundAd:
        AddCons_UI_Ext_Rec_RelRec_firstRoundAD  = getSimulationPercentDiff(recession_results_firstRoundAD,    recession_UI_results_firstRoundAD,'AggCons')
        AddInc_UI_Ext_Rec_RelRec_firstRoundAD   = getSimulationPercentDiff(recession_results_firstRoundAD,    recession_UI_results_firstRoundAD,'AggIncome')       
    
    plt.figure()
    #plt.title('Recession + UI extension', size=30)
    plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec[0:max_T],              color='blue',linestyle='-')
    plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_AD[0:max_T],           color='blue',linestyle='--')
    if Plot_1stRoundAd:
        plt.plot(x_axis,AddInc_UI_Ext_Rec_RelRec_firstRoundAD[0:max_T], color='blue',linestyle=':')
    plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec[0:max_T],             color='red',linestyle='-')
    plt.plot(x_axis,AddCons_UI_Ext_Rec_RelRec_AD[0:max_T],          color='red',linestyle='--') 
    
    plt.legend(['Income','Income (AD effects)', \
                'Consumption','Consumption (AD effects)'],loc='best')
    
    if Plot_1stRoundAd:
        plt.plot(x_axis,AddCons_Rec_TaxCut_firstRoundAD_RelRec[0:max_T],color='red',linestyle=':')
        plt.legend(['Income','Income (AD effects)','Inc, 1st round AD effects', \
                    'Consumption','Consumption (AD effects)','Cons, 1st round AD effects'])
            
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('quarter')
    plt.ylabel('% difference relative to recession')
    #plt.savefig(fig_dir +'recession_UI_relrecession.pdf')
    make_figs('recession_UI_relrecession', True , False, target_dir=fig_dir)
    plt.show() 
    
    
    #Check stimulus    
    AddCons_Rec_Check_RelRec               = getSimulationPercentDiff(recession_results,               recession_Check_results,'AggCons')
    AddInc_Rec_Check_RelRec                = getSimulationPercentDiff(recession_results,               recession_Check_results,'AggIncome')
    
    AddCons_Rec_Check_AD_RelRec            = getSimulationPercentDiff(recession_results_AD,            recession_Check_results_AD,'AggCons')
    AddInc_Rec_Check_AD_RelRec             = getSimulationPercentDiff(recession_results_AD,            recession_Check_results_AD,'AggIncome')
    
    if Plot_1stRoundAd:
        AddCons_Rec_Check_firstRoundAD_RelRec  = getSimulationPercentDiff(recession_results_firstRoundAD,  recession_Check_results_firstRoundAD,'AggCons')
        AddInc_Rec_Check_firstRoundAD_RelRec   = getSimulationPercentDiff(recession_results_firstRoundAD,  recession_Check_results_firstRoundAD,'AggIncome')
    
    
    plt.figure()
    #plt.title('Recession + Check', size=30)
    plt.plot(x_axis,AddInc_Rec_Check_RelRec[0:max_T],              color='blue',linestyle='-')
    plt.plot(x_axis,AddInc_Rec_Check_AD_RelRec[0:max_T],           color='blue',linestyle='--')
    if Plot_1stRoundAd:
        plt.plot(x_axis,AddInc_Rec_Check_firstRoundAD_RelRec[0:max_T], color='blue',linestyle=':')
    plt.plot(x_axis,AddCons_Rec_Check_RelRec[0:max_T],             color='red',linestyle='-')
    plt.plot(x_axis,AddCons_Rec_Check_AD_RelRec[0:max_T],          color='red',linestyle='--') 
    
    plt.legend(['Income','Income (AD effects)', \
                'Consumption','Consumption (AD effects)'],loc='best')
    
    if Plot_1stRoundAd:
        plt.plot(x_axis,AddCons_Rec_TaxCut_firstRoundAD_RelRec[0:max_T],color='red',linestyle=':')
        plt.legend(['Income','Income (AD effects)','Inc, 1st round AD effects', \
                    'Consumption','Consumption (AD effects)','Cons, 1st round AD effects'])
            
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('quarter')
    plt.ylabel('% difference relative to recession')
    #plt.savefig(fig_dir +'recession_Check_relrecession.pdf')
    make_figs('recession_Check_relrecession', True , False, target_dir=fig_dir)
    plt.show()        
    
    
    #########################################################################
    #########################################################################
    #########################################################################
       
    
    
    
    
    
    
    
    #%% Multipliers
    
    
    
    NPV_AddInc_UI_Rec                       = getSimulationDiff(recession_results,recession_UI_results,'NPV_AggIncome') # Policy expenditure
    NPV_Multiplier_UI_Rec                   = getNPVMultiplier(recession_results,               recession_UI_results,               NPV_AddInc_UI_Rec)
    NPV_Multiplier_UI_Rec_AD                = getNPVMultiplier(recession_results_AD,            recession_UI_results_AD,            NPV_AddInc_UI_Rec)
    if Mltp_1stRoundAd:
        NPV_Multiplier_UI_Rec_firstRoundAD  = getNPVMultiplier(recession_results_firstRoundAD,  recession_UI_results_firstRoundAD,  NPV_AddInc_UI_Rec)
    else:
        NPV_Multiplier_UI_Rec_firstRoundAD = np.zeros_like(NPV_Multiplier_UI_Rec)
    
    
    NPV_AddInc_Rec_TaxCut                   = getSimulationDiff(recession_results,recession_TaxCut_results,'NPV_AggIncome')
    NPV_Multiplier_Rec_TaxCut               = getNPVMultiplier(recession_results,               recession_TaxCut_results,               NPV_AddInc_Rec_TaxCut)
    NPV_Multiplier_Rec_TaxCut_AD            = getNPVMultiplier(recession_results_AD,            recession_TaxCut_results_AD,            NPV_AddInc_Rec_TaxCut)
    if Mltp_1stRoundAd:
        NPV_Multiplier_Rec_TaxCut_firstRoundAD  = getNPVMultiplier(recession_results_firstRoundAD,  recession_TaxCut_results_firstRoundAD,  NPV_AddInc_Rec_TaxCut)
    else:
        NPV_Multiplier_Rec_TaxCut_firstRoundAD = np.zeros_like(NPV_Multiplier_Rec_TaxCut)
        
    NPV_AddInc_Rec_Check                    = getSimulationDiff(recession_results,recession_Check_results,'NPV_AggIncome') 
    NPV_Multiplier_Rec_Check                = getNPVMultiplier(recession_results,               recession_Check_results,               NPV_AddInc_Rec_Check)
    NPV_Multiplier_Rec_Check_AD             = getNPVMultiplier(recession_results_AD,            recession_Check_results_AD,            NPV_AddInc_Rec_Check)
    if Mltp_1stRoundAd:
        NPV_Multiplier_Rec_Check_firstRoundAD   = getNPVMultiplier(recession_results_firstRoundAD,  recession_Check_results_firstRoundAD,  NPV_AddInc_Rec_Check)
    else:
        NPV_Multiplier_Rec_Check_firstRoundAD = np.zeros_like(NPV_Multiplier_Rec_Check)
    
            
    #print('NPV Multiplier UI recession no AD: \t\t',mystr(NPV_Multiplier_UI_Rec[-1]))
    print('NPV Multiplier UI recession with AD: \t\t',mystr(NPV_Multiplier_UI_Rec_AD[-1]))
    print('NPV Multiplier UI recession 1st round AD: \t',mystr(NPV_Multiplier_UI_Rec_firstRoundAD[-1]))
    print('')
    
    #print('NPV Multiplier tax cut recession no AD: \t',mystr(NPV_Multiplier_Rec_TaxCut[-1]))
    print('NPV Multiplier tax cut recession with AD: \t',mystr(NPV_Multiplier_Rec_TaxCut_AD[-1]))
    print('NPV Multiplier tax cut recession 1st round AD:  ',mystr(NPV_Multiplier_Rec_TaxCut_firstRoundAD[-1]))
    print('')
    
    #print('NPV Multiplier check recession no AD: \t\t',mystr(NPV_Multiplier_Rec_Check[-1]))
    print('NPV Multiplier check recession with AD: \t',mystr(NPV_Multiplier_Rec_Check_AD[-1]))
    print('NPV Multiplier check recession 1st round AD: \t',mystr(NPV_Multiplier_Rec_Check_firstRoundAD[-1]))
    print('')
    # Multipliers in non-AD are less than 1 -> this is because of deaths!
    
    
    
    
    # Multiplier plots for AD case
    max_T2 = 15
    nPlotDiff = 2
    
    #Cumulative
    C_Multiplier_UI_Rec_AD                = getNPVMultiplier(recession_results_AD,            recession_UI_results_AD,            NPV_AddInc_UI_Rec[-1])
    C_Multiplier_Rec_TaxCut_AD            = getNPVMultiplier(recession_results_AD,            recession_TaxCut_results_AD,        NPV_AddInc_Rec_TaxCut[-1])
    C_Multiplier_Rec_Check_AD             = getNPVMultiplier(recession_results_AD,            recession_Check_results_AD,         NPV_AddInc_Rec_Check[-1])
    x_axis = np.arange(1,max_T2+1)[::nPlotDiff]
    plt.plot(x_axis,C_Multiplier_Rec_TaxCut_AD[0:max_T2][::nPlotDiff],              color='red',linestyle='-')
    plt.plot(x_axis,C_Multiplier_UI_Rec_AD[0:max_T2][::nPlotDiff],                  color='blue',linestyle='-')
    plt.plot(x_axis,C_Multiplier_Rec_Check_AD[0:max_T2][::nPlotDiff],               color='green',linestyle='-')
    plt.legend(['Payroll Tax cut','Extended Unemployment Benefits','Stimulus Check'])
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, nPlotDiff))
    plt.xlabel('quarter')
    #plt.savefig(fig_dir +'Cummulative_multipliers.pdf')
    make_figs('Cummulative_multipliers', True , False, target_dir=fig_dir)
    plt.show()
        
    
    # #Period multiplier
    # AddInc_UI_Rec       = getSimulationDiff(recession_results,recession_UI_results,'AggIncome')
    # AddInc_Rec_TaxCut   = getSimulationDiff(recession_results,recession_TaxCut_results,'AggIncome')
    # AddInc_Rec_Check    = getSimulationDiff(recession_results,recession_Check_results,'AggIncome')
    # PM_UI_Rec = 1/100*getStimulus(recession_results_AD, recession_UI_results_AD, AddInc_UI_Rec)
    # PM_TaxCut_Rec = 1/100*getStimulus(recession_results_AD, recession_TaxCut_results_AD, AddInc_Rec_TaxCut)
    # PM_Check_Rec = 1/100*getStimulus(recession_results_AD, recession_Check_results_AD, AddInc_Rec_Check)
    # # values of inf nonsensical
    # PM_UI_Rec[PM_UI_Rec>1000] = 0
    # PM_TaxCut_Rec[PM_TaxCut_Rec>1000] = 0
    # PM_Check_Rec[PM_Check_Rec>1000] = 0
    # x_axis = np.arange(1,max_T2+1)[::nPlotDiff]
    # #plt.title('Period multipliers with AD effects', size=30)
    # plt.plot(x_axis,PM_TaxCut_Rec[0:max_T2][::nPlotDiff],              color='red',linestyle='-')
    # plt.plot(x_axis,PM_UI_Rec[0:max_T2][::nPlotDiff],                  color='blue',linestyle='-')
    # plt.plot(x_axis,PM_Check_Rec[0:max_T2][::nPlotDiff],               color='green',linestyle='-')
    # plt.legend(['Payroll tax cut','UI extension','Check'])
    # plt.xticks(np.arange(min(x_axis), max(x_axis)+1, nPlotDiff))
    # plt.xlabel('quarter')
    # plt.savefig(saved_results_dir +'P_multipliers.pdf')
    # plt.show()     
    
    # # NPV multiplier
    # x_axis = np.arange(1,max_T2+1)[::nPlotDiff]
    # #plt.title('NPV multipliers at different horizons with AD effects', size=30)
    # plt.plot(x_axis,NPV_Multiplier_Rec_TaxCut_AD[0:max_T2][::nPlotDiff],              color='red',linestyle='-')
    # plt.plot(x_axis,NPV_Multiplier_UI_Rec_AD[0:max_T2][::nPlotDiff],                  color='blue',linestyle='-')
    # plt.plot(x_axis,NPV_Multiplier_Rec_Check_AD[0:max_T2][::nPlotDiff],               color='green',linestyle='-')
    # plt.legend(['Payroll tax cut','UI extension','Check'])
    # plt.xticks(np.arange(min(x_axis), max(x_axis)+1, nPlotDiff))
    # plt.xlabel('quarter')
    # plt.savefig(saved_results_dir +'NPV_multipliers.pdf')
    # plt.show()
    
    
    
    # Share of policy expenditure during recession
    R_persist = 1.-1./Rspell        
    recession_prob_array = np.array([R_persist**t*(1-R_persist) for t in range(max_recession_duration)])
    recession_prob_array[-1] = 1.0 - np.sum(recession_prob_array[:-1])
         
    def ShareOfPolicyDuringRec(rec,TaxCut,UI,Check,recession_prob_array,max_T):  
        # considers runs different recession lengths and calculates expenditure share within those runs
        # then sums it up weighing by probability of that recession length
        ShareExpDuringRecession= np.zeros(3)
             
        for i in range(21):      
            NPV_TaxCut              = getSimulationDiff(rec[i],TaxCut[i],'NPV_AggIncome') 
            ShareExpDuringRecession[0] += NPV_TaxCut[i]/NPV_TaxCut[-1]*recession_prob_array[i]
            
            NPV_UI                  = getSimulationDiff(rec[i],UI[i],'NPV_AggIncome') 
            ShareExpDuringRecession[1] += NPV_UI[i]/NPV_UI[-1]*recession_prob_array[i]
            
            NPV_Check               = getSimulationDiff(rec[i],Check[i],'NPV_AggIncome') 
            ShareExpDuringRecession[2] += NPV_Check[i]/NPV_Check[-1]*recession_prob_array[i]
             
        return 100*ShareExpDuringRecession
        
             
    recession_all_results        = loadPickle('recession_all_results',folder_AD,locals())   
    recession_all_results_UI     = loadPickle('recessionUI_all_results',folder_AD,locals())
    recession_all_results_TaxCut = loadPickle('recessionTaxCut_all_results',folder_AD,locals())
    recession_all_results_Check  = loadPickle('recessionCheck_all_results',folder_AD,locals())
        
    [Share_TaxCut,Share_UI,ShareCheck]=ShareOfPolicyDuringRec(recession_all_results,recession_all_results_TaxCut,\
                           recession_all_results_UI,recession_all_results_Check,\
                           recession_prob_array,max_recession_duration)
    
    print('Share of Tax cut policy expenditure occuring during recession: ', mystr(Share_TaxCut)    )
    print('Share of UI policy expenditure occuring during recession: ', mystr(Share_UI) ) 
    print('Share of Check policy expenditure occuring during recession: ', mystr(ShareCheck) ) 
    
    
    def mystr3(number):
        if not np.isnan(number):
            out = "{:.3f}".format(number)
        else:
            out = ''
        return out
    
    def mystr1(number):
        if not np.isnan(number):
            out = "{:.1f}".format(number)
        else:
            out = ''
        return out
        
    output  ="\\begin{tabular}{@{}lccc@{}} \n"
    output +="\\toprule \n"
    output +="& Tax Cut    & UI extension    & Stimulus check    \\\\  \\midrule \n"
    output +="Long-run Multiplier (AD effect) &"                 + mystr3(NPV_Multiplier_Rec_TaxCut_AD[-1])             + "  & "+ mystr3(NPV_Multiplier_UI_Rec_AD[-1])               +  "  & "+  mystr3(NPV_Multiplier_Rec_Check_AD[-1])  + "     \\\\ \n"
    output +="Long-run Multiplier (1st round AD effect only) &"  + mystr3(NPV_Multiplier_Rec_TaxCut_firstRoundAD[-1])   + "  & "+ mystr3(NPV_Multiplier_UI_Rec_firstRoundAD[-1])     +  "  & "+  mystr3(NPV_Multiplier_Rec_Check_firstRoundAD[-1])  + "     \\\\ \n"
    output +="Share of policy expenditure during recession &" + mystr1(Share_TaxCut)   + "\%  & "+ mystr1(Share_UI)  +  "\%  & "+  mystr1(ShareCheck)  + " \%    \\\\ \n"
    output +="\\end{tabular}  \n"
    
    with open(table_dir + 'Multiplier.tex','w') as f:
        f.write(output)
        f.close()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #%% Function that returns information on a policy with specific RecLength and PolicyLength
    def PlotsforSpecificRecLength(RecLength,Policy): 
    
        # Policy options 'recession_UI' / 'recession_TaxCut' / 'recession_Check'
        
        recession_all_results               = loadPickle('recession_all_results',folder_noAD,locals())
        recession_all_results_AD            = loadPickle('recession_all_results_AD',folder_AD,locals())
        if Mltp_1stRoundAd:
            recession_all_results_firstRoundAD  = loadPickle('recession_all_results_firstRoundAD',folder_firstroundAD,locals())
        
        recession_all_policy_results        = loadPickle( Policy + '_all_results',folder_noAD,locals())       
        recession_all_policy_results_AD     = loadPickle(Policy + '_all_results_AD',folder_AD,locals())
        if Mltp_1stRoundAd:
            recession_all_policy_results_firstRoundAD= loadPickle(Policy + '_all_results_firstRoundAD',folder_firstroundAD,locals())
        
        
        NPV_AddInc                  = getSimulationDiff(recession_all_results[RecLength-1],recession_all_policy_results[RecLength-1],'NPV_AggIncome') # Policy expenditure
        NPV_Multiplier              = getNPVMultiplier(recession_all_results[RecLength-1],               recession_all_policy_results[RecLength-1],               NPV_AddInc)
        NPV_Multiplier_AD           = getNPVMultiplier(recession_all_results_AD[RecLength-1],            recession_all_policy_results_AD[RecLength-1],            NPV_AddInc)
        if Mltp_1stRoundAd:
            NPV_Multiplier_firstRoundAD = getNPVMultiplier(recession_all_results_firstRoundAD[RecLength-1],  recession_all_policy_results_firstRoundAD[RecLength-1],  NPV_AddInc)
        else:
            NPV_Multiplier_firstRoundAD = np.zeros_like(NPV_Multiplier_AD)
         
        Multipliers = [NPV_Multiplier,NPV_Multiplier_AD,NPV_Multiplier_firstRoundAD]
        
        PlotEach = False
        
        if PlotEach:
        
            AddCons_RelRec               = getSimulationPercentDiff(recession_all_results[RecLength-1],    recession_all_policy_results[RecLength-1],'AggCons')
            AddInc_RelRec                = getSimulationPercentDiff(recession_all_results[RecLength-1],    recession_all_policy_results[RecLength-1],'AggIncome')
            
            AddCons_RelRec_AD            = getSimulationPercentDiff(recession_all_results_AD[RecLength-1],    recession_all_policy_results_AD[RecLength-1],'AggCons')
            AddInc_RelRec_AD             = getSimulationPercentDiff(recession_all_results_AD[RecLength-1],    recession_all_policy_results_AD[RecLength-1],'AggIncome')
            
           
            plt.figure(figsize=(15,10))
            plt.title('Recession lasts ' + str(RecLength) + 'q', size=30)
            plt.plot(x_axis,AddInc_RelRec[0:max_T],              color='blue',linestyle='-')
            plt.plot(x_axis,AddInc_RelRec_AD[0:max_T],           color='blue',linestyle='--')
            plt.plot(x_axis,AddCons_RelRec[0:max_T],             color='red',linestyle='-')
            plt.plot(x_axis,AddCons_RelRec_AD[0:max_T],          color='red',linestyle='--') 
            plt.legend(['Inc, no AD effects','Inc, AD effects',\
                        'Cons, no AD effects','Cons, AD effects'], fontsize=14)
            plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
            plt.xlabel('quarter', fontsize=18)
            plt.ylabel('% diff. rel. to recession', fontsize=16)
            plt.show() 
            
        
        return Multipliers
        
    
    RecLengthInspect = 8
    Multiplier21qRecession_TaxCut = PlotsforSpecificRecLength(RecLengthInspect,'recessionTaxCut')
    print('NPV_Multiplier_Rec_TaxCut_AD for ' + str(RecLengthInspect) + ' q recession: ',mystr(Multiplier21qRecession_TaxCut[1][-1]))
    Multiplier21qRecession_UI = PlotsforSpecificRecLength(RecLengthInspect,'recessionUI')
    print('NPV_Multiplier_UI_Rec_AD for ' + str(RecLengthInspect) + ' q recession: ',mystr(Multiplier21qRecession_UI[1][-1]))
    Multiplier21qRecession_Check = PlotsforSpecificRecLength(RecLengthInspect,'recessionCheck')
    print('NPV_Multiplier_Rec_Check_AD for ' + str(RecLengthInspect) + ' q recession: ',mystr(Multiplier21qRecession_Check[1][-1]))
    
    
           
        
     
    
    output  ="\\begin{tabular}{@{}lccc@{}} \n"
    output +="\\toprule \n"
    output +="& Tax Cut    & UI extension    & Stimulus check    \\\\  \\midrule \n"
    output +="Recession lasts 2q &" + mystr3(PlotsforSpecificRecLength(2,'recessionTaxCut')[1][-1]) + "  & " + mystr3(PlotsforSpecificRecLength(2,'recessionUI')[1][-1]) + "  & " +  mystr3(PlotsforSpecificRecLength(2,'recessionCheck')[1][-1])  + "     \\\\ \n"
    output +="Recession lasts 4q &" + mystr3(PlotsforSpecificRecLength(4,'recessionTaxCut')[1][-1]) + "  & " + mystr3(PlotsforSpecificRecLength(4,'recessionUI')[1][-1]) + "  & " +  mystr3(PlotsforSpecificRecLength(4,'recessionCheck')[1][-1])  + "     \\\\ \n"
    output +="Recession lasts 8q &" + mystr3(PlotsforSpecificRecLength(8,'recessionTaxCut')[1][-1]) + "  & " + mystr3(PlotsforSpecificRecLength(8,'recessionUI')[1][-1]) + "  & " +  mystr3(PlotsforSpecificRecLength(8,'recessionCheck')[1][-1])  + "     \\\\ \n"
    output +="\\end{tabular}  \n"
    
    with open(table_dir + 'Multiplier_RecLengths.tex','w') as f:
        f.write(output)
        f.close()  
        
        
    #%% Output welfare tables
        
    Welfare_Results(saved_results_dir,table_dir,Parametrization=Parametrization)