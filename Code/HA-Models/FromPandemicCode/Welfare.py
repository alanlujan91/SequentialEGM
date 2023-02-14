# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:44:33 2021

@author: edmun
"""
from Parameters import returnParameters
from OtherFunctions import loadPickle, getSimulationDiff
import numpy as np
import pandas as pd

def Welfare_Results(saved_results_dir,table_dir,Parametrization='Baseline'):
    
    
    [max_recession_duration, Rspell, Rfree_base]  = returnParameters(Parametrization=Parametrization,OutputFor='_Output_Results.py')
    

    
    base_results                        = loadPickle('base_results',saved_results_dir,locals())
    check_results                       = loadPickle('Check_results',saved_results_dir,locals())
    UI_results                          = loadPickle('UI_results',saved_results_dir,locals())
    TaxCut_results                      = loadPickle('TaxCut_results',saved_results_dir,locals())
    
    recession_results                   = loadPickle('recession_results',saved_results_dir,locals())
    recession_results_AD                = loadPickle('recession_results_AD',saved_results_dir,locals())
    recession_UI_results                = loadPickle('recessionUI_results',saved_results_dir,locals())       
    recession_UI_results_AD             = loadPickle('recessionUI_results_AD',saved_results_dir,locals())  
    recession_Check_results             = loadPickle('recessionCheck_results',saved_results_dir,locals())       
    recession_Check_results_AD          = loadPickle('recessionCheck_results_AD',saved_results_dir,locals())
    recession_TaxCut_results            = loadPickle('recessionTaxCut_results',saved_results_dir,locals())
    recession_TaxCut_results_AD         = loadPickle('recessionTaxCut_results_AD',saved_results_dir,locals())
    
    recession_all_results                   = loadPickle('recession_all_results',saved_results_dir,locals())
    recession_all_results_AD                = loadPickle('recession_all_results_AD',saved_results_dir,locals())
    recession_UI_all_results                = loadPickle('recessionUI_all_results',saved_results_dir,locals())       
    recession_UI_all_results_AD             = loadPickle('recessionUI_all_results_AD',saved_results_dir,locals())  
    recession_Check_all_results             = loadPickle('recessionCheck_all_results',saved_results_dir,locals())       
    recession_Check_all_results_AD          = loadPickle('recessionCheck_all_results_AD',saved_results_dir,locals())
    recession_TaxCut_all_results            = loadPickle('recessionTaxCut_all_results',saved_results_dir,locals())
    recession_TaxCut_all_results_AD         = loadPickle('recessionTaxCut_all_results_AD',saved_results_dir,locals())
    
    NPV_AddInc_Rec_Check                = getSimulationDiff(recession_results,recession_Check_results,'NPV_AggIncome') 
    NPV_AddInc_UI_Rec                   = getSimulationDiff(recession_results,recession_UI_results,'NPV_AggIncome') # Policy expenditure
    NPV_AddInc_Rec_TaxCut               = getSimulationDiff(recession_results,recession_TaxCut_results,'NPV_AggIncome')
    
    NPV_AddInc_Check                = getSimulationDiff(base_results,check_results,'NPV_AggIncome') 
    NPV_AddInc_UI                   = getSimulationDiff(base_results,UI_results,'NPV_AggIncome') # Policy expenditure
    NPV_AddInc_TaxCut               = getSimulationDiff(base_results,TaxCut_results,'NPV_AggIncome')
    
    #Assumes log utility
    base_welfare   = np.log(base_results['cLvl_all_splurge'])
    check_welfare  = np.log(check_results['cLvl_all_splurge'])
    UI_welfare     = np.log(UI_results['cLvl_all_splurge'])
    TaxCut_welfare = np.log(TaxCut_results['cLvl_all_splurge'])
    
    R_persist = 1.-1./Rspell
    recession_prob_array = np.array([R_persist**t*(1-R_persist) for t in range(max_recession_duration)])
    
    recession_welfare = np.log(np.sum(np.array([recession_all_results[t]['cLvl_all_splurge']*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0) )  
    recession_welfare_AD = np.log(np.sum(np.array([recession_all_results_AD[t]['cLvl_all_splurge']*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0) )  
    recession_UI_welfare = np.log(np.sum(np.array([recession_UI_all_results[t]['cLvl_all_splurge']*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0) )  
    recession_UI_welfare_AD = np.log(np.sum(np.array([recession_UI_all_results_AD[t]['cLvl_all_splurge']*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0) )  
    recession_TaxCut_welfare = np.log(np.sum(np.array([recession_TaxCut_all_results[t]['cLvl_all_splurge']*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0) )  
    recession_TaxCut_welfare_AD = np.log(np.sum(np.array([recession_TaxCut_all_results_AD[t]['cLvl_all_splurge']*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0) )  
    recession_Check_welfare = np.log(np.sum(np.array([recession_Check_all_results[t]['cLvl_all_splurge']*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0) )  
    recession_Check_welfare_AD = np.log(np.sum(np.array([recession_Check_all_results_AD[t]['cLvl_all_splurge']*recession_prob_array[t]  for t in range(max_recession_duration)]), axis=0) )  
    
    def SP_welfare(individual_welfare, SP_discount_rate):
        welfare = np.sum(np.sum(individual_welfare, axis=1)*np.array([SP_discount_rate**t for t in range(individual_welfare.shape[0])]))
        return welfare
    
    SP_discount_rate = 1/Rfree_base[0]
    Check_welfare_impact = SP_welfare(check_welfare, SP_discount_rate)-SP_welfare(base_welfare, SP_discount_rate)
    Check_welfare_impact_recession = SP_welfare(recession_Check_welfare, SP_discount_rate)-SP_welfare(recession_welfare, SP_discount_rate)
    Check_welfare_impact_recession_AD = SP_welfare(recession_Check_welfare_AD, SP_discount_rate)-SP_welfare(recession_welfare_AD, SP_discount_rate)
    
    UI_welfare_impact = SP_welfare(UI_welfare, SP_discount_rate)-SP_welfare(base_welfare, SP_discount_rate)
    UI_welfare_impact_recession = SP_welfare(recession_UI_welfare, SP_discount_rate)-SP_welfare(recession_welfare, SP_discount_rate)
    UI_welfare_impact_recession_AD = SP_welfare(recession_UI_welfare_AD, SP_discount_rate)-SP_welfare(recession_welfare_AD, SP_discount_rate)
    
    TaxCut_welfare_impact = SP_welfare(TaxCut_welfare, SP_discount_rate)-SP_welfare(base_welfare, SP_discount_rate)
    TaxCut_welfare_impact_recession = SP_welfare(recession_TaxCut_welfare, SP_discount_rate)-SP_welfare(recession_welfare, SP_discount_rate)
    TaxCut_welfare_impact_recession_AD = SP_welfare(recession_TaxCut_welfare_AD, SP_discount_rate)-SP_welfare(recession_welfare_AD, SP_discount_rate)
    
    Check_welfare_per_dollar_AD  = (Check_welfare_impact_recession_AD  - Check_welfare_impact) /NPV_AddInc_Rec_Check[-1]
    UI_welfare_per_dollar_AD     = (UI_welfare_impact_recession_AD     - UI_welfare_impact)    /NPV_AddInc_UI_Rec[-1]
    TaxCut_welfare_per_dollar_AD = (TaxCut_welfare_impact_recession_AD - TaxCut_welfare_impact)/NPV_AddInc_Rec_TaxCut[-1]
    
    Check_welfare_per_dollar2_AD = Check_welfare_impact_recession_AD/NPV_AddInc_Rec_Check[-1] - Check_welfare_impact/NPV_AddInc_Check[-1]
    UI_welfare_per_dollar2_AD = UI_welfare_impact_recession_AD/NPV_AddInc_UI_Rec[-1] - UI_welfare_impact/NPV_AddInc_UI[-1]
    TaxCut_welfare_per_dollar2_AD = TaxCut_welfare_impact_recession_AD/NPV_AddInc_Rec_TaxCut[-1] - TaxCut_welfare_impact/NPV_AddInc_TaxCut[-1]
     
    Check_welfare_per_dollar  = (Check_welfare_impact_recession  - Check_welfare_impact) /NPV_AddInc_Rec_Check[-1]
    UI_welfare_per_dollar     = (UI_welfare_impact_recession     - UI_welfare_impact)    /NPV_AddInc_UI_Rec[-1]
    TaxCut_welfare_per_dollar = (TaxCut_welfare_impact_recession - TaxCut_welfare_impact)/NPV_AddInc_Rec_TaxCut[-1]
    
    Check_welfare_per_dollar2 = Check_welfare_impact_recession/NPV_AddInc_Rec_Check[-1] - Check_welfare_impact/NPV_AddInc_Check[-1]
    UI_welfare_per_dollar2 = UI_welfare_impact_recession/NPV_AddInc_UI_Rec[-1] - UI_welfare_impact/NPV_AddInc_UI[-1]
    TaxCut_welfare_per_dollar2 = TaxCut_welfare_impact_recession/NPV_AddInc_Rec_TaxCut[-1] - TaxCut_welfare_impact/NPV_AddInc_TaxCut[-1]
     
    all_welfare_results = pd.DataFrame([[Check_welfare_per_dollar_AD,UI_welfare_per_dollar_AD,TaxCut_welfare_per_dollar_AD],
    [Check_welfare_per_dollar2_AD,UI_welfare_per_dollar2_AD,TaxCut_welfare_per_dollar2_AD],
    [Check_welfare_per_dollar,UI_welfare_per_dollar,TaxCut_welfare_per_dollar],
    [Check_welfare_per_dollar2,UI_welfare_per_dollar2,TaxCut_welfare_per_dollar2]])
    
    
    def mystr3(number):
        if not np.isnan(number):
            out = "{:.3f}".format(number)
        else:
            out = ''
        return out
    output  ="\\begin{tabular}{@{}lccc@{}} \n"
    output +="\\toprule \n"
    output +="                          & Check      & UI    & Tax Cut    \\\\  \\midrule \n"
    output +="$\\mathcal{G}(\\text{policy})$ & "  + mystr3(Check_welfare_per_dollar) + "  & "+ mystr3(UI_welfare_per_dollar) +  "  & "+  mystr3(TaxCut_welfare_per_dollar)  + "     \\\\ \n"
    output +="$\\mathcal{G}(AD,\\text{policy})$ & "  + mystr3(Check_welfare_per_dollar_AD) + "  & "+ mystr3(UI_welfare_per_dollar_AD) +  "  & "+  mystr3(TaxCut_welfare_per_dollar_AD)  + "     \\\\ \n"
    output +="\\end{tabular}  \n"
    with open(table_dir+'welfare1.tex','w') as f:
        f.write(output)
        f.close()
    output  ="\\begin{tabular}{@{}lccc@{}} \n"
    output +="\\toprule \n"
    output +="                          & Check      & UI    & Tax Cut    \\\\  \\midrule \n"
    output +="$\\mathcal{G}(\\text{policy})$ & "  + mystr3(Check_welfare_per_dollar2) + "  & "+ mystr3(UI_welfare_per_dollar2) +  "  & "+  mystr3(TaxCut_welfare_per_dollar2)  + "     \\\\ \n"
    output +="$\\mathcal{G}(AD,\\text{policy})$ & "  + mystr3(Check_welfare_per_dollar2_AD) + "  & "+ mystr3(UI_welfare_per_dollar2_AD) +  "  & "+  mystr3(TaxCut_welfare_per_dollar2_AD)  + "     \\\\ \n"
    output +="\\end{tabular}  \n"
    with open(table_dir+'welfare2.tex','w') as f:
        f.write(output)
        f.close()
    
    
    
    periods = base_results['cLvl_all_splurge'].shape[0]
    num_agents = base_results['cLvl_all_splurge'].shape[1]
    discount_array = np.transpose(np.array([[Rfree_base[0]**(-i) for i in range(periods)]]*num_agents))
    base_weights   = base_results['cLvl_all_splurge']*discount_array
    base_welfare   = np.log(base_results['cLvl_all_splurge'])
    check_welfare  = np.log(check_results['cLvl_all_splurge'])
    UI_welfare     = np.log(UI_results['cLvl_all_splurge'])
    TaxCut_welfare = np.log(TaxCut_results['cLvl_all_splurge'])
    
    check_extra_welfare_ltd = np.sum((check_welfare - base_welfare)*base_weights)/np.sum((check_results['cLvl_all_splurge']-base_results['cLvl_all_splurge'])*discount_array)
    UI_extra_welfare_ltd    = np.sum((UI_welfare    - base_welfare)*base_weights)/np.sum((UI_results['cLvl_all_splurge']-base_results['cLvl_all_splurge'])*discount_array)
    TaxCut_extra_welfare_ltd = np.sum((TaxCut_welfare - base_welfare)*base_weights)/np.sum((TaxCut_results['cLvl_all_splurge']-base_results['cLvl_all_splurge'])*discount_array)
    
    smallUI = base_results['cLvl_all_splurge'] + (UI_results['cLvl_all_splurge'] - base_results['cLvl_all_splurge'])/10000
    smallUI_welfare     = np.log(smallUI)
    smallUI_extra_welfare    = np.sum((smallUI_welfare    - base_welfare)*base_weights)/np.sum((smallUI-base_results['cLvl_all_splurge'])*discount_array)
    
    check_extra_welfare = np.sum((check_welfare - base_welfare)*base_weights)/NPV_AddInc_Check[-1]
    UI_extra_welfare    = np.sum((UI_welfare    - base_welfare)*base_weights)/NPV_AddInc_UI[-1]
    TaxCut_extra_welfare = np.sum((TaxCut_welfare - base_welfare)*base_weights)/NPV_AddInc_TaxCut[-1]
    
    
    check_extra_welfare_AD = np.sum((recession_Check_welfare_AD - recession_welfare_AD)*base_weights)/NPV_AddInc_Rec_Check[-1]
    UI_extra_welfare_AD    = np.sum((recession_UI_welfare_AD    - recession_welfare_AD)*base_weights)/NPV_AddInc_UI_Rec[-1]
    TaxCut_extra_welfare_AD = np.sum((recession_TaxCut_welfare_AD - recession_welfare_AD)*base_weights)/NPV_AddInc_Rec_TaxCut[-1]
    
    check_extra_welfare_rec = np.sum((recession_Check_welfare - recession_welfare)*base_weights)/NPV_AddInc_Rec_Check[-1]
    UI_extra_welfare_rec    = np.sum((recession_UI_welfare    - recession_welfare)*base_weights)/NPV_AddInc_UI_Rec[-1]
    TaxCut_extra_welfare_rec = np.sum((recession_TaxCut_welfare - recession_welfare)*base_weights)/NPV_AddInc_Rec_TaxCut[-1]
    
    output  ="\\begin{tabular}{@{}lccc@{}} \n"
    output +="\\toprule \n"
    output +="                          & Check      & UI    & Tax Cut    \\\\  \\midrule \n"
    output +="$\\mathcal{G}3(\\text{policy})$ & "          + mystr3(check_extra_welfare)     + "  & "+ mystr3(UI_extra_welfare)     +  "  & "+  mystr3(TaxCut_extra_welfare)  + "     \\\\ \n"
    output +="$\\mathcal{G}3(Rec,\\text{policy})$ & "      + mystr3(check_extra_welfare_rec) + "  & "+ mystr3(UI_extra_welfare_rec) +  "  & "+  mystr3(TaxCut_extra_welfare_rec)  + "     \\\\ \n"
    output +="$\\mathcal{G}3(Rec, AD,\\text{policy})$ & "  + mystr3(check_extra_welfare_AD)  + "  & "+ mystr3(UI_extra_welfare_AD)  +  "  & "+  mystr3(TaxCut_extra_welfare_AD)  + "     \\\\ \n"
    output +="\\end{tabular}  \n"
    with open(table_dir+'welfare3.tex','w') as f:
        f.write(output)
        f.close()
        
    #### METHOD 3
    W_c = 1/(1-SP_discount_rate)*base_welfare.shape[1]
    P_c = 1/(1-SP_discount_rate)*base_results['AggCons'][0]
    
    Check_consumption_welfare   = (Check_welfare_impact_recession/W_c  - NPV_AddInc_Rec_Check[-1]/P_c)   - (Check_welfare_impact/W_c  - NPV_AddInc_Check[-1]/P_c) 
    UI_consumption_welfare      = (UI_welfare_impact_recession/W_c     - NPV_AddInc_UI_Rec[-1]/P_c)      - (UI_welfare_impact/W_c     - NPV_AddInc_UI[-1]/P_c) 
    TaxCut_consumption_welfare  = (TaxCut_welfare_impact_recession/W_c - NPV_AddInc_Rec_TaxCut[-1]/P_c)  - (TaxCut_welfare_impact/W_c - NPV_AddInc_TaxCut[-1]/P_c) 
    
    Check_consumption_welfare_AD   = (Check_welfare_impact_recession_AD/W_c  - NPV_AddInc_Rec_Check[-1]/P_c)   - (Check_welfare_impact/W_c  - NPV_AddInc_Check[-1]/P_c) 
    UI_consumption_welfare_AD      = (UI_welfare_impact_recession_AD/W_c     - NPV_AddInc_UI_Rec[-1]/P_c)      - (UI_welfare_impact/W_c     - NPV_AddInc_UI[-1]/P_c) 
    TaxCut_consumption_welfare_AD  = (TaxCut_welfare_impact_recession_AD/W_c - NPV_AddInc_Rec_TaxCut[-1]/P_c)  - (TaxCut_welfare_impact/W_c - NPV_AddInc_TaxCut[-1]/P_c) 
    
    #format as basis points
    def mystr3bp(number):
        if not np.isnan(number):
            out = "{:.3f}".format(number*10000)
        else:
            out = ''
        return out
    output  ="\\begin{tabular}{@{}lccc@{}} \n"
    output +="\\toprule \n"
    output +="                          & Check      & UI    & Tax Cut    \\\\  \\midrule \n"
    output +="$\\mathcal{C}(Rec,\\text{policy})$ & "      + mystr3bp(Check_consumption_welfare)     + "  & "+ mystr3bp(UI_consumption_welfare)     +  "  & "+  mystr3bp(TaxCut_consumption_welfare)  + "     \\\\ \n"
    output +="$\\mathcal{C}(Rec, AD,\\text{policy})$ & "  + mystr3bp(Check_consumption_welfare_AD)  + "  & "+ mystr3bp(UI_consumption_welfare_AD)  +  "  & "+  mystr3bp(TaxCut_consumption_welfare_AD)  + "     \\\\ \n"
    output +="\\end{tabular}  \n"
    with open(table_dir+'welfare4.tex','w') as f:
        f.write(output)
        f.close()
    
    print(NPV_AddInc_Check[-1])
    print(NPV_AddInc_UI[-1])
    print(NPV_AddInc_TaxCut[-1])