# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:44:33 2021

@author: edmun
"""
from Parameters import Rfree_base, Rspell, max_recession_duration
from OtherFunctions import loadPickle, getSimulationDiff
import numpy as np
import pandas as pd
from OtherFunctions import getSimulationDiff, getSimulationPercentDiff, getStimulus, getNPVMultiplier, \
                    saveAsPickleUnderVarName, loadPickle, namestr, saveAsPickle


figs_dir = './Figures/FullRun/'

base_results                        = loadPickle('base_results',figs_dir,locals())
check_results                       = loadPickle('Check_results',figs_dir,locals())
recession_results                   = loadPickle('recession_results',figs_dir,locals())
recession_results_AD                = loadPickle('recession_results_AD',figs_dir,locals())
recession_Check_results             = loadPickle('recessionCheck_results',figs_dir,locals())       
recession_Check_results_AD          = loadPickle('recessionCheck_results_AD',figs_dir,locals())

NPV_AddInc_Rec_Check                    = getSimulationDiff(recession_results,recession_Check_results,'NPV_AggIncome') 
NPV_Multiplier_Rec_Check                = getNPVMultiplier(recession_results,               recession_Check_results,               NPV_AddInc_Rec_Check)
NPV_Multiplier_Rec_Check_AD             = getNPVMultiplier(recession_results_AD,            recession_Check_results_AD,            NPV_AddInc_Rec_Check)
AddCons_relBase                 = 100*getSimulationDiff(recession_results,recession_Check_results,'NPV_AggCons')/recession_results['NPV_AggCons']
AddCons_relBase_AD              = 100*getSimulationDiff(recession_results_AD,recession_Check_results_AD,'NPV_AggCons')/recession_results_AD['NPV_AggCons']

print('Check size 1200 USD')
print('Additional consumption induced during recession:',AddCons_relBase[-1],"%")
print('Multiplier during recessions:',NPV_Multiplier_Rec_Check[-1])
print('Additional consumption induced during recession with AD:',AddCons_relBase_AD[-1],"%")
print('Multiplier during recessions with AD:',NPV_Multiplier_Rec_Check_AD[-1])

figs_dir = './Figures/FullRun_PVsame/'

check_results75                       = loadPickle('Check_results',figs_dir,locals())
recession_results_AD75                = loadPickle('recession_results_AD',figs_dir,locals())
recession_Check_results75             = loadPickle('recessionCheck_results',figs_dir,locals())       
recession_Check_results_AD75          = loadPickle('recessionCheck_results_AD',figs_dir,locals())

NPV_AddInc_Rec_Check75                    = getSimulationDiff(recession_results,recession_Check_results75,'NPV_AggIncome') 
NPV_Multiplier_Rec_Check75                = getNPVMultiplier(recession_results,               recession_Check_results75,               NPV_AddInc_Rec_Check75)
NPV_Multiplier_Rec_Check_AD75             = getNPVMultiplier(recession_results_AD,            recession_Check_results_AD75,            NPV_AddInc_Rec_Check75)
AddCons_relBase75                 = 100*getSimulationDiff(recession_results,recession_Check_results75,'NPV_AggCons')/recession_results['NPV_AggCons']
AddCons_relBase_AD75              = 100*getSimulationDiff(recession_results_AD,recession_Check_results_AD75,'NPV_AggCons')/recession_results_AD['NPV_AggCons']

print('Check size 75 USD')
print('Additional consumption induced during recession:',AddCons_relBase75[-1],"%")
print('Multiplier during recessions:',NPV_Multiplier_Rec_Check75[-1])
print('Additional consumption induced during recession with AD:',AddCons_relBase_AD75[-1],"%")
print('Multiplier during recessions with AD:',NPV_Multiplier_Rec_Check_AD75[-1])


figs_dir = './Figures/Check5k/'

check_results5k                       = loadPickle('Check_results',figs_dir,locals())
recession_results_AD5k                = loadPickle('recession_results_AD',figs_dir,locals())
recession_Check_results5k             = loadPickle('recessionCheck_results',figs_dir,locals())       
recession_Check_results_AD5k          = loadPickle('recessionCheck_results_AD',figs_dir,locals())

NPV_AddInc_Rec_Check5k                    = getSimulationDiff(recession_results,recession_Check_results5k,'NPV_AggIncome') 
NPV_Multiplier_Rec_Check5k                = getNPVMultiplier(recession_results,               recession_Check_results5k,               NPV_AddInc_Rec_Check5k)
NPV_Multiplier_Rec_Check_AD5k             = getNPVMultiplier(recession_results_AD,            recession_Check_results_AD5k,            NPV_AddInc_Rec_Check5k)
AddCons_relBase5k                 = 100*getSimulationDiff(recession_results,recession_Check_results5k,'NPV_AggCons')/recession_results['NPV_AggCons']
AddCons_relBase_AD5k              = 100*getSimulationDiff(recession_results_AD,recession_Check_results_AD5k,'NPV_AggCons')/recession_results_AD['NPV_AggCons']

print('Check size 75 USD')
print('Additional consumption induced during recession:',AddCons_relBase5k[-1],"%")
print('Multiplier during recessions:',NPV_Multiplier_Rec_Check5k[-1])
print('Additional consumption induced during recession with AD:',AddCons_relBase_AD5k[-1],"%")
print('Multiplier during recessions with AD:',NPV_Multiplier_Rec_Check_AD5k[-1])



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
output +="Size of stimulus check & \$75    &  \$1200    & \$5000    \\\\  \\midrule \n"
output +="%Add. cons. as share of baseline cons. (recession) &"      + mystr3(AddCons_relBase75[-1])             + " & " + mystr3(AddCons_relBase[-1])             + " & " + mystr3(AddCons_relBase5k[-1]) + "     \\\\ \n"
output +="Add. cons. as share of baseline cons. (recession, AD) &"  + mystr3(AddCons_relBase_AD75[-1])          + " & " + mystr3(AddCons_relBase_AD[-1])          + " & " + mystr3(AddCons_relBase_AD5k[-1]) + "     \\\\ \n"
output +="%Multiplier (recession) &"                                 + mystr3(NPV_Multiplier_Rec_Check75[-1])    + " & " + mystr3(NPV_Multiplier_Rec_Check[-1])    + " & " + mystr3(NPV_Multiplier_Rec_Check5k[-1]) + "     \\\\ \n"
output +="Multiplier (recession, AD) &"                             + mystr3(NPV_Multiplier_Rec_Check_AD75[-1]) + " & " + mystr3(NPV_Multiplier_Rec_Check_AD[-1]) + " & " + mystr3(NPV_Multiplier_Rec_Check_AD5k[-1]) + "     \\\\ \n"
output +="\\end{tabular}  \n"

with open('Tables/Multiplier_DifferentCheckSizes.tex','w') as f:
    f.write(output)
    f.close()    
