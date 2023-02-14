'''
This is the main script for the paper
'''
#from Parameters import returnParameters

import os
import sys

# for output
cwd             = os.getcwd()
folders         = cwd.split(os.path.sep)
top_most_folder = folders[-1]
if top_most_folder == 'FromPandemicCode':
    Abs_Path = cwd
else:
    Abs_Path = cwd + '\\FromPandemicCode'

sys.path.append(Abs_Path)
from Simulate import Simulate
from Output_Results import Output_Results

#%%


Run_Main                = True #DONE
Run_EqualPVs            = True #DONE
Run_ADElas_robustness   = True #DONE
Run_CRRA1_robustness    = True #DONE
Run_CRRA3_robustness    = True #DONE
Run_Rfree_robustness    = True #DONE
Run_Rspell_robustness   = True #DONE
Run_LowerUBnoB          = True #DONE


Run_Dict = dict()
Run_Dict['Run_Baseline']            = True
Run_Dict['Run_Recession ']          = True
Run_Dict['Run_Check_Recession']     = True
Run_Dict['Run_UB_Ext_Recession']    = True
Run_Dict['Run_TaxCut_Recession']    = True
Run_Dict['Run_Check']               = True
Run_Dict['Run_UB_Ext']              = True
Run_Dict['Run_TaxCut']              = True
Run_Dict['Run_AD ']                 = True
Run_Dict['Run_1stRoundAD']          = True
Run_Dict['Run_NonAD']               = True

#%% Execute main Simulation

if Run_Main:

    figs_dir = Abs_Path+'/Figures/CRRA2/'    
    Simulate(Run_Dict,figs_dir,Parametrization='Baseline')    
    Output_Results(Abs_Path+'/Figures/CRRA2/',Abs_Path+'/Figures/',Abs_Path+'/Tables/CRRA2/',Parametrization='Baseline')
    
if Run_EqualPVs:
           
    figs_dir = Abs_Path+'/Figures/CRRA2_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA2_PVSame')
    Output_Results(Abs_Path+'/Figures/CRRA2_PVSame/',Abs_Path+'/Figures/CRRA2_PVSame/',Abs_Path+'/Tables/CRRA2_PVSame/',Parametrization='CRRA2_PVSame')

# Welfare4.tex contains the relevant results for the welfare analysis





#%% 
if Run_ADElas_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = Abs_Path+'/Figures/ADElas/'
    Simulate(Run_Dict,figs_dir,Parametrization='ADElas')
    Output_Results(Abs_Path+'/Figures/ADElas/',Abs_Path+'/Figures/ADElas/',Abs_Path+'/Tables/ADElas/',Parametrization='ADElas')
     
    figs_dir = Abs_Path+'/Figures/ADElas_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='ADElas_PVSame')
    Output_Results(Abs_Path+'/Figures/ADElas_PVSame/',Abs_Path+'/Figures/ADElas_PVSame/',Abs_Path+'/Tables/ADElas_PVSame/',Parametrization='ADElas_PVSame')


    
#%% Execute robustness run
        
if Run_CRRA1_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = Abs_Path+'/Figures/CRRA1/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA1')
    Output_Results(Abs_Path+'/Figures/CRRA1/',Abs_Path+'/Figures/CRRA1/',Abs_Path+'/Tables/CRRA1/',Parametrization='CRRA1')
     
    figs_dir = Abs_Path+'/Figures/CRRA1_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA1_PVSame')
    Output_Results(Abs_Path+'/Figures/CRRA1_PVSame/',Abs_Path+'/Figures/CRRA1_PVSame/',Abs_Path+'/Tables/CRRA1_PVSame/',Parametrization='CRRA1_PVSame')

if Run_CRRA3_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = Abs_Path+'/Figures/CRRA3/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA3')
    Output_Results(Abs_Path+'/Figures/CRRA3/',Abs_Path+'/Figures/CRRA3/',Abs_Path+'/Tables/CRRA3/',Parametrization='CRRA3')
     
    figs_dir = Abs_Path+'/Figures/CRRA3_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='CRRA3_PVSame')
    Output_Results(Abs_Path+'/Figures/CRRA3_PVSame/',Abs_Path+'/Figures/CRRA3_PVSame/',Abs_Path+'/Tables/CRRA3_PVSame/',Parametrization='CRRA3_PVSame')



#%%

if Run_Rfree_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = Abs_Path+'/Figures/Rfree_1005/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1005')
    Output_Results(Abs_Path+'/Figures/Rfree_1005/',Abs_Path+'/Figures/Rfree_1005/',Abs_Path+'/Tables/Rfree_1005/',Parametrization='Rfree_1005')
     
    figs_dir = Abs_Path+'/Figures/Rfree_1005_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1005_PVSame')
    Output_Results(Abs_Path+'/Figures/Rfree_1005_PVSame/',Abs_Path+'/Figures/Rfree_1005_PVsame/',Abs_Path+'/Tables/Rfree_1005_PVsame/',Parametrization='Rfree_1005_PVSame')

    figs_dir = Abs_Path+'/Figures/Rfree_1015/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1015')
    Output_Results(Abs_Path+'/Figures/Rfree_1015/',Abs_Path+'/Figures/Rfree_1015/',Abs_Path+'/Tables/Rfree_1015/',Parametrization='Rfree_1015')
     
    figs_dir = Abs_Path+'/Figures/Rfree_1015_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rfree_1015_PVSame')
    Output_Results(Abs_Path+'/Figures/Rfree_1015_PVSame/',Abs_Path+'/Figures/Rfree_1015_PVSame/',Abs_Path+'/Tables/Rfree_1015_PVSame/',Parametrization='Rfree_1015_PVSame')


if Run_Rspell_robustness:
    
    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = Abs_Path+'/Figures/Rspell_4/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rspell_4')
    Output_Results(Abs_Path+'/Figures/Rspell_4/',Abs_Path+'/Figures/Rspell_4/',Abs_Path+'/Tables/Rspell_4/',Parametrization='Rspell_4')
     
    figs_dir = Abs_Path+'/Figures/Rspell_4_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='Rspell_4_PVSame')
    Output_Results(Abs_Path+'/Figures/Rspell_4_PVSame/',Abs_Path+'/Figures/Rspell_4_PVSame/',Abs_Path+'/Tables/Rspell_4_PVSame/',Parametrization='Rspell_4_PVSame')


if Run_LowerUBnoB:

    Run_Dict['Run_1stRoundAD']          = False

    figs_dir = Abs_Path+'/Figures/LowerUBnoB/'
    Simulate(Run_Dict,figs_dir,Parametrization='LowerUBnoB')
    Output_Results(Abs_Path+'/Figures/LowerUBnoB/',Abs_Path+'/Figures/LowerUBnoB/',Abs_Path+'/Tables/LowerUBnoB/',Parametrization='LowerUBnoB')
   
    figs_dir = Abs_Path+'/Figures/LowerUBnoB_PVSame/'
    Simulate(Run_Dict,figs_dir,Parametrization='LowerUBnoB_PVSame')
    Output_Results(Abs_Path+'/Figures/LowerUBnoB_PVSame/',Abs_Path+'/Figures/LowerUBnoB_PVSame/',Abs_Path+'/Tables/LowerUBnoB_PVSame/',Parametrization='LowerUBnoB_PVSame')
 