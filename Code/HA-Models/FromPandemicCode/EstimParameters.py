import numpy as np
import os
import sys
from HARK.distribution import Uniform

# Targets in the estimation of the discount factor distributions for each 
# education level. 
# From SCF 2004: [20,40,60,80]-percentiles of the Lorenz curve for liquid wealth
data_LorenzPts_d = np.array([0, 0.01, 0.60, 3.58])    # \
data_LorenzPts_h = np.array([0.06, 0.63, 2.98, 11.6]) # -> units: % 
data_LorenzPts_c = np.array([0.15, 0.92, 3.27, 10.3]) # /
data_LorenzPts = [data_LorenzPts_d, data_LorenzPts_h, data_LorenzPts_c]
data_LorenzPtsAll = np.array([0.03, 0.35, 1.84, 7.42])
# From SCF 2004: Average liquid wealth to permanent income ratio 
data_avgLWPI = np.array([15.7, 47.7, 111])*4 # weighted average of fractions in percent
# From SCF 2004: Total LW over total PI by education group
data_LWoPI = np.array([28.1, 59.6, 162])*4 # units: %
# From SCF 2004: Weighted median of liquid wealth to permanent income ratio
data_medianLWPI = np.array([1.16, 7.55, 28.2])*4 # weighted median of fractions in percent

# Population share of each type
data_EducShares = [0.093, 0.527, 0.38] # Proportion of dropouts, HS grads, college types, SCF 2004 
# Wealth share of each type 
data_WealthShares = np.array([0.008, 0.179, 0.812])*100 # Percentage of total wealth of dropouts, HS grads, college types, SCF 2004 

# Parameters concerning the distribution of discount factors
# Initial values for estimation, taken from pandemic paperCondMrkvArrays_base
# Note: not really using these anymore
num_types = 3
DiscFacMeanD = 0.9647   # Mean intertemporal discount factor for dropout types
DiscFacMeanH = 0.98051  # Mean intertemporal discount factor for high school types
DiscFacMeanC = 0.99160  # Mean intertemporal discount factor for college types

DiscFacInit = [DiscFacMeanD, DiscFacMeanH, DiscFacMeanC]
DiscFacSpreadD = 0.025
DiscFacSpreadH = 0.01676
DiscFacSpreadC = 0.00480  # Half-width of uniform distribution of discount factors

# Define the distribution of the discount factor for each eduation level
DiscFacCount = 7
DiscFacDstnD = Uniform(DiscFacMeanD-DiscFacSpreadD, DiscFacMeanD+DiscFacSpreadD).approx(DiscFacCount)
DiscFacDstnH = Uniform(DiscFacMeanH-DiscFacSpreadH, DiscFacMeanH+DiscFacSpreadH).approx(DiscFacCount)
DiscFacDstnC = Uniform(DiscFacMeanC-DiscFacSpreadC, DiscFacMeanC+DiscFacSpreadC).approx(DiscFacCount)
DiscFacDstns = [DiscFacDstnD, DiscFacDstnH, DiscFacDstnC]

# Parameters concerning Markov transition matrix
#https://www.statista.com/statistics/232942/unemployment-rate-by-level-of-education-in-the-us/
Urate_normal_d = 0.085       # Unemployment rate in normal times, dropouts 2004
Urate_normal_h = 0.044       # Unemployment rate in normal times, highschooler+some college 2004
Urate_normal_c = 0.027       # Unemployment rate in normal times, college 2004

Uspell_normal = 1.5          # Average duration of unemployment spell in normal times, in quarters
UBspell_normal = 2           # Average duration of unemployment benefits in normal times, in quarters

# Basic model parameters: CRRA, growth factors, unemployment parameters (for normal times)
CRRA = 2.0                 # Coefficient of relative risk aversion (1, 2 or 3)
if len(sys.argv) >= 3:
    CRRA = float(sys.argv[2])
    if (CRRA != 1.0 and CRRA != 2.0 and CRRA != 3.0):
        print('The Splurge was only estimated for CRRA = 1.0, 2.0 and 3.0')
# Read in estimated Splurge --> depends on CRRA: 
f = open('../Target_AggMPCX_LiquWealth/Result_CRRA_'+str(CRRA)+'.txt', 'r')
dictload = eval(f.read())
Splurge = dictload['splurge'] 
# Splurge = 0.31 
# Splurge = 0.3138321699039471 # CRRA=1
# Splurge = 0.3067109441016833 # CRRA=2

PopGroFac = 1.0         #1.01**0.25  # Population growth factor
PermGroFacAgg = 1.0     #1.01**0.25 # Technological growth rate or aggregate productivity growth factor
IncUnemp = 0.7              # Unemployment benefits replacement rate (proportion of permanent income)
IncUnempNoBenefits = 0.5    # Unemployment income when benefits run out (proportion of permanent income)
if len(sys.argv) >= 5:
    IncUnemp = float(sys.argv[3])
    IncUnempNoBenefits = float(sys.argv[4])

# Parameters concerning the initial distribution of permanent income 
# "newborn" = 25 years old in SCF 2004
pLvlInitMean_d = np.log(6.2)   # Average quarterly permanent income of "newborn" HS dropout ($1000)
pLvlInitMean_h = np.log(11.1)  # Average quarterly permanent income of "newborn" HS graduate ($1000)
pLvlInitMean_c = np.log(14.5)  # Average quarterly permanent income of "newborn" HS  ($1000)
pLvlInitStd_d  = 0.32          # Standard deviation of initial log permanent income 
pLvlInitStd_h  = 0.42          # Standard deviation of initial log permanent income 
pLvlInitStd_c  = 0.53          # Standard deviation of initial log permanent income 

# Parameters concerning grid sizes: assets, permanent income shocks, transitory income shocks
aXtraMin = 0.001        # Lowest non-zero end-of-period assets above minimum gridpoint
aXtraMax = 40           # Highest non-zero end-of-period assets above minimum gridpoint
aXtraCount = 48         # Base number of end-of-period assets above minimum gridpoints
aXtraExtra = [0.002,0.003] # Additional gridpoints to "force" into the grid
aXtraNestFac = 3        # Exponential nesting factor for aXtraGrid (how dense is grid near zero)
PermShkCount = 7        # Number of points in equiprobable discrete approximation to permanent shock distribution
TranShkCount = 7        # Number of points in equiprobable discrete approximation to transitory shock distribution


# Size of simulations
AgentCountTotal = 50000 # Total simulated population
T_sim = 80              # Number of quarters to simulate in counterfactuals

# Basic lifecycle length parameters (don't touch these)
T_cycle = 1

# Define grid of aggregate assets to labor
CgridBase = np.array([0.8, 1.0, 1.2])  

num_base_MrkvStates = 2 + UBspell_normal #employed, unemployed with 2 quarters benefits, unemployed with 1 quarter benefit, unemployed no benefits
num_experiment_periods = 20

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
MrkvArray_base_d = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_d)
CondMrkvArrays_base_h = makeCondMrkvArrays_base(Urate_normal_h, Uspell_normal, UBspell_normal)
MrkvArray_base_h = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_h)
CondMrkvArrays_base_c = makeCondMrkvArrays_base(Urate_normal_c, Uspell_normal, UBspell_normal)
MrkvArray_base_c = makeFullMrkvArray(MacroMrkvArray_base, CondMrkvArrays_base_c)

# Define permanent income growth rates
PermGroFac_base =   [1.0]
PermGroFac_base_d = [1.0 + 0.01421/4]  # From Pandemic paper: avg growth rates during 
PermGroFac_base_h = [1.0 + 0.01812/4]  # working life for each education group 
PermGroFac_base_c = [1.0 + 0.01958/4]

# # Define the permanent and transitory shocks 
# TranShkStd = [0.1]
# PermShkStd = [0.05]
# Variances from Sticky expectations paper: 
TranShkStd = [np.sqrt(0.12)]
PermShkStd = [np.sqrt(0.003)]

Rfree_base = [1.01]             #Baseline
if len(sys.argv) >= 2:
    Rfree_base = [float(sys.argv[1])]
LivPrb_base = [1.0-1/160.0]     # 40 years (160 quarters) working life 

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

# find intial distribution of states for each education type
vals_d, vecs_d = np.linalg.eig(np.transpose(MrkvArray_base_d[0])) 
vals_d = vals_d.real
vecs_d = vecs_d.real
dist_d = np.abs(np.abs(vals_d) - 1.)
idx_d = np.argmin(dist_d)
init_mrkv_dist_d = vecs_d[:,idx_d].astype(float)/np.sum(vecs_d[:,idx_d].astype(float))

vals_h, vecs_h = np.linalg.eig(np.transpose(MrkvArray_base_h[0])) 
vals_h = vals_h.real
vecs_h = vecs_h.real
dist_h = np.abs(np.abs(vals_h) - 1.)
idx_h = np.argmin(dist_h)
init_mrkv_dist_h = vecs_h[:,idx_h].astype(float)/np.sum(vecs_h[:,idx_h].astype(float))

vals_c, vecs_c = np.linalg.eig(np.transpose(MrkvArray_base_c[0])) 
dist_c = np.abs(np.abs(vals_c) - 1.)
idx_c = np.argmin(dist_c)
init_mrkv_dist_c = vecs_c[:,idx_c].astype(float)/np.sum(vecs_c[:,idx_c].astype(float))

# Define a parameter dictionary for dropout type
init_dropout = {"cycles": 0, # This will be overwritten at type construction
                "T_cycle": T_cycle,
                'T_sim': 400, #Simulate up to age 400
                'T_age': None,
                'AgentCount': 10000,
                "PermGroFacAgg": PermGroFacAgg,
                "PopGroFac": PopGroFac,
                "CRRA": CRRA,
                "DiscFac": 0.98, # This will be overwritten at type construction
                "Rfree_base" : Rfree_base,
                "PermGroFac_base": PermGroFac_base_d,
                "LivPrb_base": LivPrb_base,
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
                'num_experiment_periods' : num_experiment_periods,
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
    'pLvlInitMean': pLvlInitMean_h,
    'pLvlInitStd': pLvlInitStd_h,
    "MrkvPrbsInit" : np.array(list(init_mrkv_dist_h)),
    'Urate_normal' : Urate_normal_h,
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
    'pLvlInitMean': pLvlInitMean_c,
    'pLvlInitStd': pLvlInitStd_c,
    "MrkvPrbsInit" : np.array(list(init_mrkv_dist_c)),
    'Urate_normal' : Urate_normal_c,
    'EducType' : 2}
init_college = init_dropout.copy()
init_college.update(adj_college)

# Define a dictionary to represent the baseline scenario
base_dict = {'shock_type' : "base",
             'UpdatePrb' : 1.0,
             'Splurge' : Splurge
             }

frictionless_changes = {
             'UpdatePrb' : 1.0
             }

# Parameters for AggregateDemandEconomy economy
intercept_prev = np.ones((num_base_MrkvStates,num_base_MrkvStates ))    # Intercept of aggregate savings function
slope_prev = np.zeros((num_base_MrkvStates,num_base_MrkvStates ))       # Slope of aggregate savings function
ADelasticity = 0.75                                                     # Elasticity of productivity to consumption

num_max_iterations_solvingAD = 30
convergence_tol_solvingAD = 1E-6
Cfunc_iter_stepsize       = 1

# Make a dictionary to specify a Cobb-Douglas economy
init_ADEconomy = {'intercept_prev': intercept_prev,
                     'slope_prev': slope_prev,
                     'ADelasticity' : 0.0,
                     'demand_ADelasticity' : ADelasticity,
                     'Cfunc_iter_stepsize' : Cfunc_iter_stepsize,
                     'MrkvArray' : MrkvArray_base_h,
                     'num_base_MrkvStates' : num_base_MrkvStates,
                     "MrkvArray_base" : MrkvArray_base_h, 
                     'CgridBase' : CgridBase,
                     'EconomyMrkvNow_init': 0,
                     'act_T' : 400
                     }