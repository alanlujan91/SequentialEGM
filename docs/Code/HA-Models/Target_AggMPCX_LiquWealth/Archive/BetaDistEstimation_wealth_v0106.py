'''
This is the main file for the cstwMPC project.  It estimates one version of the model
each time it is executed.  The following parameters *must* be defined in the __main__
namespace in order for this file to run correctly:
    
param_name : str
    Which parameter to introduce heterogeneity in (usually DiscFac).
dist_type : str
    Which type of distribution to use (can be 'uniform' or 'lognormal').
do_param_dist : bool
    Do param-dist version if True, param-point if False.
do_lifecycle : bool
    Use lifecycle model if True, perpetual youth if False.
do_agg_shocks : bool
    Whether to solve the FBS aggregate shocks version of the model or use idiosyncratic shocks only.
do_liquid : bool
    Matches liquid assets data when True, net worth data when False.
do_tractable : bool
    Whether to use an extremely simple alternate specification of households' optimization problem.
run_estimation : bool
    Whether to actually estimate the model specified by the other options.
run_sensitivity : [bool]
    Whether to run each of eight sensitivity analyses; currently inoperative.  Order:
    rho, xi_sigma, psi_sigma, mu, urate, mortality, g, R
find_beta_vs_KY : bool
    Whether to computes K/Y ratio for a wide range of beta; should have do_param_dist = False and param_name = 'DiscFac'.
    Currently inoperative.
path_to_models : str
    Absolute path to the location of this file.
    
All of these parameters are set when running this file from one of the do_XXX.py
files in the root directory.
'''
from __future__ import division, print_function
from __future__ import absolute_import

from builtins import str
from builtins import range

import os

import numpy as np
from copy import copy, deepcopy
from time import clock

# new way of loading modules
from HARK.distribution import approxMeanOneLognormal, combineIndepDstns, approxUniform, approxLognormal
from HARK.utilities import getPercentiles, getLorenzShares, calcSubpopAvg
from HARK.distribution import DiscreteDistribution

from HARK import Market
import HARK.ConsumptionSaving.ConsIndShockModel as Model
from HARK.ConsumptionSaving.ConsAggShockModel import CobbDouglasEconomy, AggShockConsumerType
from cstwMPC_Adjusted import cstwMPCagent, cstwMPCmarket, getKYratioDifference, findLorenzDistanceAtTargetKY, calcStationaryAgeDstn
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

from IPython import get_ipython # Needed to test whether being run from command line or interactively

import SetupParamsCSTW as Params

mystr = lambda number : "{:.3f}".format(number)



#%%
run_estimation = True
dist_type = 'uniform'


param_name = 'DiscFac'
param_text = 'beta'
do_lifecycle = False
life_text = 'PY'
do_param_dist = True
model_text = 'Dist'
do_liquid = False
wealth_text = 'NetWorth'
do_agg_shocks = False
shock_text = 'Ind'
spec_name = life_text + param_text + model_text + shock_text + wealth_text

if do_param_dist:
    pref_type_count = 7       # Number of discrete beta types in beta-dist
else:
    pref_type_count = 1       # Just one beta type in beta-point
    
#%%    

###############################################################################
### ACTUAL WORK BEGINS BELOW THIS LINE  #######################################
###############################################################################

  
if do_liquid:
    lorenz_target = np.array([0.0024, 0.0168, 0.0591, 0.1653]) # Data from Elin for financial assets in 2017
    lorenz_target_interp = np.interp(np.arange(0.01,1.00,0.01),np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]),np.array([0.0003, 0.0024, 0.0073, 0.0168, 0.0331, 0.0591, 0.0997, 0.1653, 0.2827, 0.3912, 0.5883, 0.7824, 1])) 
    lorenz_long_data = np.hstack((np.array(0.0),lorenz_target_interp,np.array(1.0))) 
    KY_target = 9.2088 #This is not correct and taken from below
else: # This is hacky until I can find the liquid wealth data and import it
    lorenz_target = np.array([-0.0388, -0.0182, 0.0856, 0.3024]) # Data from Elin for net wealth in 2017
    #I set value at 0.01 at about 1/10 of the value at 0.1 following the logic in data file
    lorenz_target_interp = np.interp(np.arange(0.01,1.00,0.01),np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]),np.array([-0.005, -0.0363, -0.0388, -0.0359, -0.0182, 0.0217, 0.0856, 0.1768, 0.3024, 0.4824, 0.6145, 0.7902, 0.9001, 1]))
    lorenz_long_data = np.hstack((np.array(0.0),lorenz_target_interp,np.array(1.0)))  
    KY_target = 9.2088
    
# Set total number of simulated agents in the population
if do_param_dist:
    if do_agg_shocks:
        Population = Params.pop_sim_agg_dist
    else:
        Population = Params.pop_sim_ind_dist
else:
    if do_agg_shocks:
        Population = Params.pop_sim_agg_point
    else:
        Population = Params.pop_sim_ind_point
    


# Make AgentTypes for estimation
PerpetualYouthType = cstwMPCagent(**Params.init_infinite)
PerpetualYouthType.AgeDstn = np.array(1.0)
# Set Borrowing constraint to -5 of permanent income
# since a quarterly model, -20 of qu. permanent income
#PerpetualYouthType.BoroCnstArt = -20
#PerpetualYouthType.IncUnemp = 0.68
PerpetualYouthType.update()

#PerpetualYouthType.solve()

#%%

EstimationAgentList = []
for n in range(pref_type_count):
    EstimationAgentList.append(deepcopy(PerpetualYouthType))

# Give all the AgentTypes different seeds
for j in range(len(EstimationAgentList)):
    EstimationAgentList[j].seed = j

# Make an economy for the consumers to live in
market_dict = copy(Params.init_market)
market_dict['AggShockBool'] = do_agg_shocks
market_dict['Population'] = Population
EstimationEconomy = cstwMPCmarket(**market_dict)
EstimationEconomy.IncUnemp = PerpetualYouthType.IncUnemp
EstimationEconomy.agents = EstimationAgentList
EstimationEconomy.KYratioTarget = KY_target
EstimationEconomy.LorenzTarget = lorenz_target
EstimationEconomy.LorenzData = lorenz_long_data

EstimationEconomy.PopGroFac = 1.0
EstimationEconomy.TypeWeight = [1.0]
EstimationEconomy.act_T = Params.T_sim_PY
EstimationEconomy.ignore_periods = Params.ignore_periods_PY


#%%
# center=0.9879177102415481
# spread=0.004534079415384556
# EstimationEconomy(LorenzBool = False, ManyStatsBool = False) # Make sure we're not wasting time calculating stuff
# EstimationEconomy.distributeParams(param_name,pref_type_count,center,spread,dist_type) # Distribute parameters
# EstimationEconomy.solve()


#%%
# Estimate the model as requested
if run_estimation:
    print('Beginning an estimation with the specification name ' + spec_name + '...')
    
    # Choose the bounding region for the parameter search
    param_range = [0.95,0.995]
    spread_range = [0.004,0.008]

    if do_param_dist:
        # Run the param-dist estimation
        paramDistObjective = lambda spread : findLorenzDistanceAtTargetKY(
                                                        Economy = EstimationEconomy,
                                                        param_name = param_name,
                                                        param_count = pref_type_count,
                                                        center_range = param_range,
                                                        spread = spread,
                                                        dist_type = dist_type)
        t_start = clock()
        spread_estimate = (minimize_scalar(paramDistObjective,bracket=spread_range,tol=1e-2,method='brent')).x
        center_estimate = EstimationEconomy.center_save
        t_end = clock()
    else:
        # Run the param-point estimation only
        paramPointObjective = lambda center : getKYratioDifference(Economy = EstimationEconomy,
                                             param_name = param_name,
                                             param_count = pref_type_count,
                                             center = center,
                                             spread = 0.0,
                                             dist_type = dist_type)
        t_start = clock()
        center_estimate = brentq(paramPointObjective,param_range[0],param_range[1],xtol=1e-2)
        spread_estimate = 0.0
        t_end = clock()

    # Display statistics about the estimated model
    #center_estimate = 0.986609223266
    #spread_estimate = 0.00853886395698
    EstimationEconomy.LorenzBool = True
    EstimationEconomy.ManyStatsBool = True
    EstimationEconomy.distributeParams(param_name, pref_type_count,center_estimate,spread_estimate, dist_type)
    EstimationEconomy.solve()
    EstimationEconomy.calcLorenzDistance()
    print('Estimate is center=' + str(center_estimate) + ', spread=' + str(spread_estimate) + ', took ' + str(t_end-t_start) + ' seconds.')
    EstimationEconomy.center_estimate = center_estimate
    EstimationEconomy.spread_estimate = spread_estimate
    EstimationEconomy.showManyStats(spec_name)
    print('These results have been saved to ./Code/Results/' + spec_name + '.txt\n\n')


#%%
# np.savetxt('Results/IncUnemp86_BoroConst20_LorenzLong_hist.dat', EstimationEconomy.LorenzLong_hist)
# np.savetxt('Results/IncUnemp86_BoroConst20_ignore_periods.dat', [EstimationEconomy.ignore_periods])
# np.savetxt('Results/IncUnemp86_BoroConst20_LorenzData.dat', EstimationEconomy.LorenzData)

# np.savetxt('Results/BoroConst20_LorenzLong_hist.dat', EstimationEconomy.LorenzLong_hist)
# np.savetxt('Results/BoroConst20_ignore_periods.dat', [EstimationEconomy.ignore_periods])
# np.savetxt('Results/BoroConst20_LorenzData.dat', EstimationEconomy.LorenzData)

np.savetxt('Results/LorenzLong_hist.dat', EstimationEconomy.LorenzLong_hist)
np.savetxt('Results/ignore_periods.dat', [EstimationEconomy.ignore_periods])
np.savetxt('Results/LorenzData.dat', EstimationEconomy.LorenzData)

# np.savetxt('Results/Liquid_LorenzLong_hist.dat', EstimationEconomy.LorenzLong_hist)
# np.savetxt('Results/Liquid_ignore_periods.dat', [EstimationEconomy.ignore_periods])
# np.savetxt('Results/Liquid_LorenzData.dat', EstimationEconomy.LorenzData)
    
# np.savetxt('Results/IncUnemp86_LorenzLong_hist.dat', EstimationEconomy.LorenzLong_hist)
# np.savetxt('Results/IncUnemp86_ignore_periods.dat', [EstimationEconomy.ignore_periods])
# np.savetxt('Results/IncUnemp86_LorenzData.dat', EstimationEconomy.LorenzData)
    
#%%

IncUnemp86_BoroConst20_LorenzLong_hist = np.loadtxt('Results/IncUnemp86_BoroConst20_LorenzLong_hist.dat')
IncUnemp86_BoroConst20_ignore_periods = np.loadtxt('Results/IncUnemp86_BoroConst20_ignore_periods.dat')
IncUnemp86_BoroConst20_LorenzData = np.loadtxt('Results/IncUnemp86_BoroConst20_LorenzData.dat')

BoroConst20_LorenzLong_hist = np.loadtxt('Results/BoroConst20_LorenzLong_hist.dat')
BoroConst20_ignore_periods = np.loadtxt('Results/BoroConst20_ignore_periods.dat')
BoroConst20_LorenzData = np.loadtxt('Results/BoroConst20_LorenzData.dat')

IncUnemp86_LorenzLong_hist = np.loadtxt('Results/IncUnemp86_LorenzLong_hist.dat')
IncUnemp86_ignore_periods = np.loadtxt('Results/IncUnemp86_ignore_periods.dat')
IncUnemp86_LorenzData = np.loadtxt('Results/IncUnemp86_LorenzData.dat')

LorenzLong_hist = np.loadtxt('Results/LorenzLong_hist.dat')
ignore_periods = np.loadtxt('Results/ignore_periods.dat')
LorenzData = np.loadtxt('Results/LorenzData.dat')


#%%
LorenzSim_IncUnemp86_BoroConst20    = np.hstack((np.array(0.0),np.mean(np.array(IncUnemp86_BoroConst20_LorenzLong_hist)[EstimationEconomy.ignore_periods:,:],axis=0),np.array(1.0)))
LorenzSim_BoroConst20               = np.hstack((np.array(0.0),np.mean(np.array(BoroConst20_LorenzLong_hist)[EstimationEconomy.ignore_periods:,:],axis=0),np.array(1.0)))
LorenzSim_IncUnemp86                = np.hstack((np.array(0.0),np.mean(np.array(IncUnemp86_LorenzLong_hist)[EstimationEconomy.ignore_periods:,:],axis=0),np.array(1.0)))
LorenzSim                           = np.hstack((np.array(0.0),np.mean(np.array(LorenzLong_hist)[EstimationEconomy.ignore_periods:,:],axis=0),np.array(1.0)))

LorenzAxis = np.arange(101,dtype=float)
line1,=plt.plot(LorenzAxis,EstimationEconomy.LorenzData,'-k',linewidth=2,label='Data: Net Wealth')
line2,=plt.plot(LorenzAxis,LorenzSim,'--b',linewidth=1.5,label='Base model')
line3,=plt.plot(LorenzAxis,LorenzSim_BoroConst20,'--g',linewidth=2,label='BoroCnstArt = -20')
line4,=plt.plot(LorenzAxis,LorenzSim_IncUnemp86,'--c',linewidth=2,label='IncUnemp = 0.68')
line5,=plt.plot(LorenzAxis,LorenzSim_IncUnemp86_BoroConst20,'--r',linewidth=1.5,label='BoroCnstArt - 20, IncUnemp = 0.68')
plt.xlabel('Income percentile',fontsize=12)
plt.ylabel('Cumulative wealth share',fontsize=12)
plt.ylim([-0.05,1.0])
plt.legend(handles=[line1,line2,line3,line4,line5])
plt.show()

#%%

Liquid_LorenzLong_hist = np.loadtxt('Results/Liquid_LorenzLong_hist.dat')
Liquid_LorenzData = np.loadtxt('Results/Liquid_LorenzData.dat')

Liquid_LorenzSim  = np.hstack((np.array(0.0),np.mean(np.array(Liquid_LorenzLong_hist)[400:,:],axis=0),np.array(1.0)))

LorenzAxis = np.arange(101,dtype=float)
line1,=plt.plot(LorenzAxis,Liquid_LorenzData,'-k',linewidth=2,label='Data: Liquid Wealth')
line2,=plt.plot(LorenzAxis,Liquid_LorenzSim,'--m',linewidth=1.5,label='BoroCnstArt = 0, IncUnemp = 0.68')
plt.xlabel('Income percentile',fontsize=12)
plt.ylabel('Cumulative wealth share',fontsize=12)
plt.ylim([-0.05,1.0])
plt.legend(handles=[line1,line2])
plt.show()