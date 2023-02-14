# Import python tools
import sys
import os
import numpy as np
import random
from copy import deepcopy

# Import needed tools from HARK
from HARK.distribution import approxUniform
from HARK.utilities import getPercentiles, getLorenzShares
from HARK.parallel import multiThreadCommands
from HARK.estimation import minimizeNelderMead
from HARK.ConsumptionSaving.ConsIndShockModel import *
from HARK.cstwMPC.SetupParamsCSTW import init_infinite


# for plotting
import matplotlib.pyplot as plt

# Set key problem-specific parameters
TypeCount =  8      # Number of consumer types with heterogeneous discount factors
AdjFactor = 1.0     # Factor by which to scale all of MPCs in Table 9
T_kill = 400        # Don't let agents live past this age (expressed in quarters)
drop_corner = True  # If True, ignore upper left corner when calculating distance

# Set standard HARK parameter values (from stickyE paper)
base_params = deepcopy(init_infinite)
base_params['LivPrb']       = [0.995]       #from stickyE paper
base_params['Rfree']        = 1.015         #from stickyE paper
base_params['Rsave']        = 1.015         #from stickyE paper
base_params['Rboro']        = 1.025         #from stickyE paper
base_params['PermShkStd']   = [0.003**0.5]  #from stickyE paper
base_params['TranShkStd']   = [0.120**0.5]  #from stickyE paper
base_params['T_age']        = 400           # Kill off agents if they manage to achieve T_kill working years
base_params['AgentCount']   = 10000         # Number of agents per instance of IndShockConsType
base_params['pLvlInitMean'] = np.log(23.72) 
base_params['T_sim']        = 800


Parametrization = 'NOR_final'
if  Parametrization == 'NOR_final':
    base_params['LivPrb']       = [0.996]
    base_params['Rfree']        = 1.00496
    base_params['Rsave']        = 1.00496
    base_params['Rboro']        = 1.00496 #1.20**0.25
    base_params['pLvlInitMean'] = 1 
    base_params['UnempPrb']     = 0.044
    base_params['IncUnemp']     = 0.5
    base_params['PermShkStd']   = [(0.02/4)**0.5]
    base_params['TranShkStd']   = [(0.03*4)**0.5]
    base_params['BoroCnstArt']  = -0.8
    
    
  
guess_splurge_beta_nabla = [0.4,0.98,0.01]
#guess_splurge_beta_nabla = [0.3,0.95,0.01]
#guess_splurge_beta_nabla = [0.3,0.98,0.02]
#guess_splurge_beta_nabla = [0.3,0.98,0.01]
#guess_splurge_beta_nabla = [0.35,0.985,0.04]  
#guess_splurge_beta_nabla = [0.32,0.98,0.02]
    
    
# Define the MPC targets from Fagereng et al Table 9; element i,j is lottery quartile i, deposit quartile j
MPC_target_base = np.array([[1.047, 0.745, 0.720, 0.490],
                            [0.762, 0.640, 0.559, 0.437],
                            [0.663, 0.546, 0.390, 0.386],
                            [0.354, 0.325, 0.242, 0.216]])
MPC_target = AdjFactor*MPC_target_base

# Define the agg MPCx targets from Fagereng et al. Figure 2; first element is same-year response, 2nd element, t+1 response etcc
Agg_MPCX_target = np.array([0.5056845, 0.1759051, 0.1035106, 0.0444222, 0.0336616])

# Define the four lottery sizes, in thousands of USD; these are eyeballed centers/averages
# 5th element is used as rep. lottery win to get at aggregate MPC / MPCX 
lottery_size_USD = np.array([1.625, 3.3741, 7.129, 40.0, 7.129])
lottery_size_NOK = lottery_size_USD * (10/1.1) #in Fagereng et al it is mention that 1000 NOK = 110 USD
# We express lottery winnings relative to permanent income
lottery_size = lottery_size_NOK / (270/4) # Håkon estimated yearly permanent income to be 270k, i.e. 270/4 on a quarterly basis
RandomLotteryWin = True #if True, then the 5th element will be replaced with a random lottery size win draw from the 1st to 4th element for each agent

#%%

# Make several consumer types to be used during estimation

BaseType = KinkedRconsumerType(**base_params)
EstTypeList = []
for j in range(TypeCount):
    EstTypeList.append(deepcopy(BaseType))
    EstTypeList[-1](seed = j)
    
    
# Define the objective function

def FagerengObjFunc(SplurgeEstimate,center,spread,verbose=False,estimation_mode=True,target='AGG_MPC'):
    '''
    Objective function for the quick and dirty structural estimation to fit
    Fagereng, Holm, and Natvik's Table 9 results with a basic infinite horizon
    consumption-saving model (with permanent and transitory income shocks).

    Parameters
    ----------
    center : float
        Center of the uniform distribution of discount factors.
    spread : float
        Width of the uniform distribution of discount factors.
    verbose : bool
        When True, print to screen MPC table for these parameters.  When False,
        print (center, spread, distance).

    Returns
    -------
    distance : float
        Euclidean distance between simulated MPCs and (adjusted) Table 9 MPCs.
    '''
    
    # Give our consumer types the requested discount factor distribution
    beta_set = approxUniform(N=TypeCount,bot=center-spread,top=center+spread).X
    for j in range(TypeCount):
        EstTypeList[j](DiscFac = beta_set[j])

    # Solve and simulate all consumer types, then gather their wealth levels
    multiThreadCommands(EstTypeList,['solve()','initializeSim()','simulate()','unpackcFunc()'])
    WealthNow = np.concatenate([ThisType.aLvlNow for ThisType in EstTypeList])

    # Get wealth quartile cutoffs and distribute them to each consumer type
    quartile_cuts = getPercentiles(WealthNow,percentiles=[0.25,0.50,0.75])
    for ThisType in EstTypeList:
        WealthQ = np.zeros(ThisType.AgentCount,dtype=int)
        for n in range(3):
            WealthQ[ThisType.aLvlNow > quartile_cuts[n]] += 1
        ThisType(WealthQ = WealthQ)
       
    # Get wealth quartile cutoffs to plot Lorenz curve
    order = np.argsort(WealthNow)
    WealthNow_sorted = WealthNow[order]
    Lorenz_Data = getLorenzShares(WealthNow_sorted,percentiles=np.arange(0.01,1.00,0.01),presorted=True) 
    Lorenz_Data = np.hstack((np.array(0.0),Lorenz_Data,np.array(1.0)))  


    permNow = np.concatenate([ThisType.pLvlNow for ThisType in EstTypeList])   
    Wealth_Perm_Ratio = WealthNow / permNow
    order2 = np.argsort(Wealth_Perm_Ratio)
    Wealth_Perm_Ratio = Wealth_Perm_Ratio[order2]
    Wealth_Perm_Ratio_adj = Wealth_Perm_Ratio - Wealth_Perm_Ratio[0] # add lowest possible value to everyone
    Lorenz_Data_Adj = getLorenzShares(Wealth_Perm_Ratio_adj,percentiles=np.arange(0.01,1.00,0.01),presorted=True) 
    Lorenz_Data_Adj = np.hstack((np.array(0.0),Lorenz_Data_Adj,np.array(1.0)))  
        
    N_Quarter_Sim = 20; # Needs to be dividable by four
    N_Year_Sim = int(N_Quarter_Sim/4)
    N_Lottery_Win_Sizes = 5 # 4 lottery size bin + 1 representative one for agg MPCX

    
    EmptyList = [[],[],[],[],[]]
    MPC_set_list = [deepcopy(EmptyList),deepcopy(EmptyList),deepcopy(EmptyList),deepcopy(EmptyList)]
    MPC_Lists    = [deepcopy(MPC_set_list),deepcopy(MPC_set_list),deepcopy(MPC_set_list),deepcopy(MPC_set_list)]    
    # additional list for 5th Lottery bin, just need for elements for four years
    MPC_List_Add_Lottery_Bin = EmptyList
    
    
    for ThisType in EstTypeList:
        
        c_base = np.zeros((ThisType.AgentCount,N_Quarter_Sim))                        #c_base (in case of no lottery win) for each quarter
        c_base_Lvl = np.zeros((ThisType.AgentCount,N_Quarter_Sim))                    #same in levels
        c_actu = np.zeros((ThisType.AgentCount,N_Quarter_Sim,N_Lottery_Win_Sizes))    #c_actu (actual consumption in case of lottery win in one random quarter) for each quarter and lottery size
        c_actu_Lvl = np.zeros((ThisType.AgentCount,N_Quarter_Sim,N_Lottery_Win_Sizes))#same in levels
        a_actu = np.zeros((ThisType.AgentCount,N_Quarter_Sim,N_Lottery_Win_Sizes))    #a_actu captures the actual market resources after potential lottery win (last index) was added and c_actu deducted
        T_hist = np.zeros((ThisType.AgentCount,N_Quarter_Sim))
        P_hist = np.zeros((ThisType.AgentCount,N_Quarter_Sim)) 
            
        # LotteryWin is an array with AgentCount x 4 periods many entries; there is only one 1 in each row indicating the quarter of the Lottery win for the agent in each row
        # This can be coded more efficiently
        LotteryWin = np.zeros((ThisType.AgentCount,N_Quarter_Sim))   
        for i in range(ThisType.AgentCount):
            LotteryWin[i,random.randint(0,3)] = 1
            
        MPC_this_type = np.zeros((ThisType.AgentCount,N_Lottery_Win_Sizes,N_Year_Sim)) #Empty array, MPC for each Lottery size and agent
        
        for period in range(N_Quarter_Sim): #Simulate for 4 quarters as opposed to 1 year
            
            # Simulate forward for one quarter
            ThisType.simulate(1)           
            
            # capture base consumption which is consumption in absence of lottery win
            c_base[:,period] = ThisType.cNrmNow 
            c_base_Lvl[:,period] = c_base[:,period] * ThisType.pLvlNow
            
        
            for k in range(N_Lottery_Win_Sizes): # Loop through different lottery sizes
                
                Llvl = lottery_size[k]*LotteryWin[:,period]  #Lottery win occurs only if LotteryWin = 1 for that agent
                
                if RandomLotteryWin and k == 5:
                    for i in range(ThisType.AgentCount):
                        Llvl[i] = lottery_size[random.randint(0,3)]*LotteryWin[i,period]
                
                Lnrm = Llvl/ThisType.pLvlNow
                SplurgeNrm = SplurgeEstimate*Lnrm  #Splurge occurs only if LotteryWin = 1 for that agent
                

                        
                
                if period == 0:
                    m_adj = ThisType.mNrmNow + Lnrm - SplurgeNrm
                    c_actu[:,period,k] = ThisType.cFunc[0](m_adj) + SplurgeNrm
                    c_actu_Lvl[:,period,k] = c_actu[:,period,k] * ThisType.pLvlNow
                    a_actu[:,period,k] = ThisType.mNrmNow + Lnrm - c_actu[:,period,k] #save for next periods
                else:  
                    T_hist[:,period] = ThisType.TranShkNow 
                    P_hist[:,period] = ThisType.PermShkNow
                    for i_agent in range(ThisType.AgentCount):
                        if ThisType.TranShkNow[i_agent] == 1.0: # indicator of death
                            a_actu[i_agent,period-1,k] = np.exp(base_params['aNrmInitMean'])
                    m_adj = a_actu[:,period-1,k]*base_params['Rfree']/ThisType.PermShkNow + ThisType.TranShkNow + Lnrm - SplurgeNrm #continue with resources from last period
                    c_actu[:,period,k] = ThisType.cFunc[0](m_adj) + SplurgeNrm
                    c_actu_Lvl[:,period,k] = c_actu[:,period,k] * ThisType.pLvlNow
                    a_actu[:,period,k] = a_actu[:,period-1,k]*base_params['Rfree']/ThisType.PermShkNow + ThisType.TranShkNow + Lnrm - c_actu[:,period,k] 
                    
                if period%4 + 1 == 4: #if we are in the 4th quarter of a year
                    year = int((period+1)/4)
                    c_actu_Lvl_year = c_actu_Lvl[:,(year-1)*4:year*4,k]
                    c_base_Lvl_year = c_base_Lvl[:,(year-1)*4:year*4]
                    MPC_this_type[:,k,year-1] = (np.sum(c_actu_Lvl_year,axis=1) - np.sum(c_base_Lvl_year,axis=1))/(lottery_size[k])
                
        # Sort the MPCs into the proper MPC sets
        for q in range(4):
            these = ThisType.WealthQ == q
            for k in range(N_Lottery_Win_Sizes-1):  #only consider here 4 Lottery bins
                for y in range(N_Year_Sim):
                    MPC_Lists[k][q][y].append(MPC_this_type[these,k,y])
                    
        # sort MPCs for addtional Lottery bin
        for y in range(N_Year_Sim):
            MPC_List_Add_Lottery_Bin[y].append(MPC_this_type[:,4,y])
    

    # Calculate average within each MPC set
    simulated_MPC_means = np.zeros((N_Lottery_Win_Sizes-1,4,N_Year_Sim))
    for k in range(4):
        for q in range(4):
            for y in range(N_Year_Sim):
                MPC_array = np.concatenate(MPC_Lists[k][q][y])
                simulated_MPC_means[k,q,y] = np.mean(MPC_array)
                
    # Calculate aggregate MPC and MPCx
    simulated_MPC_mean_add_Lottery_Bin = np.zeros((N_Year_Sim))
    for y in range(N_Year_Sim):
        MPC_array = np.concatenate(MPC_List_Add_Lottery_Bin[y])
        simulated_MPC_mean_add_Lottery_Bin[y] = np.mean(MPC_array)
            
    # Calculate Euclidean distance between simulated MPC averages and Table 9 targets
    
    
    diff_MPC = simulated_MPC_means[:,:,0] - MPC_target
    if drop_corner:
        diff_MPC[0,0] = 0.0
    distance_MPC = np.sqrt(np.sum((diff_MPC)**2))   
      
    diff_Agg_MPC = simulated_MPC_mean_add_Lottery_Bin - Agg_MPCX_target
    distance_Agg_MPC = np.sqrt(np.sum((diff_Agg_MPC)**2))     
    distance_Agg_MPC_24 = np.sqrt(np.sum((diff_Agg_MPC[2:4])**2))
    distance_Agg_MPC_01 = np.sqrt(np.sum((diff_Agg_MPC[0:1])**2))
    
    target = 'AGG_MPC'
    if target == 'MPC':
        distance = distance_MPC
    elif target == 'AGG_MPC':
        distance = distance_Agg_MPC
    elif target == 'AGG_MPC_234':
        distance = distance_Agg_MPC_24
    elif target == 'MPC_plus_AGG_MPC_1':
        distance = distance_MPC + distance_Agg_MPC_01
        
        
    if verbose:
        print(simulated_MPC_means)
    else:
        print (SplurgeEstimate, center, spread, distance)
        
    if estimation_mode:
        return distance
    else:
        return [distance_MPC,distance_Agg_MPC,simulated_MPC_means,simulated_MPC_mean_add_Lottery_Bin,c_actu_Lvl,c_base_Lvl,LotteryWin,Lorenz_Data,Lorenz_Data_Adj,Wealth_Perm_Ratio]



#%% Conduct the estimation for beta, dist and splurge

f_temp = lambda x : FagerengObjFunc(x[0],x[1],x[2])
opt = minimizeNelderMead(f_temp, guess_splurge_beta_nabla, verbose=True)
print('Finished estimating')
print('Optimal splurge is ' + str(opt[0]) )
print('Optimal (beta,nabla) is ' + str(opt[1]) + ',' + str(opt[2]))

[distance_MPC,distance_Agg_MPC,simulated_MPC_means,simulated_MPC_mean_add_Lottery_Bin,c_actu_Lvl,c_base_Lvl,LotteryWin,Lorenz_Data,Lorenz_Data_Adj,Wealth_Perm_Ratio]=FagerengObjFunc(opt[0],opt[1],opt[2],estimation_mode=False,target='AGG_MPC')

print('Results for parametrization: ', Parametrization)
print('Agg MPC from first year to year t+4 \n', simulated_MPC_mean_add_Lottery_Bin, '\n')#%% Plot aggregate MPC and MPCX
print('Distance for Agg MPC is', distance_Agg_MPC, '\n')
print('Distance for MPC matrix is', distance_MPC, '\n')

import matplotlib.pyplot as plt
xAxis = np.arange(0,5)
line1,=plt.plot(xAxis,simulated_MPC_mean_add_Lottery_Bin,':b',linewidth=2,label='Model')
line2,=plt.plot(xAxis,Agg_MPCX_target,'-k',linewidth=2,label='Data')
plt.legend(handles=[line1,line2])
plt.title('Aggregate MPC from lottery win')
plt.xlabel('Year')
plt.show()


print('Lorenz shares at 20th, 40th, 60th and 80th percentile', Lorenz_Data_Adj[20], Lorenz_Data_Adj[40], Lorenz_Data_Adj[60], Lorenz_Data_Adj[80], '\n')
print('Last percentile with negative assets', np.argmin(Lorenz_Data), '% \n')
print('Percentile with zero cummulative assets', np.argwhere(Lorenz_Data>0)[0]-1, '% \n')
print('guess_splurge_beta_nabla = ', guess_splurge_beta_nabla)

LorenzAxis = np.arange(101,dtype=float)
line1,=plt.plot(LorenzAxis,Lorenz_Data_Adj,'-k',linewidth=2,label='Lorenz')
plt.xlabel('Income percentile',fontsize=12)
plt.ylabel('Cumulative wealth share',fontsize=12)
plt.legend(handles=[line1])
plt.show()



#%% Plot Surface

from mpl_toolkits import mplot3d

mesh_size = 4




def z_function(x, y, fixed_splurge):
    [distance_MPC,distance_Agg_MPC,simulated_MPC_means,simulated_MPC_mean_add_Lottery_Bin,c_actu_Lvl,c_base_Lvl,LotteryWin,Lorenz_Data,Lorenz_Data_Adj,Wealth_Perm_Ratio]=FagerengObjFunc(fixed_splurge,x,y,estimation_mode=False,target='AGG_MPC')
    return distance_Agg_MPC

beta = np.linspace(0.96, 0.99, mesh_size)
nabla = np.linspace(0.01, 0.06, mesh_size)

X, Y = np.meshgrid(beta, nabla)
Z = np.empty([mesh_size,mesh_size])
for i in range(mesh_size):
    for j in range(mesh_size):
        Z[j][i] = z_function(beta[i],nabla[j],0.32)
        
# Z2 = np.empty([mesh_size,mesh_size])
# for i in range(mesh_size):
#     for j in range(mesh_size):
#         Z2[j][i] = z_function(beta[i],nabla[j],0.35)    



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
minpoint = np.min(Z)
indexp = np.argmin(Z)
minbeta = np.asarray(X).reshape(-1)[indexp]
minnabla = np.asarray(Y).reshape(-1)[indexp]
ax.set_title(['splurge = 0.32, min=', minpoint, ' for beta = ', minbeta, ' and nabla = ', minnabla]);
plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z2, rstride=1, cstride=1,
#                 cmap='winter', edgecolor='none')
# minpoint = np.min(Z2)
# indexp = np.argmin(Z2)
# minbeta = np.asarray(X).reshape(-1)[indexp]
# minnabla = np.asarray(Y).reshape(-1)[indexp]
# ax.set_title(['splurge = 0.35, min=', minpoint, ' for beta = ', minbeta, ' and nabla = ', minnabla]);
# plt.show()

#%%
beta_reduced = beta[18:22]
nabla_reduced = nabla[0:22]
X_reduced, Y_reduced = np.meshgrid(beta_reduced, nabla_reduced)
Z_reduced = np.empty([22,4])
for i in range(4):
    for j in range(22):
        Z_reduced[j][i] = Z2[j][i+18]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X_reduced, Y_reduced, Z_reduced, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
ax.view_init(-140, 30)
        
#%%
for i in range(9):
    print(i+16)

