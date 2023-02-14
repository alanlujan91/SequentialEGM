# Import python tools
import sys
import os
import numpy as np
import random
from copy import deepcopy

# Import needed tools from HARK
from HARK.utilities import approxUniform
from HARK.utilities import getPercentiles
from HARK.parallel import multiThreadCommands
from HARK.estimation import minimizeNelderMead
from HARK.ConsumptionSaving.ConsIndShockModel import *
from HARK.cstwMPC.SetupParamsCSTW import init_infinite

# Set key problem-specific parameters
TypeCount =  8      # Number of consumer types with heterogeneous discount factors
AdjFactor = 1.0     # Factor by which to scale all of MPCs in Table 9
T_kill = 400        # Don't let agents live past this age (expressed in quarters)
Splurge = 1.2775    # Consumers automatically spend this amount of any lottery prize
do_secant = True    # If True, calculate MPC by secant, else point MPC
drop_corner = True  # If True, ignore upper left corner when calculating distance

# Set standard HARK parameter values
base_params = deepcopy(init_infinite)
base_params['LivPrb']       = [0.995]                       #from stickyE paper
base_params['Rfree']        = 1.015                         #from stickyE paper
base_params['PermShkStd']   = [0.003**0.5]                  #from stickyE paper
base_params['TranShkStd']   = [0.120**0.5]                  #from stickyE paper
base_params['T_age']        = 400           # Kill off agents if they manage to achieve T_kill working years
base_params['AgentCount']   = 10000         # Number of agents per instance of IndShockConsType
base_params['pLvlInitMean'] = np.log(23.72) # From Table 1, in thousands of USD (Q-I: where is this from?)

# T_sim needs to be long enough to reach the ergodic distribution
base_params['T_sim'] = 800
# Define the MPC targets from Fagereng et al Table 9; element i,j is lottery quartile i, deposit quartile j
MPC_target_base = np.array([[1.047, 0.745, 0.720, 0.490],
                            [0.762, 0.640, 0.559, 0.437],
                            [0.663, 0.546, 0.390, 0.386],
                            [0.354, 0.325, 0.242, 0.216]])
MPC_target = AdjFactor*MPC_target_base

# Define the agg MPCx targets from Fagereng et al. Figure 2; first element is same-year response, 2nd element, t+1 response etcc
Agg_MPCX_target = np.array([0.5056845, 0.1759051, 0.1035106, 0.0444222, 0.0336616])

# Define the four lottery sizes, in thousands of USD; these are eyeballed centers/averages
lottery_size = np.array([1.625, 3.3741, 7.129, 40.0])

#%% Checking when ergodic distribution is reached

def CheckErgodicDistribution(CheckType,Ergodic_Tol):
    '''
    This function checks, for an instance of the IndShockConsumerType, how many
    periods need to be simulated until the ergodic distribution of wealth has 
    been reached. The ergodic distribution is assumed to be reached when the 
    percentage change in mean and std of the wealth distribution is smaller 
    than Ergodic_Tol.

    Inputs
    ----------
    CheckType : IndShockConsumerType
        Instance of cons type which is simulated.
    Ergodic_Tol : float
        Ergodic distribution is reached when the percentage difference in 
        mean and std of the wealth distribution does not change by more than 
        Ergodic_Tol from one period to the next.

    Returns
    -------
    Text output
    '''

    CheckType.track_vars = ['aNrmNow']
    CheckType.solve()
    CheckType.initializeSim()
    CheckType.simulate()
    
    Mean_aNrmNow = np.mean(CheckType.aNrmNow_hist,axis=1)
    Std_aNrmNow = np.std(CheckType.aNrmNow_hist,axis=1) 
    Perc_Change_Mean_aNrmNow = 100*abs(Mean_aNrmNow[1:CheckType.T_sim] - Mean_aNrmNow[0:CheckType.T_sim-1])/Mean_aNrmNow[1:CheckType.T_sim]
    Perc_Change_Std_aNrmNow = 100*abs(Std_aNrmNow[1:CheckType.T_sim] - Std_aNrmNow[0:CheckType.T_sim-1])/Std_aNrmNow[1:CheckType.T_sim]
    
    
    Periods_Above_Tol = np.argwhere(Perc_Change_Mean_aNrmNow>Ergodic_Tol)+2
    LastPeriod_Above_Tol = int(max(Periods_Above_Tol))
    if LastPeriod_Above_Tol < CheckType.T_sim-50: # Last period should at least be 50 periods before end
        print('The simulation required ', LastPeriod_Above_Tol, ' periods for the change in mean wealth to become permanently smaller than ', Ergodic_Tol, '%.')
    else:
        print('The simulation never reached a stable mean wealth value below the imposed tolerance of ', Ergodic_Tol, '%.')
      
        
    Periods_Above_Tol = np.argwhere(Perc_Change_Std_aNrmNow>Ergodic_Tol)+2
    LastPeriod_Above_Tol = int(max(Periods_Above_Tol)) 
    if LastPeriod_Above_Tol < CheckType.T_sim-50:
        print('The simulation required ', LastPeriod_Above_Tol, ' periods for the change in the standard deviation of wealth to become permanently smaller than ', Ergodic_Tol, '%.')
    else:
        print('The simulation never reached a stable standard deviation of wealth value below the imposed tolerance of ', Ergodic_Tol, '%.')
      

CheckType = IndShockConsumerType(**base_params)
CheckType(DiscFac = 0.96)  # Check only for center Disc Fac
CheckType(T_sim = 1000)
CheckErgodicDistribution(CheckType,2)

#%%

# Make several consumer types to be used during estimation

BaseType = IndShockConsumerType(**base_params)
EstTypeList = []
for j in range(TypeCount):
    EstTypeList.append(deepcopy(BaseType))
    EstTypeList[-1](seed = j)
    
    
# Define the objective function

def FagerengObjFunc(SplurgeEstimate,center,spread,verbose=False):
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
    beta_set = approxUniform(N=TypeCount,bot=center-spread,top=center+spread)[1]
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
       
        
    N_Quarter_Sim = 16; # Needs to be dividable by four
    N_Year_Sim = int(N_Quarter_Sim/4);

    # Keep track of MPC sets in lists of lists of arrays
    MPC_set_list = [ [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]] ]
    
    EmptyList = [[],[],[],[]]
    MPC_set_list = [deepcopy(EmptyList),deepcopy(EmptyList),deepcopy(EmptyList),deepcopy(EmptyList)]
    MPC_Lists = [deepcopy(MPC_set_list),deepcopy(MPC_set_list),deepcopy(MPC_set_list),deepcopy(MPC_set_list)]    
    
    
    for ThisType in EstTypeList:
        
        c_base = np.zeros((ThisType.AgentCount,N_Quarter_Sim))      #c_base (in case of no lottery win) for each quarter
        c_base_Lvl = np.zeros((ThisType.AgentCount,N_Quarter_Sim))  #same in levels
        c_actu = np.zeros((ThisType.AgentCount,N_Quarter_Sim,4))    #c_actu (actual consumption in case of lottery win in one random quarter) for each quarter and lottery size
        c_actu_Lvl = np.zeros((ThisType.AgentCount,N_Quarter_Sim,4))#same in levels
        a_actu = np.zeros((ThisType.AgentCount,N_Quarter_Sim,4))    #a_actu captures the actual market resources after potential lottery win (last index) was added and c_actu deducted
        T_hist = np.zeros((ThisType.AgentCount,N_Quarter_Sim))
        P_hist = np.zeros((ThisType.AgentCount,N_Quarter_Sim)) 
            
         # LotteryWin is an array with AgentCount x 4 periods many entries; there is only one 1 in each row indicating the quarter of the Lottery win for the agent in each row
        LotteryWin = np.zeros((ThisType.AgentCount,N_Quarter_Sim))   
        for i in range(ThisType.AgentCount):
            LotteryWin[i,random.randint(0,3)] = 1
            
        MPC_this_type = np.zeros((ThisType.AgentCount,4,int(N_Quarter_Sim/4))) #Empty array, MPC for each Lottery size and agent
        
        for period in range(N_Quarter_Sim): #Simulate for 4 quarters as opposed to 1 year
            
            # Simulate forward for one quarter
            ThisType.simulate(1)           
            
            # capture base consumption which is consumption in absence of lottery win
            c_base[:,period] = ThisType.cNrmNow 
            c_base_Lvl[:,period] = c_base[:,period] * ThisType.pLvlNow
            
        
            for k in range(4): # Loop through different lottery sizes
                
                Llvl = lottery_size[k]*LotteryWin[:,period]  #Lottery win occurs only if LotteryWin = 1 for that agent
                Lnrm = Llvl/ThisType.pLvlNow
                SplurgeNrm = SplurgeEstimate/ThisType.pLvlNow*LotteryWin[:,period]  #Splurge occurs only if LotteryWin = 1 for that agent
                
                if period == 0:
                    m_adj = ThisType.mNrmNow + Lnrm - SplurgeNrm
                    c_actu[:,period,k] = ThisType.cFunc[0](m_adj) + SplurgeNrm
                    c_actu_Lvl[:,period,k] = c_actu[:,period,k] * ThisType.pLvlNow
                    a_actu[:,period,k] = ThisType.mNrmNow + Lnrm - c_actu[:,period,k] #save for next periods
                else:  
                    T_hist[:,period] = ThisType.TranShkNow 
                    P_hist[:,period] = ThisType.PermShkNow
                    for i_agent in range(ThisType.AgentCount):
                        if ThisType.TranShkNow[i_agent] == 1.0:
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
            for k in range(4):
                for y in range(4):
                    MPC_Lists[k][q][y].append(MPC_this_type[these,k,y])

    # Calculate average within each MPC set
    simulated_MPC_means = np.zeros((4,4,N_Year_Sim))
    for k in range(4):
        for q in range(4):
            for y in range(4):
                MPC_array = np.concatenate(MPC_Lists[k][q][y])
                simulated_MPC_means[k,q,y] = np.mean(MPC_array)
            
    # Calculate Euclidean distance between simulated MPC averages and Table 9 targets
    diff = simulated_MPC_means[:,:,0] - MPC_target
    if drop_corner:
        diff[0,0] = 0.0
    distance = np.sqrt(np.sum((diff)**2))
    if verbose:
        print(simulated_MPC_means)
    else:
        #print (center, spread, distance)
        print (SplurgeEstimate, distance)
    #return distance
    return [distance,simulated_MPC_means,c_actu_Lvl,c_base_Lvl,LotteryWin,T_hist,P_hist]


#%% Test function
beta_dist_estimate = [0.9868607230094457,0.006068381572212068]
[distance,simulated_MPC_means,c_actu_Lvl,c_base_Lvl,LotteryWin,T_hist,P_hist]=FagerengObjFunc(Splurge,beta_dist_estimate[0],beta_dist_estimate[1])
print('MPC in first year \n', simulated_MPC_means[:,:,0],'\n')
print('MPCX in year t+1 \n', simulated_MPC_means[:,:,1],'\n')
print('MPCX in year t+2 \n', simulated_MPC_means[:,:,2],'\n')
print('MPCX in year t+3 \n', simulated_MPC_means[:,:,3],'\n')

## Output for debugging purposes
# print('Difference between actual and base consumption: \n',c_actu_Lvl[0:5,:,2]-c_base_Lvl[0:5,:],'\n')
# print('Lottery win indicator: \n', LotteryWin[0:5,:],'\n')

# x=(c_actu_Lvl[:,1,0]-c_base_Lvl[:,1])>0

#%% Plot the evolution of MPC and MPCX

import matplotlib.pyplot as plt
for lottSize in range(4):
    plt.subplot(2,2,lottSize+1)
    for q in range(4):
        labStr = "Wealth Q=" + str(q)
        plt.plot(simulated_MPC_means[lottSize,q,:], label=labStr)
        plt.xticks(ticks=range(4))
    plt.title('Lottery size = %d' %lottSize)
plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.legend(loc='best')
plt.show()

#%% Conduct the estimation

beta_dist_estimate = [0.9868607230094457,0.006068381572212068]
f_temp = lambda x : FagerengObjFunc(x,beta_dist_estimate[0],beta_dist_estimate[1])
SplurgeEstimateStart = np.array([0.7])
opt_splurge = minimizeNelderMead(f_temp, SplurgeEstimateStart, verbose=True)
print('Finished estimating for scaling factor of ' + str(AdjFactor) + ' and (beta,nabla) of ' + str(beta_dist_estimate))
print('Optimal splurge is ' + str(opt_splurge) )
# REsult is 1.2775
# distance is 0.7057710207647322