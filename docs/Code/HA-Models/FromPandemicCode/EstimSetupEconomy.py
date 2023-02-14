from EstimParameters import T_sim, init_dropout, init_highschool, init_college, init_ADEconomy, DiscFacDstns,\
     AgentCountTotal, EducShares, base_dict, figs_dir, num_max_iterations_solvingAD,\
     convergence_tol_solvingAD, UBspell_normal, num_base_MrkvStates
from EstimAggFiscalModel import AggFiscalType, AggregateDemandEconomy
from HARK.distribution import DiscreteDistribution
import numpy as np
from copy import deepcopy

base_dict_agg = deepcopy(base_dict)
    
# Make baseline types - for now only one type, might add more
num_types = 3
# This is not the number of discount factors, but the number of household types; in pandemic paper, there were different education groups

InfHorizonTypeAgg_d = AggFiscalType(**init_dropout)
InfHorizonTypeAgg_h = AggFiscalType(**init_highschool)
InfHorizonTypeAgg_c = AggFiscalType(**init_college)
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
    
# Make the overall list of types
TypeList = []
n = 0
myAggTotal = 0
for e in range(num_types):
    for b in range(DiscFacDstns[0].X.size):
        DiscFac = DiscFacDstns[e].X[b]
        AgentCount = int(np.floor(AgentCountTotal*EducShares[e]*DiscFacDstns[e].pmf[b]))
        ThisType = deepcopy(BaseTypeList[e])
        ThisType.AgentCount = AgentCount
        ThisType.DiscFac = DiscFac
        ThisType.seed = n
        TypeList.append(ThisType)
        n += 1
        myAggTotal += AgentCount 
AggDemandEconomy.agents = TypeList

print(myAggTotal)

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

