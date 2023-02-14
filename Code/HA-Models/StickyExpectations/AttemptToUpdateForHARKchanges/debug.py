# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:47:03 2020

@author: edmun
"""


#orig_count = 2500
#for agent in StickySOmarkovEconomy.agents:
#    agent.AgentCount = 2500
#    
for i in [3]:
#for i in range(4):

    Economy = deepcopy(StickySOmarkovEconomy)
    if i==0 or i==1: #sticky economies
        for agent in Economy.agents:
            agent(UpdatePrb = UpdatePrb)
    if i==2 or i==3: #frictionless economies
        for agent in Economy.agents:
            agent(UpdatePrb = 1.0)
    for n in range(num_agg_sims):
        Economy.Shk_idx = 0         
        for agent in Economy.agents:
            agent.pLvlNow = agent.pLvlNow_start
            agent.pLvlTrue = agent.pLvlTrue_start
            agent.pLvlErrNow = agent.pLvlErrNow_start
            agent.aLvlNow = agent.aLvlNow_start
            agent.aNrmNow = agent.aNrmNow_start
            agent.mLvlTrueNow = agent.mLvlTrueNow_start
            agent.mNrmNow = agent.mNrmNow_start
        if i==0 or i==2: # no tax cut economies
            Economy.MrkvNow_init = 0
            Economy.MrkvNow_hist[:] = 0
            Economy.makeAggShkHist_fixMrkv()
        if i==1 or i==3: # tax cut economies
            Economy.MrkvNow_init = 1
            Economy.MrkvNow_hist[:] = 0   
            # Initialize the Markov history and set up transitions
            MrkvNow_hist = np.zeros(Economy.act_T, dtype=int)
            cutoffs = np.cumsum(Economy.MrkvArray, axis=1)
            MrkvNow = Economy.MrkvNow_init
            t = 0
            draws = Uniform(seed=75+n).draw(N=T_after)
#                for t in range(draws.size):  # Add act_T_orig more periods
#                    MrkvNow_hist[t] = MrkvNow
#                    MrkvNow = np.searchsorted(cutoffs[MrkvNow, :], draws[t])
            MrkvNow_hist[0]=1
            K=1
            for k in range(K):
                MrkvNow_hist[k+1]=2
                MrkvNow_hist[K+1]=3
            
            Economy.MrkvNow_hist = MrkvNow_hist

            Economy.makeAggShkHist_fixMrkv()
        for k in range(len(Economy.agents)):
            Economy.agents[k].getEconomyData(Economy) # Have the consumers inherit relevant objects from the economy



    cLvlMean_hist = np.zeros(T_sim) + np.nan
    pLvlMean_hist = np.zeros(T_sim) + np.nan
    tranMean_hist = np.zeros(T_sim) + np.nan
    
#        pLvlMean_hist = np.zeros(T_sim) + np.nan
#        MrkvMean_hist = np.zeros(T_sim) + np.nan
    
    for t in range(T_sim):
        Economy.sow()       # Distribute aggregated information/state to agents
        Economy.cultivate() # Agents take action
        Economy.reap()      # Collect individual data from agents
        Economy.mill()      # Process individual data into aggregate data
        cLvl_new = np.concatenate([Economy.agents[j].cLvlNow for j in range(len(Economy.agents))])
        cLvlMean_hist[t] = np.mean(cLvl_new)
        pLvl_new = np.concatenate([np.array([Economy.agents[j].pLvlNow]) for j in range(len(Economy.agents))])
        pLvlMean_hist[t] = np.mean(pLvl_new)
        tran_new = np.concatenate([Economy.agents[j].TranShkNow for j in range(len(Economy.agents))])
        tranMean_hist[t] = np.mean(tran_new)
        
        tran_new = np.concatenate([np.array([Economy.agents[j].MrkvNow]) for j in range(len(Economy.agents))])
        tranMean_hist[t] = np.mean(tran_new)
        
        
    if i==0:
        cLvl_StickyNone = cLvlMean_hist
        pLvl_StickyNone = pLvlMean_hist
        tran_StickyNone = tranMean_hist
    if i==1:
        cLvl_StickyTaxCut = cLvlMean_hist
        pLvl_StickyTaxCut = pLvlMean_hist
        tran_StickyTaxCut = tranMean_hist
    if i==2:
        cLvl_FrictionlessNone = cLvlMean_hist
        pLvl_FrictionlessNone = pLvlMean_hist
        tran_FrictionlessNone = tranMean_hist
    if i==3:
        cLvl_FrictionlessTaxCut = cLvlMean_hist
        pLvl_FrictionlessTaxCut = pLvlMean_hist
        tran_FrictionlessTaxCut = tranMean_hist
        
        
plt.plot(cLvl_FrictionlessNone)
plt.plot(cLvl_FrictionlessTaxCut)
plt.plot(pLvl_FrictionlessNone)
plt.plot(pLvl_FrictionlessTaxCut)
plt.plot(tran_FrictionlessNone)
plt.plot(tran_FrictionlessTaxCut)
#plt.plot(pLvl_FrictionlessNone*tran_FrictionlessNone)
#plt.plot(pLvl_FrictionlessTaxCut*tran_FrictionlessTaxCut)
            