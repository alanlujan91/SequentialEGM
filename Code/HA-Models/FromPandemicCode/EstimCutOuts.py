
# -----------------------------------------------------------------------------
def calcEstimStats(Agents):
    '''
    Calculate the average LW/PI-ratio, total LW / total PI and the 20th, 40th, 
    60th, and 80th percentile points of the Lorenz curve for (liquid) wealth for 
    this set of Agents. Common use: Agents is either all agents in the economy, 
    or Agents with a common education type. 
    
    Parameters
    ----------
    Agents : [AgentType]
        List of AgentTypes in the economy.
        
    Returns
    -------
    avgLWPI : float 
        The weighted average of all Agents' LW/PI-ratio 
    LorenzPts : [float]
        The 20th, 40th, 60th, and 80th percentile points of the Lorenz curve for 
        (liquid) wealth for this set of Agents.
    '''
    aNrmAll = np.concatenate([ThisType.aNrmNow for ThisType in Agents])
    aLvlAll = np.concatenate([ThisType.aLvlNow for ThisType in Agents])
    pLvlAll = np.concatenate([ThisType.pLvlNow for ThisType in Agents])
    numAgents = 0
    for ThisType in Agents: 
        numAgents += ThisType.AgentCount
    weights = np.ones(numAgents) / numAgents      # just using equal weights for now

    # Lorenz points:
    # LorenzPts = 100* getLorenzShares(aLvlAll, weights=weights, percentiles = [0.2, 0.4, 0.6, 0.8] )
    LorenzPts = 100* getLorenzShares(aLvlAll, percentiles = [0.2, 0.4, 0.6, 0.8] )
    # Weighted average of LW/PI: 
    avgLWPI = np.dot(aNrmAll, weights) * 100
    # Total LW / total PI: 
    LWoPI = np.dot(aLvlAll, weights) / np.dot(pLvlAll, weights) * 100

    Stats = namedtuple("Stats", ["LorenzPts", "avgLWPI", "LWoPI"])

    return Stats(LorenzPts, avgLWPI, LWoPI) 
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def betasObjFunc(betas, spread, print_mode=False):
    '''
    Objective function for the estimation of discount factor distributions for the 
    three education groups. The groups differ in the centering of their discount 
    factor distributions, but have the same spread around the central value.
    
    Parameters
    ----------
    betas : [float]
        Central values of the discount factor distributions for each education
        level.
    spread : float
        Half the width of each discount factor distribution.
    print_mode : boolean, optional
        If true, statistics for each education level are printed. The default is False.

    Returns
    -------
    distance : float
        The distance of the estimation targets between those in the data and those
        produced by the model. 
    '''
    # # Set seed to ensure distance only changes due to different parameters 
    # random.seed(1234)

    beta_d, beta_h, beta_c = betas

    # # Overwrite the discount factor distribution for each education level with new values
    dfs_d = Uniform(beta_d-spread, beta_d+spread).approx(DiscFacCount)
    dfs_h = Uniform(beta_h-spread, beta_h+spread).approx(DiscFacCount)
    dfs_c = Uniform(beta_c-spread, beta_c+spread).approx(DiscFacCount)
    dfs = [dfs_d, dfs_h, dfs_c]

    # # Update discount factors of all agents 
    # for e in range(num_types):
    #     for b in range(DiscFacCount):
    #         TypeList[b+e*DiscFacCount].DiscFac = DiscFacDstns[e].X[b]
    #         TypeList[b+e*DiscFacCount].seed = n

    # Make a new list of types with updated discount factors 
    TypeListNew = []
    n = 0
    for e in range(num_types):
        for b in range(DiscFacCount):
            AgentCount = int(np.floor(AgentCountTotal*data_EducShares[e]*dfs[e].pmf[b]))
            ThisType = deepcopy(BaseTypeList[e])
            ThisType.AgentCount = AgentCount
            ThisType.DiscFac = dfs[e].X[b]
            ThisType.seed = n
            TypeListNew.append(ThisType)
            n += 1
    base_dict['Agents'] = TypeListNew

    AggDemandEconomy.agents = TypeListNew
    AggDemandEconomy.solve()

    AggDemandEconomy.reset()
    for agent in AggDemandEconomy.agents:
        agent.initializeSim()
        agent.AggDemandFac = 1.0
        agent.RfreeNow = 1.0
        agent.CaggNow = 1.0

    AggDemandEconomy.makeHistory()   
    AggDemandEconomy.saveState()   

    # Simulate each type to get a new steady state solution 
    # solve: done in AggDemandEconomy.solve(), initializeSim: done in AggDemandEconomy.reset() 
    # baseline_commands = ['solve()', 'initializeSim()', 'simulate()', 'saveState()']
    baseline_commands = ['simulate()', 'saveState()']
    multiThreadCommandsFake(TypeListNew, baseline_commands)
    
    Stats_d = calcEstimStats(TypeListNew[0:DiscFacCount])
    Stats_h = calcEstimStats(TypeListNew[DiscFacCount:2*DiscFacCount])
    Stats_c = calcEstimStats(TypeListNew[2*DiscFacCount:3*DiscFacCount])
    WealthShares = calcWealthShareByEd(TypeListNew)

    # Calculate distance from data moments
    sumSquares_d = np.sum((np.array(Stats_d.LorenzPts)-data_LorenzPts[0])**2) \
        + (Stats_d.avgLWPI-data_avgLWPI[0])**2
    sumSquares_h = np.sum((np.array(Stats_h.LorenzPts)-data_LorenzPts[1])**2) \
        + (Stats_h.avgLWPI-data_avgLWPI[1])**2
    sumSquares_c = np.sum((np.array(Stats_c.LorenzPts)-data_LorenzPts[2])**2) \
        + (Stats_c.avgLWPI-data_avgLWPI[2])**2
    sumSquares_ws = np.sum((WealthShares-data_WealthShares)**2)
    
    distance = np.sqrt(sumSquares_d+sumSquares_h+sumSquares_c+sumSquares_ws)

    # When testing, print stats by education level
    if print_mode:
        print('Lorenz shares Dropouts:')
        print(Stats_d.LorenzPts)
        print('Lorenz shares Highschoolers:')
        print(Stats_h.LorenzPts)
        print('Lorenz shares Collegegoers:')
        print(Stats_c.LorenzPts)
        print('Average LW/PI-ratios: D = ' + mystr(Stats_d.avgLWPI) + ' H = ' + mystr(Stats_h.avgLWPI) \
              + ' C = ' + mystr(Stats_c.avgLWPI)) 
        print('Distance = ' + mystr(distance))
        print('Total LW/Total PI (not targeted): D = ' + mystr(Stats_d.LWoPI) + ' H = ' + mystr(Stats_h.LWoPI) \
              + ' C = ' + mystr(Stats_c.LWoPI))
        print('Wealth Shares: D = ' + mystr(WealthShares[0]) + \
              ' H = ' + mystr(WealthShares[1]) + ' C = ' + mystr(WealthShares[2]))

    return distance 
# -----------------------------------------------------------------------------
