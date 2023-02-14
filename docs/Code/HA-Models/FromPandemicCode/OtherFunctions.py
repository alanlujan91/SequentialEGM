import pickle
import os.path



def namestr(obj,namespace):
    return [name for name in namespace if namespace[name] is obj][0]

def saveAsPickle(name,obj,save_dir):
     with open(save_dir + name + '.csv', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)    

def saveAsPickleUnderVarName(obj,save_dir,scope):
    with open(save_dir + namestr(obj,scope) + '.csv', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
def loadPickle(filename,load_dir,scope):
    if os.path.isfile(load_dir + filename +'.csv'):
        SavedFile = open(load_dir + filename +'.csv', 'rb') 
        return pickle.load(SavedFile)
    else:
        return 0
    

def getSimulationDiff(simulation_base,simulation_alternative,simulation_variable):
    return simulation_alternative[simulation_variable]-simulation_base[simulation_variable]
 
def getSimulationPercentDiff(simulation_base,simulation_alternative,simulation_variable):
    SimDiff = getSimulationDiff(simulation_base,simulation_alternative,simulation_variable)
    return 100*SimDiff/simulation_base[simulation_variable]

def getStimulus(simulation_base,simulation_alternative,Gov_Spending):
    AddCons = getSimulationDiff(simulation_base,simulation_alternative,'AggCons')
    return  100*AddCons/Gov_Spending

def getNPVMultiplier(simulation_base,simulation_alternative,Gov_Spending):
    AddCons = getSimulationDiff(simulation_base,simulation_alternative,'NPV_AggCons')
    return  AddCons/Gov_Spending
