# filename: do_all.py

# Import the exec function
from builtins import exec
import sys 
import os

#%%
# Step 1: Estimation of the splurge factor: 
# This file replicates the results from section 3.1 in the paper, creates Figure 1 (in Target_AggMPCX_LiquWealth/Figures),
# and saves results in Target_AggMPCX_LiquWealth as .txt files to be used in the later steps. For the robustness checks, 
# the script also estimates the splurge factor for CRRA = 1 and 3.
print('Step 1: Estimating the splurge factor\n')
script_path = "Target_AggMPCX_LiquWealth/Estimation_BetaNablaSplurge.py"
#exec(open(script_path).read())
print('Concluded Step 1.\n\n')


#%%
# Step 2: Baseline results. Estimate the discount factor distributions and plot figure 2. This replicates results from section 3.4 in the paper. 
print('Step 2: Estimating discount factor distributions (this takes a while!)\n')
os.chdir('FromPandemicCode')
# Test done
#exec(open("EstimAggFiscalMAIN.py").read())
#exec(open("createLPfig.py").read()) # No argument -> create baseline figure
os.chdir('../')
print('Concluded Step 2.\n\n')


#%%
# Step 3: Robustness results. Estimate discount factor distributions for alternative values and plot figures 7 and 8. This replicates results from sections 5.1, 5.2, 5.3 and Appendix A. 
print('Step 3: Robustness results (note: this repeats step 2 five times)\n')
run_robustness_results = True  
if run_robustness_results:
    os.chdir('FromPandemicCode')
    # Order of input arguments: interest rate, risk aversion, replacement rate w/benefits, replacement rate w/o benefits
    #sys.argv = ['EstimAggFiscalMAINtest.py', '1.005']
    #exec(open("EstimAggFiscalMAINtest.py").read())
    #sys.argv = ['EstimAggFiscalMAINtest.py', '1.015']
    #exec(open("EstimAggFiscalMAINtest.py").read())
    #sys.argv = ['createLPfig.py', '1'] # Argument 1 -> create figure for different interest rates 
    #exec(open("createLPfig.py").read()) 
    
    #sys.argv = ['EstimAggFiscalMAINtest.py', '1.01', '1.0']
    #exec(open("EstimAggFiscalMAINtest.py").read())
    #sys.argv = ['EstimAggFiscalMAINtest.py', '1.01', '3.0']
    #exec(open("EstimAggFiscalMAINtest.py").read())
    #sys.argv = ['createLPfig.py', '2'] # Argument 2 -> create figure for different risk aversions 
    #exec(open("createLPfig.py").read()) 

# Test done
#    sys.argv = ['EstimAggFiscalMAIN.py', '1.01', '2.0', '0.3', '0.15']    
#    exec(open("EstimAggFiscalMAIN.py").read())
    os.chdir('../')
else:
    print('Skipping robustness results this time (see do_all.py line 32)')
print('Concluded Step 3.\n\n')

#%%
# Step 4: Comparing fiscal stimulus policies: This file replicates the results from section 4 in the paper, 
# creates Figures 3-6 (located in FromPandemicCode/Figures), creates tables (located in FromPandemicCode/Tables)
# and creates robustness results
print('Step 4: Comparing policies\n')
script_path = "AggFiscalMAIN.py"
os.chdir('FromPandemicCode')
#exec(open(script_path).read())
os.chdir('../')
print('Concluded Step 4. \n')