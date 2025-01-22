#HW1.py
#Jacob Child
#Jan  20th, 2025
#%% needed packages
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

#%% Pseudocode: 
# 1.2 Unconstrained Brachistochrone Problem: Solve the Brachistochrone Problem (with friction)
# using an unconstrained optimizer (like scipy.optimize). The problem is defined in the textbook in Appendix D.1.7. Complete the following:
# (a) Plot the optimal shape with n = 12 (10 design variables, the beginning and end should be fixed).
# (b) In a table, report the travel time between the two end points when n = 12. Don’t forget to put g = 9.81, the acceleration of gravity, back in.
# (c) Study the effect of increased problem dimensionality. Start with 4 points and double the dimension each time up to 128 (i.e., 4, 8, 16, 32, 64, 128). Plot and discuss the increase in computational expense with problem size. Include one plot that shows the number of total function calls for each dimension (i.e. 4, 8, etc.) and a second plot that calculates the wall time for each dimension.
# (d) In 200 words or less describe what you learned.
# Hint: Start with just 1 design variable (i.e. n = 3, which includes the two fixed endpoints) to make sure things are working correctly. Hint: When solving the higher-dimensional cases, it is more effective to start with the solution interpolated from a lower-dimensional case; this is called a warm start.

# Part a pseudocode: code up the objective function, have some way to change the number of design variables and make sure the starts and ends are fixed
#%%
def Brachistocrone(ysf):
    nf = np.size(ysf)
    # x will be fixed uniform spacing, y's are the design variables
    xsf = np.linspace(0,1,nf)
    # just in case ysf has a problem, make sure the first and last are fixed
    ysf[0] = 1
    ysf[-1] = 0
    # Givens
    mu_kf = 0.3 # coefficient of kinetic friction 
    gf = 9.81 #m/s^2 
    hf = ysf[0] - ysf[-1] # height difference
    # Terms 
    deltaxs = np.diff(xsf) 
    deltays = np.diff(ysf)
    TopTerm = np.sqrt(deltaxs**2 + deltays**2)
    BtmTerm1 = np.sqrt(hf - ysf - mu_kf*xsf)
    BtmTerm2 = np.sqrt(hf - ysf + mu_kf*xsf)
    # The summation only goes to n-1, so I need to delete some of the terms.
    FinalTerm = np.sum(TopTerm / (BtmTerm1[1:] + BtmTerm2[0:-1]))
    return FinalTerm

def WarmStartTerms(ysf, nf): 
    x_old = np.linspace(0,1,np.size(ysf))
    x_new = np.linspace(0,1,nf)
    return np.interp(x_new, x_old, ysf)



#%% First run through
# Starting Hint with one design variable 
N = 3
init_y = np.array([1, 0.5, 0])
my_bounds = (-2,2)
res3 = minimize(Brachistocrone, init_y)

# %% N =12 run through
res12 = minimize(Brachistocrone, np.linspace(1,0,12))
time_12 = res12.fun*(np.sqrt(2/9.81))
# %%
plt.plot(np.linspace(0,1,12), res12.x, marker = 'o')
plt.xlabel('x location (m)')
plt.ylabel('y location (m)')
plt.savefig('Plots/N12Shape_Brach.png')
# %% Function to time and run the optimization 
def TimeNMin(fxf, yof, nf):
    init_yf = WarmStartTerms(yof, nf)
    start_time = time.time()
    resf = minimize(fxf, init_yf)
    end_time = time.time()
    return end_time - start_time, resf

# %%
t3, res3_2 = TimeNMin(Brachistocrone, init_y, 3)
# %% Run all the Different sizes 
Ns = np.array([4, 8, 16, 32, 64, 128])
Times = np.zeros(np.size(Ns))
FuncCalls = np.zeros(np.size(Ns))
BallTimes = np.zeros(np.size(Ns))
Results_x = []
Results = []
init_y4 = np.linspace(1,0,4)
Times[0], res4 = TimeNMin(Brachistocrone, init_y4, Ns[0])
Results.append(res4)
BallTimes[0] = res4.fun*(np.sqrt(2/9.81))
FuncCalls[0] = Results[0].nfev
Results_x.append(Results[0].x)
# loop through the rest of the sizes
for i in range(1, np.size(Ns)):
    Times[i], resf = TimeNMin(Brachistocrone, Results[i-1].x, Ns[i])
    Results.append(resf)
    BallTimes[i] = resf.fun*(np.sqrt(2/9.81))
    FuncCalls[i] = Results[i].nfev
    Results_x.append(Results[i].x)

# %%
plt.plot(Ns, FuncCalls, marker = 'o')
# plt.title("Function Calls vs. N")
plt.xlabel("N")
plt.ylabel("Function Calls")
plt.savefig("Plots/FunctionCalls1_brach.png")
# %%
plt.plot(Ns, Times, marker = 'o')
plt.title("Computational Time vs. N")
plt.xlabel("N")
plt.ylabel("Time (s)")
plt.savefig("Plots/Time1_brach.png")
# %%
plt.plot(np.linspace(0,1,128), Results[-1].x)
plt.title("Final Path Shape")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.savefig("BrachistocroneFinalPath1.png")
# %%
plt.plot(Ns, BallTimes)
plt.title("Ball Time")
plt.xlabel("N")
plt.ylabel("Time (s)")
plt.savefig("BallTimes1.png")
# %%
table = np.column_stack((Ns, BallTimes)) 
print(" N  | Ball Time (s)")
print("----|--------------")
for row in table:
    print(f"{row[0]:<3} | {row[1]:<7}")
# %%
