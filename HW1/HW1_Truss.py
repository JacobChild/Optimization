#HW1_Truss.py
#Jacob Child
# %% Import needed packages 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from truss import *
# %% Problem description 
# 1.3 Constrained Truss Problem: Solve the ten-bar truss problem defined in D.2.2 of the book using an accessible optimizer. Code to analyze the truss is available in the resource repository: https://github.com/mdobook/resources/tree/main/exercises/tenbartruss
# Report and discuss the following:
# (a) the optimal mass and corresponding cross-sectional areas
# (b) a convergence plot: x-axis should be some measure of computational time (e.g., major iterations, function calls) on a linear scale, the y-axis should be some measure of convergence. If your solver gives you “first order optimality” that is ideal (we will learn what that means later), but other reasonable metrics can be used instead.
# (c) the number of function calls required to converge (functions calls from the truss function).
# (d) In 200 words or less describe what you learned.

# %% Background 
# The truss function takes in an array of cross-sectional areas and returns the total mass and an array of stresses for each truss member.
# The objective of the optimization is to minimize the mass of the structure, subject to the constraints that every segment does not yield in compression or tension.
# Constraint: The yield stress of all elements is 25 × 103 psi, except for member 9, which uses a stronger alloy with a yield stress of 75 × 103 psi.
# Bounds: Each element should have a cross-sectional area of at least 0.1 in2 for manufacturing reasons (bound constraint).

# %% Functions for setup (objective and constraints) 
TrussFunctionCalls = 0
def TrussMass(Af):
    global TrussFunctionCalls
    TrussFunctionCalls += 1 
    massf, _ = truss(Af)
    return massf

def g(Af):
    _, stressf = truss(Af)
    return yield_stresses - np.abs(stressf) 



# %% Setup 
E = 25.0 * 10**3 # psi
E_special = 75.0 * 10**3 # psi
A_min = 0.1 # in^2

yield_stresses = np.ones(10) * E
yield_stresses[8] = E_special

area_bounds = [(A_min, None)] * 10 # in^2

stress_constraints = [{'type':'ineq','fun':g}]

# %% Try to get some outputs 
def MyCallbackFunc(xk):
    massf, _ = truss(xk)
    MyMassTracker.append(massf)
    
# %% Optimization
MyMassTracker = []
x0 = np.ones(10)
res = minimize(TrussMass,x0,constraints=stress_constraints,bounds=area_bounds, callback = MyCallbackFunc)


# %% plot
plt.plot(range(2,res.njev+1), np.diff(MyMassTracker), marker = 'o')
plt.xlabel('Iteration')
plt.ylabel(r'Slope $(M_{i} - M_{i-1})$')
plt.xticks(np.arange(2, res.njev+1, step=1))
plt.savefig("Plots/FuncCallsVsSlope_truss.png")

# %% Mass plot 
plt.plot(range(1,res.njev+1), MyMassTracker, marker = 'o')
plt.xlabel('Iteration')
plt.ylabel(r'Mass (lbs)')
plt.xticks(np.arange(1, res.njev+1, step=1))
plt.savefig("Plots/FuncCallsVsMass_truss.png")

# %% Other outputs
# %%
col1 = range(1,11)
table = np.column_stack((col1, res.x)) 
print(" Member | Cross Sectional Area (in^2)")
print("--------|----------------------------")
for row in table:
    print(f"{row[0]:<7} | {row[1]:<7}")

# %%
