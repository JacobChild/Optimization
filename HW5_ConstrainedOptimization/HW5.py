#HW5.py
#Jacob Child
#March 3rd, 2025

#%% Packages 
#! .venv\Scripts\Activate.ps1
import numpy as np
from scipy.optimize import minimize
import sympy as sym
import matplotlib.pyplot as plt

# %% Problem 1
#! get TA help
s1_0 = 0
s2_0 = 0
sig1_0 = 0
sig2_0 = 0
s1_1 = 1
s2_1 = 1
sig1_1 = 1
sig2_1 = 1
x,y,sig1,s1,sig2,s2 = sym.symbols(('x,y,sig1,s1,sig2,s2'))
#if s1 = 0 and s2 = 0, then g1 and g2 are active
dldx = sym.Eq(2*(x+2*y-7) + 2*(2*x + y -5)*2 + 2* sig1 * x + sig2  , 0)
dldy = sym.Eq(2*(x+2*y-7)*2 + 2*(2*x+y-5) + 2*sig1*y + sig2  , 0)
dldsig1 = sym.Eq(x**2 + y**2 - 10 + s1**2*s1_0  , 0)
dlds1 = sym.Eq(2*sig1*s1*s1_0  , 0)
dldsig2 = sym.Eq(x+y-3+s2**2 * s2_0  , 0)
dlds2 = sym.Eq(2+sig2*s2 * s2_0  , 0)
result_s1s2_0 = sym.solve([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2], (x,y,sig1,s1,sig2,s2)) #not solvable I think
#if sig1 and sig2 = 0, then both are inactive 
dldx = sym.Eq(2*(x+2*y-7) + 2*(2*x + y -5)*2 + 2* sig1 * x*sig1_0 + sig2*sig2_0  , 0)
dldy = sym.Eq(2*(x+2*y-7)*2 + 2*(2*x+y-5) + 2*sig1*y *sig1_0 + sig2 *sig2_0 , 0)
dldsig1 = sym.Eq(x**2 + y**2 - 10 + s1**2  , 0)
dlds1 = sym.Eq(2*sig1*s1 *sig1_0 , 0)
dldsig2 = sym.Eq(x+y-3+s2**2  , 0)
dlds2 = sym.Eq(2+sig2*s2 * sig2_0  , 0)
result_sig1sig2_0 = sym.solve([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2], (x,y,sig1,s1,sig2,s2)) #not solvable
# if s1 = 0 and sig2 = 0, so g2 is inactive, and g1 is active
dldx = sym.Eq(2*(x+2*y-7) + 2*(2*x + y -5)*2 + 2* sig1 * x + sig2 *sig2_0 , 0)
dldy = sym.Eq(2*(x+2*y-7)*2 + 2*(2*x+y-5) + 2*sig1*y + sig2 *sig2_0 , 0)
dldsig1 = sym.Eq(x**2 + y**2 - 10 + s1**2*s1_0  , 0)
dlds1 = sym.Eq(2*sig1*s1*s1_0  , 0)
dldsig2 = sym.Eq(x+y-3+s2**2  , 0)
dlds2 = sym.Eq(2+sig2*s2 *sig2_0  , 0)
result_s1sig2_0 = sym.solve([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2], (x,y,sig1,s1,sig2,s2))
#if s2 = 0 and sig1 = 0 g1 is inactive and g2 is active 
dldx = sym.Eq(2*(x+2*y-7) + 2*(2*x + y -5)*2 + 2* sig1 *sig1_0* x + sig2  , 0)
dldy = sym.Eq(2*(x+2*y-7)*2 + 2*(2*x+y-5) + 2*sig1*y*sig1_0 + sig2  , 0)
dldsig1 = sym.Eq(x**2 + y**2 - 10 + s1**2  , 0)
dlds1 = sym.Eq(2*sig1*s1*sig1_0  , 0)
dldsig2 = sym.Eq(x+y-3+s2**2 * s2_0  , 0)
dlds2 = sym.Eq(2+sig2*s2 * s2_0  , 0)
result_sig1s2_0 = sym.solve([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2], (x,y,sig1,s1,sig2,s2))

# %% Problem 1 AI recommended
#! get TA help
# Define symbols
x, y, sig1, s1, sig2, s2 = sym.symbols('x y sig1 s1 sig2 s2')

# Define the Lagrangian
L = 2*(x + 2*y - 7)**2 + 2*(2*x + y - 5)**2 + sig1 * (x**2 + y**2 - 10) + sig2 * (x + y - 3)

# Stationarity conditions
dldx = sym.Eq(sym.diff(L, x), 0)
dldy = sym.Eq(sym.diff(L, y), 0)
dldsig1 = sym.Eq(x**2 + y**2 - 10, 0)
dldsig2 = sym.Eq(x + y - 3, 0)

# Complementary slackness conditions
dlds1 = sym.Eq(sig1 * (x**2 + y**2 - 10), 0)
dlds2 = sym.Eq(sig2 * (x + y - 3), 0)

# Solve the system of equations
result = sym.solve([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2], (x, y, sig1, s1, sig2, s2))

print(result)

# %% Problem 2 Functions
# Problem 2 Constrainted Optimization: choose three functions from https://en.wikipedia.org/wiki/Test_functions_for_optimization and with each function identify two constraints. Solve each and show a contour plot (with constraints, iteration histroy, and optimal point), and a convergence plot for each

def McCormick(xf): #actual min = f(-0.54719, -1.54719) = -1.9133
    x, y = xf
    return np.sin(x+y) + (x-y)**2 -1.5*x + 2.5*y + 1.0


def three_hump_camel(xf): #actual min = f(0,0) = 0
    x, y = xf
    return 2.0*x**2 - 1.05*x**4 + x**6 / 6.0 + x*y + y**2

def Himmelblau(xf): #actual min = 0 at (3,2), (-2.805118, 3.1312), (-3.77931, -3.283186), (3.584428, -1.848126)
    x, y = xf
    return (x**2 + y - 11.0)**2 + (x + y**2 - 7)**2

# All can be plot between -4 and 4 in x and y
def plotter(ptsf, funcf, bounds = (-4,4), highres = True, resolution=100, padding=0.5):
    if highres:
        # Determine the bounds based on the input points
        x_min, x_max = np.min(ptsf[:, 0]), np.max(ptsf[:, 0])
        y_min, y_max = np.min(ptsf[:, 1]), np.max(ptsf[:, 1])
        # Add some padding to the bounds
        x_min, x_max = x_min - padding, x_max + padding
        y_min, y_max = y_min - padding, y_max + padding
    else:
        x_min = y_min = bounds[0]
        x_max = y_max = bounds[1]
    
    # Create a meshgrid with specified resolution
    x_examp = np.linspace(x_min, x_max, resolution)
    y_examp = np.linspace(y_min, y_max, resolution)
    X_examp_mesh, Y_examp_mesh = np.meshgrid(x_examp, y_examp)
    Z_examp = funcf([X_examp_mesh, Y_examp_mesh])

    # Create the contour plot
    plt.contour(X_examp_mesh, Y_examp_mesh, Z_examp, levels=50, cmap='viridis')

    # Scatter plot the points in ptsf
    num_points = ptsf.shape[0]
    colors = plt.cm.Blues(np.linspace(0, 1, num_points))

    for i, (pt, color) in enumerate(zip(ptsf, colors)):
        plt.scatter(pt[0], pt[1], color=color, label=f'Point {i}')
        plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')

    plt.colorbar(label='Function value')
    plt.xlabel('x')
    plt.ylabel('y')
    # print('funcf: ', funcf)
    plt.title(f'Contour plot of {funcf.__name__}')
    # plt.legend()
    plt.show()
    
# %%
x0 = np.array([-2.,-2.])
point_tracker = []
def my_tracker(xk):
    point_tracker.append(np.copy(xk))


res1 = minimize(Himmelblau,x0, callback=my_tracker)
plotter(np.array(point_tracker), Himmelblau, highres=False)
#TODO: implement constraints, what is interesting for each function, check it can be solved, then, only after, update plotter to work with constraint functions
# %%
