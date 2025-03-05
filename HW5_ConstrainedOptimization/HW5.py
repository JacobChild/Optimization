#HW5.py
#Jacob Child
#March 3rd, 2025

#%% Packages 
#! .venv\Scripts\Activate.ps1
import numpy as np
from scipy.optimize import minimize, fsolve
import matplotlib.pyplot as plt
import jax
import copy
# %% Problem 1

def res_talker(xf, stringer):
    x, y, sig1, s1, sig2, s2 = xf
    print(stringer)
    print('x: ', x)
    print('y: ', y)
    print('sig1: ', sig1)
    print('s1: ', s1)
    print('sig2: ', sig2)
    print('s2: ', s2)

    
    
#if s1 = 0 and s2 = 0, then g1 and g2 are active
def dL_s1_0_s2_0(xf):
    x, y, sig1, s1, sig2, s2 = xf
    s1 = 0
    s2 = 0
    dldx = 10*x + 8*y -34 + 2*sig1*x + sig2 
    dldy = 8*x + 10*y -38 + 2*sig1*y + sig2 
    dldsig1 = x**2 + y**2 - 10 + s1**2
    dlds1 = 2 * sig1 * s1
    dldsig2 = x + y - 3 + s2**2 
    dlds2 = 2 * sig2 * s2  
    return np.array([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2])

def dL_sig1_0_sig2_0(xf):
    x, y, sig1, s1, sig2, s2 = xf
    sig1 = 0
    sig2 = 0
    dldx = 10*x + 8*y -34 + 2*sig1*x + sig2 
    dldy = 8*x + 10*y -38 + 2*sig1*y + sig2 
    dldsig1 = x**2 + y**2 - 10 + s1**2
    dlds1 = 2 * sig1 * s1
    dldsig2 = x + y - 3 + s2**2 
    dlds2 = 2 * sig2 * s2 
    return np.array([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2])

def dL_s1_0_sig2_0(xf):
    x, y, sig1, s1, sig2, s2 = xf
    s1 = 0
    sig2 = 0
    dldx = 10*x + 8*y -34 + 2*sig1*x + sig2 
    dldy = 8*x + 10*y -38 + 2*sig1*y + sig2 
    dldsig1 = x**2 + y**2 - 10 + s1**2
    dlds1 = 2 * sig1 * s1
    dldsig2 = x + y - 3 + s2**2 
    dlds2 = 2 * sig2 * s2 
    return np.array([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2])

def dL_sig1_0_s2_0(xf):
    x, y, sig1, s1, sig2, s2 = xf
    sig1 = 0
    s2 = 0
    dldx = 10*x + 8*y -34 + 2*sig1*x + sig2 
    dldy = 8*x + 10*y -38 + 2*sig1*y + sig2 
    dldsig1 = x**2 + y**2 - 10 + s1**2
    dlds1 = 2 * sig1 * s1 
    dldsig2 = x + y - 3 + s2**2 
    dlds2 = 2 * sig2 * s2  
    return np.array([dldx, dldy, dldsig1, dlds1, dldsig2, dlds2])

x0 = np.zeros(6)
r1 = fsolve(dL_s1_0_s2_0, x0)
res_talker(r1, 'dL_s1_0_s2_0') #neg sig1, not feasible

x0 = np.zeros(6)
r1 = fsolve(dL_sig1_0_sig2_0, x0)
res_talker(r1, 'dL_sig1_0_sig2_0') #neg sig2, not feasible

x0 = np.zeros(6)
r1 = fsolve(dL_s1_0_sig2_0, x0)
res_talker(r1, 'dL_s1_0_sig2_0') #sig2 blew up, but I'm forcing it to zero, so it doesn't matter. Everything else is feasible. ANSWER

x0 = np.zeros(6)
r1 = fsolve(dL_sig1_0_s2_0, x0)
res_talker(r1, 'dL_sig1_0_s2_0') #sig1 blew up, but same as above, however, sig1 is zero *and* s1 came out to be zero, that is infeasible


# %% Problem 2 Functions
# Problem 2 Constrainted Optimization: choose three functions from https://en.wikipedia.org/wiki/Test_functions_for_optimization and with each function identify two constraints. Solve each and show a contour plot (with constraints, iteration histroy, and optimal point), and a convergence plot for each

# All can be plot between -4 and 4 in x and y
def plotter(ptsf, funcf, bounds = (-4.,4.), highres = True, flag = True, resolution=100, padding=0.5):
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
        #if in the bounds
        # if np.any(np.abs(ptsf) < bounds[1]):
        #     print('turning on text')
        #     plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')

    plt.colorbar(label='Function value')
    plt.xlabel('x')
    plt.ylabel('y')
    # print('funcf: ', funcf)
    plt.title(f'Contour plot of {funcf.__name__}')
    #check if the abs of any of the ptsf values are greater than the bounds, if so, set plt.xlim and ylim to bounds 
    if np.any(np.abs(ptsf) > bounds[1]) and flag:
        print('a point (or more) was out of plotting bounds')
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[0], bounds[1])
    else: 
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    # plt.legend()
    plt.show()
    
def mesher(funcf, bounds=(-4,4), resolution = 100):
    x_min = y_min = bounds[0]
    x_max = y_max = bounds[1]
    # Create a meshgrid with specified resolution
    x_examp = np.linspace(x_min, x_max, resolution)
    y_examp = np.linspace(y_min, y_max, resolution)
    X_examp_mesh, Y_examp_mesh = np.meshgrid(x_examp, y_examp)
    Z_examp = funcf([X_examp_mesh, Y_examp_mesh])
    return X_examp_mesh, Y_examp_mesh, Z_examp
    
def convg_plotter(xf, funcf):
    best = funcf(xf[-1])
    tracker = []
    for x in xf:
        fx = funcf(x)
        tracker.append(np.abs(best - fx))
    plt.semilogy(np.array(tracker))
    plt.xlabel('Iteration')
    plt.ylabel('abs(f* - f(x))')
    plt.title(f'Convergence of Constrained {funcf.__name__}')
    plt.show()
    
def convg_plotter2(xf, funcf):
    gradf = jax.grad(funcf)
    tracker = []
    for x in xf:
        tracker.append(np.linalg.norm(gradf(x)))
    plt.semilogy(np.array(tracker))
    plt.xlabel('Iteration')
    plt.ylabel('Gradient of f(x)')
    plt.title(f'Convergence of Constrained {funcf.__name__}')
    plt.show()

# %% Write a basic constrained gradient-based optimizer using a penalty method. Test your method out on your three functions and constraints you identified in question 5.2 above. In a table, compare the solution accuracy and convergence efficiency of your own method against the optimizer you used in 5.2. Briefly describe what you learned in 400 words or less.
# Quadratic penalty method
#equality: fhat(x) = f(x) + mu/2 * sum(h_i(x)^2)
#inequality:fhat = f(x) + mu/2 * sum(max(0,g(x))^2)
#combined fhat = f(x) + mu_h/2 * sum(h(x)^2) + mu_g/2 * sum(max(0, g(x))^2)
def quad_penalty(x0f, funcf, hf, gf, mu_h=0.2, mu_g=0.2, rho = 1.2, tol=1e-4, max_iter = 100): #AI created from my pseudocode, then I went line by line to understand etc
    xstars = []
    all_points = []
    x = x0f
    all_points.append(x)
        
    #while not converged
    for _ in range(max_iter):
        inner_point_tracker = []
        inner_point_tracker.append(x)
        #new fhat objective func
        def fhat(x):
            # print(f"h(x): {hf(x)}, g(x): {gf(x)}")
            return funcf(x) + mu_h/2 * np.sum(hf(x)**2) + mu_g/2 * np.sum(np.maximum(0, -gf(x))**2) #I made gf(x) negative because this method says that negative is feasible and positive isn't, but scipy wants positive is feasible
        #solve
        res = minimize(fhat, x, callback=inner_point_tracker.append)
        x = res.x
        xstars.append(x)
        temp1 = copy.deepcopy(inner_point_tracker)
        all_points.extend(temp1)
        #converged?
        if np.linalg.norm(hf(x)) < tol and np.all(gf(x) <= tol):
            #hf(x) is normed as "eq constraints should be exactly zero, so the norm is used to measure how close they are to zero"
            #np.all(gf(x) <= tol) "checks if all ineq const are ... w/in tol... should be less than or = to zero, so np.all checks if all are satisfied"
            print('converged')
            return np.array(all_points) #xstars
        #update params
        mu_h *= rho 
        mu_g *= rho 
        
    print('max_iters reached')
    return np.array(all_points) #xstars
#! somehow all_points isn't actually returning all_points

def summarizer(xf_scipy, xf_mine, funcf):
    print(f'Summary stats for {funcf.__name__}')
    print('scipy iters: ', xf_scipy.shape[0])
    print('My penalty iters: ', xf_mine.shape[0])
    fx_scipy = funcf(xf_scipy[-1])
    fx_mine = funcf(xf_mine[-1])
    print('Percent f(x) off: ', (fx_scipy - fx_mine) / fx_scipy * 100, '%' )
    print('Distance off: ', np.linalg.norm(xf_scipy[-1] - xf_mine[-1]))
    
# %% McCormick
def McCormick(xf): #actual min = f(-0.54719, -1.54719) = -1.9133
    x, y = xf
    return jax.numpy.sin(x+y) + (x-y)**2 -1.5*x + 2.5*y + 1.0
# can I force it to end up around (1.5,3) is that another local min type thing? or (1.5,0) on that little saddle looking thing? 
def g_McCormick_1(xf):
    x, y = xf
    # in the circle centered at 1.5,1 with a radius of 2
    return  2. **2 - ((x-1.5)**2 + (y-1)**2) #positive means feasible 
def g_McCormick_2(xf):
    x, y = xf
    # for fun, a sin(x) line that goes through the circle (equality constraint)
    return y - (np.sin(x) + 1) # if zero, means feasible 

McCormick_constraints = [{'type': 'ineq', 'fun': g_McCormick_1},
                         {'type': 'eq', 'fun': g_McCormick_2}]
x0_Mc = np.array([-2., -2.])
point_tracker = []
point_tracker.append(x0_Mc)

def my_tracker(xk):
    point_tracker.append(np.copy(xk))

#solve/minimize 
res_mc = minimize(McCormick,x0_Mc, callback=my_tracker, constraints = McCormick_constraints)
#using my quad_penalty method 
res_my_mc = quad_penalty(x0_Mc, McCormick,g_McCormick_2, g_McCormick_1, mu_g = .8, rho=1.5)
summarizer(np.array(point_tracker), res_my_mc, McCormick)

#Plotting
X_mc, Y_mc, Z_mc = mesher(McCormick, bounds = (-3,3))
plt.contour(X_mc, Y_mc, Z_mc, levels=50, cmap = 'viridis') #background contour
#constraints
mc_circle = plt.Circle((1.5,1),2, color = 'g', linestyle = '--', label = 'Circle Constraint', alpha = 0.5)
plt.gca().add_artist(mc_circle)
mc_xg_vals = np.linspace(-3,3)
mc_yg_vals = np.sin(mc_xg_vals) + 1 
plt.plot(mc_xg_vals, mc_yg_vals, 'r--', label = 'Sin line constraint')


res_pts = np.array(point_tracker)
num_points = res_pts.shape[0]
num_points2 = res_my_mc.shape[0]
colors = plt.cm.Blues(np.linspace(0, 1, num_points))
colors2 = plt.cm.Oranges(np.linspace(0, 1, num_points2))
# plt.plot(res_pts)
for i, (pt, color) in enumerate(zip(res_pts, colors)):
    plt.scatter(pt[0], pt[1], color=color, edgecolor='black')
    plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')
    
for i, (pt, color) in enumerate(zip(res_my_mc, colors2)):
    plt.scatter(pt[0], pt[1], color=color, edgecolor='black')
    plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')

plt.colorbar(label='Function value')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('McCormick Minimization with constraints')
plt.legend()
plt.show()

convg_plotter2(res_pts, McCormick)


# %% Three Hump Camel 
def three_hump_camel(xf): #actual min = f(0,0) = 0
    x, y = xf
    return 2.0*x**2 - 1.05*x**4 + x**6 / 6.0 + x*y + y**2

#if x0 is np.array([-2., -2.]) I end up in the left hump local minimum. Lets see if I can constrain it out of there
def g_threehump_1(xf):
    x, y = xf
    # in the circle centered at 0,0 (the real min) with a radius of 3
    return 1.**2 - ((x-2)**2 + (y)**2) #positive means feasible
#ineq line constraint at x > -1
def g_threehump_2(xf):
    x, y = xf
    return x + 1 #positive means feasible

threehump_constraints = [{'type': 'ineq', 'fun': g_threehump_1},
                         {'type': 'ineq', 'fun': g_threehump_2}]

x0_th = np.array([-2., -2.])
point_tracker = []
point_tracker.append(x0_th)

def my_tracker(xk):
    point_tracker.append(np.copy(xk))

#solve/minimize 
res_th = minimize(three_hump_camel,x0_th, callback=my_tracker, constraints = threehump_constraints)
#my quad_penalty method
comb_g_th = lambda x: np.array([g_threehump_1(x), g_threehump_2(x)])
def g_throw(xf):
    return 0

res_my_th = quad_penalty(x0_th, three_hump_camel ,g_throw, comb_g_th)

summarizer(np.array(point_tracker), res_my_th, three_hump_camel)
#Plotting
X_th, Y_th, Z_th = mesher(three_hump_camel, bounds = (-3,3))
plt.contour(X_th, Y_th, Z_th, levels=50, cmap = 'viridis') #background contour
#constraints
th_circle = plt.Circle((2,0), 1, color = 'g', linestyle = '--', label = 'Circle Constraint', alpha = 0.5)
plt.gca().add_artist(th_circle)

plt.axvline(x=-1, color = 'r', linestyle = '--', label = 'x > -1 constraint')
plt.fill_betweenx(np.linspace(-3,3),-1,-3, color = 'r', alpha = 0.5)


res_pts = np.array(point_tracker)
num_points = res_pts.shape[0]
num_points2 = res_my_th.shape[0]
colors = plt.cm.Blues(np.linspace(0, 1, num_points))
colors2 = plt.cm.Oranges(np.linspace(0, 1, num_points2))
# plt.plot(res_pts)
for i, (pt, color) in enumerate(zip(res_pts, colors)):
    plt.scatter(pt[0], pt[1], color=color, edgecolor='black')
    plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')
    
for i, (pt, color) in enumerate(zip(res_my_th, colors2)):
    plt.scatter(pt[0], pt[1], color=color, edgecolor='black')
    plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')

plt.colorbar(label='Function value')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Three Hump Camel Minimization with constraints')
plt.legend()
plt.show()

convg_plotter2(res_pts, three_hump_camel)

# %% Himmelblau

def Himmelblau(xf): #actual min = 0 at (3,2), (-2.805118, 3.1312), (-3.77931, -3.283186), (3.584428, -1.848126)
    x, y = xf
    return (x**2 + y - 11.0)**2 + (x + y**2 - 7)**2

# lets see if we can get the (3.58, -1.84) minimum, the bottom right of the 4
def g_Himmelblau_1(xf):
    x, y = xf
    # I want y < 0
    return -y #positive means feasible

def g_Himmelblau_2(xf):
    x, y = xf
    return 2.**2 - ( (x-2)**2 + (y+2)**2) #positive means feasible


Himmel_constraints = [{'type': 'ineq', 'fun': g_Himmelblau_1},
                         {'type': 'ineq', 'fun': g_Himmelblau_2}]


x0_Himmel = np.array([0., 0.])
point_tracker = []
point_tracker.append(x0_Himmel)

def my_tracker(xk):
    point_tracker.append(np.copy(xk))

#solve/minimize 
res_h = minimize(Himmelblau,x0_Himmel, callback=my_tracker, constraints = Himmel_constraints)

#my quad penalty method 
comb_g_h = lambda x: np.array([g_Himmelblau_1(x), g_Himmelblau_2(x)])
res_my_h = quad_penalty(x0_Himmel, Himmelblau,g_throw, comb_g_h, mu_g=0.8)
summarizer(np.array(point_tracker), res_my_h, Himmelblau)

#Plotting
X_h, Y_h, Z_h = mesher(Himmelblau, bounds = (-4,4))
plt.contour(X_h, Y_h, Z_h, levels=50, cmap = 'viridis') #background contour
#constraints
h_circle = plt.Circle((2,-2), 2, color = 'g', linestyle = '--', label = 'Circle Constraint', alpha = 0.5)
plt.gca().add_artist(h_circle)

plt.axhline(y=0, color = 'r', linestyle = '--', label = 'x > -1 constraint')
plt.fill_between(np.linspace(-4,4),0, 4, color = 'r', alpha = 0.5)


res_pts = np.array(point_tracker)
num_points = res_pts.shape[0]
num_points2 = res_my_h.shape[0]
colors = plt.cm.Blues(np.linspace(0, 1, num_points))
colors2 = plt.cm.Oranges(np.linspace(0, 1, num_points2))
# plt.plot(res_pts)
for i, (pt, color) in enumerate(zip(res_pts, colors)):
    plt.scatter(pt[0], pt[1], color=color, edgecolor='black')
    plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')
    
for i, (pt, color) in enumerate(zip(res_my_h, colors2)):
    plt.scatter(pt[0], pt[1], color=color, edgecolor='black')
    plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')

plt.colorbar(label='Function value')
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Himmelblau Minimization with constraints')
plt.legend()
plt.show()

convg_plotter2(res_pts, Himmelblau)


# %%
