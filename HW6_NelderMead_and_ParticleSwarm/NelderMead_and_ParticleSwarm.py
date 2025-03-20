#NelderMead_and_ParticleSwarm.py
#Jacob Child
#March 17th, 2025 Happy St. Patty's day!

#%% packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import qmc
#%% Functions Needed
def simple_parab_func(xf):
    if xf.size == 1:
        return xf**2
    else:
        return np.sum(xf**2)

def simple_parab_vecfunc(xf):
    return xf**2
    
def egg_carton_func(xf):
    # print('xf: ', xf)
    # global xtracker
    # xtracker = np.vstack([xtracker, xf])
    x1, x2 = xf
    return 0.1*x1**2 + 0.1*x2**2 - np.cos(3*x1) - np.cos(3*x2)

def egg_carton_func_forplot(xf):
    x1, x2 = xf
    return 0.1*x1**2 + 0.1*x2**2 - np.cos(3*x1) - np.cos(3*x2)

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
    colors = plt.cm.Blues(np.linspace(1, 1, num_points))

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

# %% Problem 1 - Own Version of the Nelder Mead Algorithm 
#Nelder Mead Pseudocode: 
#Inputs: funcf (function to min), x0 (initial guess), tau_x (simplex size tolerances), tau_f (function value standard deviation tolerance)
#Outputs: x_min (minimum), f_min (minimum function value)
def s_func(x0f, length=1.0):
    n = len(x0f)
    
    # Constants from the formula
    a = length / (n * np.sqrt(2)) * (np.sqrt(n+1) - 1)
    b = length / (n * np.sqrt(2)) * (np.sqrt(n+1) - 1) + length/np.sqrt(2)
    
    # Create a matrix where diagonal elements are b and off-diagonal elements are a
    S = np.full((n, n), a)
    np.fill_diagonal(S, b)
    
    # Create the simplex: first vertex is x0f, followed by x0f + each row of S
    simplex = np.vstack([x0f, x0f + S])
    
    return simplex
    
def Nelder_Mead(funcf, x0, tau_x, tau_f):
    #1. Initialize simplex, x_j = x_0 + s_j, where s_j is from eqn 7.2
#eqn 7.2 s_j = (if j = 1) 1 / (n*sqrt(2)) * (sqrt(n+1) - 1) + 1/sqrt(2) or, if j != 1, 1 / (n*sqrt(2)) * (sqrt(n+1) - 1)
    simplex = s_func(x0)
    #2. while delta_x > tau_x and delta_f > tau_f
    delta_x = 1
    delta_f = 1
    n = len(x0)
    global xtracker
    while delta_x > tau_x and delta_f > tau_f:
        #3. Order the simplex from smallest to largest function value
        simplex = simplex[np.argsort([funcf(x) for x in simplex])]
        xtracker = np.vstack([xtracker, simplex])
        #4. Compute the centroid of the simplex, x_c = 1/n * sum(x_j), exclude the worst point, ie go to n-1
        x_c = np.mean(simplex[:-1], axis=0)
        #5. Reflect the worst point (eqn 7.3), x_r = x_c + alpha*(x_c - x_n), where alpha = 1
        x_r = x_c + (x_c - simplex[-1])
        #6. If f(x_r) < f(x_0) (ie the best point from earlier), then expand the simplex, x_e = x_c + gamma*(x_c - x_n), where gamma = 2
        f_x0 = funcf(simplex[0])
        f_xr = funcf(x_r)
        if f_xr < f_x0:
            x_e = x_c + 2*(x_c - simplex[-1])
            #7. If f(x_e) < f(x_0) then x_n = x_e (accept expansion and replace the worst point), else x_n = x_r (just accept relfection)
            if funcf(x_e) < f_x0:
                simplex[-1] = x_e
            else:
                simplex[-1] = x_r
        #8 else if (from step 6) f(x_r) <= f(x_n-1) (is reflected better than second worst point?) then x_n = x_r (accept reflection)
        elif f_xr <= funcf(simplex[-2]):
            simplex[-1] = x_r
        #9 else, if f(x_r) > f(x_n) (is reflected worse than the worst point?) then do inside contraction (eqn 7.3), x_ic = x_c - rho*(x_c - x_n), where rho = 0.5
        else:
            if f_xr > funcf(simplex[-1]): 
                x_ic = x_c - 0.5*(x_c - simplex[-1])
                #10 if f(x_ic) < f(x_n) then x_n = x_ic (accept inside contraction), ie is inside contraction better than worst point?
                if funcf(x_ic) < funcf(simplex[-1]):
                    simplex[-1] = x_ic
                else: 
                    #11 else, shrink, x_j = x_0 + sigma*(x_j - x_0), where sigma = 0.5 (shrink the simplex) eqn 7.5, this will likely be a loop for each x_j (point in simplex)
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
            else:
                #12 else (from step 7), outside contraction, x_oc = x_c + rho*(x_c - x_n), where rho = 0.5
                x_oc = x_c + 0.5*(x_c - simplex[-1])
                #13 if f(x_oc) < f(x_r) then x_n = x_oc (accept outside contraction)
                if funcf(x_oc) < f_xr:
                    simplex[-1] = x_oc
                else:
                    #14 else, shrink, x_j = x_0 + sigma*(x_j - x_0), where sigma = 0.5 (shrink the simplex) eqn 7.5, this will likely be a loop for each x_j (point in simplex)
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                        
            #update delta_x and delta_f
            delta_x = np.sum([np.linalg.norm(x - simplex[-1]) for x in simplex[:-1]]) 
            delta_f = np.sqrt(np.sum([(funcf(xi) - np.mean([funcf(x) for x in simplex]))**2 for xi in simplex])/(n+1))
    #15 end everything else, go back to step 2
    simplex = simplex[np.argsort([funcf(x) for x in simplex])]
    xtracker = np.vstack([xtracker, simplex])
    return simplex, funcf(simplex[0])



#%% Problem 1 Run it
x0 = np.array([1., 1.])
tau_x = 1e-5
tau_f = 1e-5
xtracker = np.array([x0])
x_min, f_min = Nelder_Mead(egg_carton_func, x0, tau_x, tau_f)
# %%
def plot_simplex(simplex, funcf, bounds=(-4., 4.), resolution=100, padding=0.5):
    """
    Plots the current state of the Nelder-Mead simplex on top of the function's contour plot.

    Parameters:
        simplex (ndarray): The current simplex, a 2D array where each row is a vertex.
        funcf (function): The function being minimized.
        bounds (tuple): The bounds for the plot (x_min, x_max).
        resolution (int): The resolution of the contour plot.
        padding (float): Padding around the simplex for the plot.
    """
    # Determine the bounds based on the simplex
    x_min, x_max = np.min(simplex[:, 0]), np.max(simplex[:, 0])
    y_min, y_max = np.min(simplex[:, 1]), np.max(simplex[:, 1])
    x_min, x_max = x_min - padding, x_max + padding
    y_min, y_max = y_min - padding, y_max + padding

    # Create a meshgrid with specified resolution
    x_examp = np.linspace(max(bounds[0], x_min), min(bounds[1], x_max), resolution)
    y_examp = np.linspace(max(bounds[0], y_min), min(bounds[1], y_max), resolution)
    X_examp_mesh, Y_examp_mesh = np.meshgrid(x_examp, y_examp)
    Z_examp = funcf([X_examp_mesh, Y_examp_mesh])

    # Create the contour plot
    plt.contour(X_examp_mesh, Y_examp_mesh, Z_examp, levels=50, cmap='viridis')
    
    # set up colors for the points and triangles (ie all the points and sides of a simplex are the same color), so have each color repeat 3 times?
    # Set up colors for the points
    num_points = simplex.shape[0]
    point_colors = plt.cm.Blues(np.linspace(0.3, 1, num_points))

    # Set up colors for the triangles
    num_triangles = num_points // 3
    triangle_colors = plt.cm.Blues(np.linspace(0.3, 1, num_triangles))

    # Connect the simplex vertices with lines in sets of 3
    for i in range(0, len(simplex), 3):  # Iterate in steps of 3
        if i + 2 < len(simplex):  # Ensure valid sets of 3 vertices
            plt.plot(
                [simplex[i][0], simplex[i + 1][0], simplex[i + 2][0], simplex[i][0]],  # Close the triangle
                [simplex[i][1], simplex[i + 1][1], simplex[i + 2][1], simplex[i][1]],
                color=triangle_colors[i // 3]  # Use a color for each triangle
            )
            
            # Plot the simplex points
    for i, vertex in enumerate(simplex):
        plt.scatter(vertex[0], vertex[1], color=point_colors[i], edgecolor='black')

    # Add labels and title
    plt.colorbar(label='Function value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Simplex Evolution on {funcf.__name__}')
    plt.xlim(max(bounds[0], x_min), min(bounds[1], x_max))
    plt.ylim(max(bounds[0], y_min), min(bounds[1], y_max))
    plt.legend()
    plt.show()

# %% 6.2 Use existing optimizer Nelder-Mead method and compare to my own, vary 5 initial guesses, put in table 
x0_1 = np.array([1., 1.])
xtracker = np.array([x0_1])
x_min_1, f_min_1 = Nelder_Mead(egg_carton_func, x0_1, tau_x, tau_f)
print('x0_1 iters (mine): ', xtracker.shape[0], 'x_min_1: ', x_min_1[0], 'f_min_1: ', f_min_1)
x0_1_sp = minimize(egg_carton_func, x0_1, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5}
)
print('x0_1 iters (scipy): ', x0_1_sp.nit, 'x0_1: ', x0_1_sp.x, 'f_min_1: ', x0_1_sp.fun)
x0_2 = np.array([5., 5.])
xtracker = np.array([x0_2])
x_min_2, f_min_2 = Nelder_Mead(egg_carton_func, x0_2, tau_x, tau_f)
print('x0_2 iters (mine): ', xtracker.shape[0], 'x_min_2: ', x_min_2[0], 'f_min_2: ', f_min_2)
x0_2_sp = minimize(egg_carton_func, x0_2, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5}
)
print('x0_2 iters (scipy): ', x0_2_sp.nit, 'x0_2: ', x0_2_sp.x, 'f_min_2: ', x0_2_sp.fun)
x0_3 = np.array([-3., -3.])
xtracker = np.array([x0_3])
x_min_3, f_min_3 = Nelder_Mead(egg_carton_func, x0_3, tau_x, tau_f)
print('x0_3 iters (mine): ', xtracker.shape[0], 'x_min_3: ', x_min_3[0], 'f_min_3: ', f_min_3)
x0_3_sp = minimize(egg_carton_func, x0_3, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5}
)
print('x0_3 iters (scipy): ', x0_3_sp.nit, 'x0_3: ', x0_3_sp.x, 'f_min_3: ', x0_3_sp.fun)
x0_4 = np.array([0., 0.])
xtracker = np.array([x0_4])
x_min_4, f_min_4 = Nelder_Mead(egg_carton_func, x0_4, tau_x, tau_f)
print('x0_4 iters (mine): ', xtracker.shape[0], 'x_min_4: ', x_min_4[0], 'f_min_4: ', f_min_4)
x0_4_sp = minimize(egg_carton_func, x0_4, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5}
)
print('x0_4 iters (scipy): ', x0_4_sp.nit, 'x0_4: ', x0_4_sp.x, 'f_min_4: ', x0_4_sp.fun)
x0_5 = np.array([-1., 1.])
xtracker = np.array([x0_5])
x_min_5, f_min_5 = Nelder_Mead(egg_carton_func, x0_5, tau_x, tau_f)
print('x0_5 iters (mine): ', xtracker.shape[0], 'x_min_5: ', x_min_5[0], 'f_min_5: ', f_min_5)
x0_5_sp = minimize(egg_carton_func_forplot, x0_5, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5}
)
print('x0_5 iters (scipy): ', x0_5_sp.nit, 'x0_5: ', x0_5_sp.x, 'f_min_5: ', x0_5_sp.fun)

#%% Plots for 6.1
plot_simplex(xtracker, egg_carton_func_forplot, bounds=(-4., 4.), resolution=100, padding=0.5)
#convergence plot 
fs_convg = np.array([egg_carton_func(x) for x in xtracker])
plt.plot(fs_convg)
plt.xlabel('Iteration * 3 (each iteration has 3 function evaluations)')
plt.ylabel('Function value')
plt.title('Convergence of Nelder-Mead')
plt.show()

# %% Implement my own version of particle swarm optimization
#PSO Pseudocode (based off of algorithm 7.6 in the book):
#Inputs: funcf (function to min), bounds (xbar_upper, xbar_lower), n_particles (number of particles), n_iter (number of iterations), alpha (inertia weight), Beta_max (cognitive/self weight, generally c1), gamma_max (social weight, generally c2), delta_xmax (max velocity)
#Outputs: x_min (minimum), f_min (minimum function value)
#notes: x_i, i denotes the individual, k is the iteration
def particle_swarm(funcf, bounds, n_particles, n_iter, alpha, Beta_max, gamma_max, coef_flag = False):
    k = 0
    global All_particles_tracker
    #1. initialize positions x and velocities delta_x, use latin hypercube sampling
    #My particle matrix headers: matrix size is n_particles x vars_long. x,y, f(x,y), x_best, y_best, f_best, delta_x, delta_y
    # particles = np.zeros((n_particles, 11))
    # best_particle = np.zeros((1, 11))
    #LHS for x and y points 
    sampler = qmc.LatinHypercube(d=2, seed = 42)  # d=2 for two dimensions (x and t)
    # Generate LHS samples
    lhs_samples = sampler.random(n_particles)
    # Scale the samples to the desired range
    samples = qmc.scale(lhs_samples, bounds[0], bounds[1])
    x = samples[:,0].reshape(n_particles, 1)
    y = samples[:,1].reshape(n_particles, 1)
    #initalize random velocities
    delta_xmax = 0.1*(bounds[1] - bounds[0]) #max velocity is 10% of the range
    delta_x = np.random.uniform(-delta_xmax, delta_xmax, n_particles).reshape(n_particles, 1)
    delta_y = np.random.uniform(-delta_xmax, delta_xmax, n_particles).reshape(n_particles, 1)
    f0 = np.array([funcf(xf) for xf in samples]).reshape(n_particles, 1)
    alphas = np.random.uniform(0.8, 1.2, n_particles).reshape(n_particles, 1)
    betas = np.random.uniform(0., 2.0, n_particles).reshape(n_particles, 1)
    gammas = np.random.uniform(0., 2.0, n_particles).reshape(n_particles, 1)
    particles = np.hstack([x, y, f0, x, y, f0, delta_x, delta_y, alphas, betas, gammas])
    best_particle = particles[np.argmin(particles[:,2])]
    xs = particles[:,0:2]
    plotter(xs, funcf, bounds=bounds, highres=True, resolution = 1000)
    #2. Big while not converged loop 
    while np.abs(np.median(particles[:,6:8])) > 1e-7 and k < n_iter:
        #headers: x [0], y [1], f(x,y) [2], x_best [3], y_best [4], f_best [5], delta_x [6], delta_y [7], alpha [8], beta [9], gamma [10]
        #3. for each particle i = 1 to n_particles, update bests
        for p in particles:
            #4. Update best individual point, ie if f(x_i) < f(x_i_best) then x_i_best = x_i (this is in that points' memory)
            if p[2] < p[5]:
                p[3] = p[0]
                p[4] = p[1]
                p[5] = p[2]
            #5. Update best swarm point, ie if f(x_i) < f(x_best) then x_best = x_i (this is in the swarms' memory)
            if p[2] < best_particle[5]:
                best_particle = p
        #6. for each particle i = 1 to n_particles, update velocities and positions
        for p in particles:
            #headers: x [0], y [1], f(x,y) [2], x_best [3], y_best [4], f_best [5], delta_x [6]
            #7. Update the velocity, delta_x_i_k+1 = alpha * delta_x_i_k + Beta_max * (x_i_best - x_i) + gamma_max * (x_best - x_i)
            if coef_flag:
                p[6:8] = alpha*p[6:8] + Beta_max*(p[3:5] - p[0:2]) + gamma_max*(best_particle[3:5] - p[0:2])
            else:
                p[6:8] = p[8]*p[6:8] + p[9]*(p[3:5] - p[0:2]) + p[10]*(best_particle[3:5] - p[0:2])
            #8. limit velocity: delta_x_i_k+1 = max(min(delta_x_i_k+1, delta_xmax), -delta_xmax)
            p[6:8] = np.maximum(np.minimum(p[6:8], delta_xmax), -delta_xmax)
            #9. Update the position, x_i_k+1 = x_i_k + delta_x_i_k+1
            p[0:2] = p[0:2] + p[6:8]
            #10. enforce bounds: x_i_k+1 = max(min(x_i_k+1, xbar_upper), xbar_lower)
            p[0:2] = np.maximum(np.minimum(p[0:2], bounds[1]), bounds[0])
            #11. Update the function value
            p[2] = funcf(p[0:2])
        #12. end for i
        #plot the particles every 20 iters, ie 0, 20, 40 etc 
        if k % 20 == 0:
            xs = particles[:,0:2]
            plotter(xs, funcf, bounds=(-3,3), highres=False, resolution = 1000)
        
        
        k += 1
    #13. end for k
    print('k: ', k)
    # print('Conv: ', np.mean(particles[:,6:8]))
    print('Conv_med: ', np.median(particles[:,6:8]))
    xs = particles[:,0:2]
    plotter(xs, funcf, bounds=(-3,3), highres=True, resolution = 1000)
    return best_particle[0:2], best_particle[5]

#%% speedy_particle_swarm See copilot for suggestions later
# %% 6.3 Run the PSO, vary population size, weights, convergence criteria?
bounds = (-4., 4.)
n_particles = 100
n_iter = 400
alpha = 1.0 #inertia weight btw 0 and 1
Beta_max = 1.0 #cognitive/self weight, generally c1, btwn 0 and 2
gamma_max = 1.0 #social weight, generally c2, btwn 0 and 2
print('standard run')
delta_xmax = 0.1*(bounds[1] - bounds[0]) #max velocity is 10% of the range
x_min_std, f_min_std = particle_swarm(egg_carton_func, bounds, n_particles, n_iter, alpha, Beta_max, gamma_max)

#small population
n_particles = 50
print('small population')
x_min_small, f_min_small = particle_swarm(egg_carton_func, bounds, n_particles, n_iter, alpha, Beta_max, gamma_max)

n_particles = 200 
print('large population')
x_min_large, f_min_large = particle_swarm(egg_carton_func, bounds, n_particles, n_iter, alpha, Beta_max, gamma_max)

#different weights
n_particles = 100
#almost all inertia
alpha = 1.0 
Beta_max = 0.1
gamma_max = 0.1
n_particles = 100
print('almost all inertia')
x_min_inertia, f_min_inertia = particle_swarm(egg_carton_func, bounds, n_particles, n_iter, alpha, Beta_max, gamma_max, coef_flag = True)
#almost all cognitive
alpha = 0.1
Beta_max = 1.0
gamma_max = 0.1
print('almost all cognitive')
x_min_cognitive, f_min_cognitive = particle_swarm(egg_carton_func, bounds, n_particles, n_iter, alpha, Beta_max, gamma_max, coef_flag = True)
#almost all social
alpha = 0.1
Beta_max = 0.1
gamma_max = 1.0
print('almost all social')
x_min_social, f_min_social = particle_swarm(egg_carton_func, bounds, n_particles, n_iter, alpha, Beta_max, gamma_max, coef_flag = True)


# plot note, use quiver
# %% 6.4 Apply PSO to a new test function
def Himmelblau(xf): #actual min = 0 at (3,2), (-2.805118, 3.1312), (-3.77931, -3.283186), (3.584428, -1.848126)
    x, y = xf
    return (x**2 + y - 11.0)**2 + (x + y**2 - 7)**2

def three_hump_camel(xf): #actual min = f(0,0) = 0
    x, y = xf
    return 2.0*x**2 - 1.05*x**4 + x**6 / 6.0 + x*y + y**2

All_particles_tracker = []
bounds = (-15., 15.) #for fun make them large
n_particles = 100
n_iter = 400
x_threehump, f_threehump = particle_swarm(three_hump_camel, bounds, n_particles, n_iter, alpha, Beta_max, gamma_max)



# %%
