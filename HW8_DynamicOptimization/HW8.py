#HW8.py
#Jacob Child
#April 9th, 2025

#%% Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# %% 8.1 Knapsack Problem
#Background: 5 workers can give a collective 40hrs of work tomorrow. There are a set of tasks where hours are the cost to the company and the value is the benefit to the company. The goal is to maximize the value of the tasks that can be completed in 40hrs. Two people can work on the same project, but the project will only be completed once. 
#constraints: each task is only done once, each task can only be done by a max of two people, 40 hours is all that is available.

try:
    tasks = np.loadtxt('knapsack_tasks.txt', dtype = int) #(index, hours, value)
except FileNotFoundError:
    tasks = np.loadtxt('HW8_DynamicOptimization/knapsack_tasks.txt', dtype = int) #(index, hours, value)

# %% Baby knapsack (from class)
todos = np.array([[1,4,4],[2,5,3],[3,2,3],[4,6,7],[5,1,2]]) #(index, hours, value)
K = 10 # max hours available
n = len(todos) # number of tasks
V = np.zeros((K+1, n)) # value matrix
S = np.zeros((K+1, n), dtype=bool) # selection matrix

for i in range(0, n): # 0,n works when V is K+1, n+1
    for k in range(1, K+1): #k is current capacity and doubles as the index of the task, 0, k+1 works when V is K+1, n+1
        wi = todos[i, 1] # weight of the task
        vi = todos[i, 2] # value of the task
        if wi > k:
            V[k, i] = V[k, i-1]
        else:
            if vi + V[k-wi, i-1] > V[k, i-1]:
                V[k, i] = vi + V[k-wi, i-1]
                S[k][i] = True
            else:
                V[k, i] = V[k, i-1]
# add column of 0s to the left of V and S
V = np.insert(V, 0, 0, axis=1)
S = np.insert(S, 0, False, axis=1)

#what items were selected?
selected_items = []
for i in range(n, 0, -1):
    if S[K][i]:
        selected_items.append(i)
        K -= todos[i-1][1]
        
# %% Knapsack Problem 
#setup
K = 40 # max hours available
n = len(tasks) # number of tasks
V = np.zeros((K+1, n)) # value matrix
S = np.zeros((K+1, n), dtype=bool) # selection matrix

for i in range(0, n): # 0,n works when V is K+1, n+1
    for k in range(1, K+1): #k is current capacity and doubles as the index of the task, 0, k+1 works when V is K+1, n+1
        wi = tasks[i, 1] # weight of the task
        vi = tasks[i, 2] # value of the task
        if wi > k:
            V[k, i] = V[k, i-1]
        else:
            if vi + V[k-wi, i-1] > V[k, i-1]:
                V[k, i] = vi + V[k-wi, i-1]
                S[k][i] = True
            else:
                V[k, i] = V[k, i-1]
# add column of 0s to the left of V and S
V = np.insert(V, 0, 0, axis=1)
S = np.insert(S, 0, False, axis=1)

#what items were selected?
selected_items = []
for i in range(n, 0, -1):
    if S[K][i]:
        selected_items.append(i)
        K -= tasks[i-1][1]
    
#Output tables
np.savetxt('Outputs/knapsack_value.csv', V, fmt='%d')
np.savetxt('Outputs/knapsack_selection.csv', S, fmt='%d')

# %% Knapsack, but greedy algorithm
#greedy just takes the highest value tasks until it runs out of hours
sorted_tasks = tasks[tasks[:, 2].argsort()[::-1]] # sort tasks by value
selected_tasks = []
V_greedy = 0 # value of selected tasks
remaining_hours = 40
for task in sorted_tasks:
    if task[1] <= remaining_hours:
        selected_tasks.append(task)
        remaining_hours -= task[1]
        V_greedy += task[2]
    else:
        break




# %% Problem 8.2: Simulated Annealing to the TSP
# TSP setup, 49 points in an area 100 x 100 square, randomly generated
np.random.seed(0)  # For reproducibility
n_points = 49
tsp_points = np.random.rand(n_points, 2) * 100  # Random points in a 100x100 area
# add the [0,0] start point to the beginning of points np array
tsp_points = np.vstack(([0, 0], tsp_points))

#functions 
def total_distance(pts_idx_f, pointsf):
    """
    Calculate the total distance of the path given the indices of the points.
    """
    path_points = pointsf[pts_idx_f]
    path_points = np.vstack((path_points, path_points[0]))  # Close the loop
    # distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    distances = np.sqrt(np.sum(np.diff(path_points, axis=0)**2, axis=1))  # Vectorized distance calculation
    return np.sum(distances)


def neighbor(pts_idx_f):
    """
    Generate a neighbor solution by, with a random decision, either reverse a random segment of path, or randomly choose two points and move the path segments to follow another randomly chosen point.
    """
    picker = np.random.randint(0, 2) # random decision to reverse a segment or swap two points
    pt1 = np.random.randint(1, len(pts_idx_f)-1) # random point to start the segment
    pt2 = np.random.randint(1, len(pts_idx_f)-1) # random point to end the segment
    if pt1 > pt2: # make sure pt1 is less than pt2
        start = pt2
        end = pt1
    elif pt1 < pt2:
        start = pt1
        end = pt2
    elif pt1 == pt2:
        start = pt1
        end = pt2 + 1
    else:
        print("Error in neighbor function")
        
    if picker == 0:
        # print("Reversing a segment")
        # print("start val is", pts_idx_f[start], "end val is", pts_idx_f[end])
        # Reverse a random segment of the path
        new_path = np.concatenate((pts_idx_f[:start], pts_idx_f[start:end+1][::-1], pts_idx_f[end+1:]))
    else:
        # print("Moving a segment")
        # the start and end of the segment are already chosen. choose a point to put it behind that isn't in the already chosen segment
        sin_seg = np.delete(pts_idx_f, np.arange(start, end+1)) # get the points that are not in the segment
        seg = pts_idx_f[start:end+1]
        if len(sin_seg) - 1 < 2:
            #recursion
            # print("Recursion")
            return neighbor(pts_idx_f)
                                
        move_idx = np.random.randint(1, len(sin_seg)-1) # choose a point to put the segment behind
        # print("start is", start, "end is", end, "move_idx is", move_idx)
        # print("sin_seg is", sin_seg, "seg is", seg)
        # print(f"Moving segment {seg} behind point {sin_seg[move_idx]}")
        new_path = np.insert(sin_seg, move_idx, seg) # insert the segment behind the chosen point
                
    return new_path


def sim_anneal(pts_idx, pointsf, initial_temp=1000, cooling_rate=0.995, max_iter=10000):
    """
    Simulated Annealing algorithm for TSP.
    """
    current_solution = pts_idx.copy()
    current_cost = total_distance(current_solution, pointsf)
    best_solution = current_solution.copy()
    best_cost = current_cost

    temp = initial_temp

    for i in range(max_iter):
        x_new = neighbor(current_solution) # get a new solution
        new_cost = total_distance(x_new, pointsf)
        delta_cost = new_cost - current_cost
        if new_cost <= current_cost:
            current_solution = x_new.copy()
            current_cost = new_cost
        else:
            if np.exp(-delta_cost / temp) >= np.random.rand():
                current_solution = x_new.copy()
                current_cost = new_cost
        
        temp *= cooling_rate
        if current_cost < best_cost:
            best_solution = current_solution.copy()
            best_cost = current_cost
        # print(f"Iteration {i}: Current cost: {current_cost}, Best cost: {best_cost}, Temperature: {temp}")
    
    return best_solution, best_cost
    

# %% Run the Simulated Annealing algorithm
init_idxs = np.random.permutation(len(tsp_points))
print("starting cost is", total_distance(init_idxs, tsp_points))
best_solution, best_cost = sim_anneal(init_idxs, tsp_points, initial_temp=1000, cooling_rate=0.995, max_iter=10000)
print("best cost is", best_cost)

# %% Run 100 times and do a histogram of the results
costs = []
paths = []
for i in range(100):
    # init_idxs = np.random.permutation(len(tsp_points))
    best_solution, best_cost = sim_anneal(init_idxs, tsp_points, initial_temp=1000, cooling_rate=0.995, max_iter=10000)
    costs.append(best_cost)
    # save 5 paths, ie every 20 iterations
    if i % 20 == 0:
        paths.append(best_solution.copy())
    
#plot histogram of costs
plt.hist(costs, bins=20)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Histogram of TSP costs')
plt.grid()
plt.show()

# %% 8.2 Plotting 
#plot the 5 paths in a 2x3 grid
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()
for i, path in enumerate(paths):
    path_points = tsp_points[path]
    path_points = np.vstack((path_points, path_points[0]))  # Close the loop
    axs[i].plot(path_points[:, 0], path_points[:, 1], marker='o')
    score = total_distance(path, tsp_points)
    axs[i].set_title(f'Path {i+1} (Cost: {score:.2f})')
    axs[i].set_xlabel('X')
    axs[i].set_ylabel('Y')
    axs[i].grid()
    axs[i].set_xlim(0, 100)
    axs[i].set_ylim(0, 100)
    axs[i].set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()


# %% 8.2 Continued, Explore parameters
#Comparing Cooling rate
low_rate = .9995
high_rate = .5
low_rate_costs = []
high_rate_costs = []
for i in range(30):
    # init_idxs = np.random.permutation(len(tsp_points))
    best_solution, best_cost = sim_anneal(init_idxs, tsp_points, initial_temp=1000, cooling_rate=low_rate, max_iter=10000)
    low_rate_costs.append(best_cost)
    best_solution, best_cost = sim_anneal(init_idxs, tsp_points, initial_temp=1000, cooling_rate=high_rate, max_iter=10000)
    high_rate_costs.append(best_cost)

#plot side by side histograms of costs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(low_rate_costs, bins=20)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Histogram of TSP costs (Low Rate, 0.9995)')
plt.grid()
plt.subplot(1, 2, 2)
plt.hist(high_rate_costs, bins=20)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Histogram of TSP costs (High Rate, 0.5)')
plt.grid()
plt.show()

# %% 8.2 Comparing Max iters
#comparing iterations 
low_iter = 1000
high_iter = 10000
low_iter_costs = []
high_iter_costs = []
for i in range(30):
    # init_idxs = np.random.permutation(len(tsp_points))
    best_solution, best_cost = sim_anneal(init_idxs, tsp_points, initial_temp=1000, cooling_rate=0.995, max_iter=low_iter)
    low_iter_costs.append(best_cost)
    best_solution, best_cost = sim_anneal(init_idxs, tsp_points, initial_temp=1000, cooling_rate=0.995, max_iter=high_iter)
    high_iter_costs.append(best_cost)

#plot side by side histograms of costs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(low_iter_costs, bins=20)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Histogram of TSP costs (Low Iterations)')
plt.grid()
plt.subplot(1, 2, 2)
plt.hist(high_iter_costs, bins=20)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Histogram of TSP costs (High Iterations)')
plt.grid()
plt.show()

# %% 8.2 Continued, Explore parameters, change the starting point
init_idxs1 = np.random.permutation(len(tsp_points))
init_idxs2 = np.random.permutation(len(tsp_points))
init1_costs = []
init2_costs = []
for i in range(30):
    # init_idxs = np.random.permutation(len(tsp_points))
    best_solution, best_cost = sim_anneal(init_idxs1, tsp_points, initial_temp=1000, cooling_rate=0.995, max_iter=10000)
    init1_costs.append(best_cost)
    best_solution, best_cost = sim_anneal(init_idxs2, tsp_points, initial_temp=1000, cooling_rate=0.995, max_iter=10000)
    init2_costs.append(best_cost)

#plot side by side histograms of costs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(init1_costs, bins=20)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Histogram of TSP costs (Initial Random Path 1)')
plt.grid()
plt.subplot(1, 2, 2)
plt.hist(init2_costs, bins=20)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Histogram of TSP costs (Initial Random Path 2)')
plt.grid()
plt.show()
# %% 8.3 Multi-Objective Optimization, Define the Pareto Front and plot 10 pts on it, and 50 that are dominated
def Himmelblau(xf): #actual min = 0 at (3,2), (-2.805118, 3.1312), (-3.77931, -3.283186), (3.584428, -1.848126)
    x, y = xf
    return (x**2 + y - 11.0)**2 + (x + y**2 - 7)**2

def three_hump_camel(xf): #actual min = f(0,0) = 0
    x, y = xf
    return 2.0*x**2 - 1.05*x**4 + x**6 / 6.0 + x*y + y**2

# Epsilon Constraint Method, minimize one objective while setting the other as an additional constraint
epsilons = np.linspace(0, 20, 100) # 10 points on the pareto front
pareto_points_th = []
pareto_values_th = []
pareto_points_h = []
pareto_values_h = []
x0 = np.array([1., 1.]) # initial guess

#three hump as the constraint
for eps in epsilons:
    cons = ({'type': 'ineq', 'fun': lambda x: eps - three_hump_camel(x)}) #positive is feasible, so this is f_j <= eps
    res = minimize(Himmelblau, x0, constraints=cons, method='SLSQP')
    pareto_points_th.append(res.x)
    pareto_values_th.append(res.fun)
    
#Himmelblau as the constraint
for eps in epsilons:
    cons = ({'type': 'ineq', 'fun': lambda x: eps - Himmelblau(x)}) #positive is feasible, so this is f_j <= eps
    res = minimize(three_hump_camel, x0, constraints=cons, method='SLSQP')
    pareto_points_h.append(res.x)
    pareto_values_h.append(res.fun)

#generate random dominated points by adding arbitrary noise to the pareto points
dominated_points_th = []
dominated_values_th = []
dominated_points_h = []
dominated_values_h = []
for i in range(75):
    #generate random noise
    noise = np.abs(np.random.normal(0.1, 0.5, size=(2,)))  # Ensure noise is positive
    #add noise to the pareto points
    dominated_points_th.append(pareto_points_th[i] + noise)
    dominated_values_th.append(Himmelblau(pareto_points_th[i] + noise))
    dominated_points_h.append(pareto_points_h[i] + noise)
    dominated_values_h.append(three_hump_camel(pareto_points_h[i] + noise))
    
# %% Plotting the front

def is_dominated(candidate, pareto_points): #AI generated
    """
    Check if a candidate point is dominated by any Pareto-optimal point.
    
    Parameters:
        candidate (tuple): (f1, f2) values of the candidate point.
        pareto_points (list of tuples): List of (f1, f2) values of Pareto-optimal points.

    Returns:
        bool: True if candidate is dominated, False otherwise.
    """
    for pareto in pareto_points:
        if all(p <= c for p, c in zip(pareto, candidate)) and any(p < c for p, c in zip(pareto, candidate)):
            return True  # Found a Pareto point that dominates candidate
    return False  # Candidate is not dominated

# Example: Checking if dominated points are actually dominated
dominated_flags = [is_dominated((dom_h, dom_th), zip(pareto_values_h, pareto_values_th)) 
                   for dom_h, dom_th in zip(dominated_values_h, dominated_values_th)]

dominated_points_th = [p for p, flag in zip(dominated_points_th, dominated_flags) if flag]
dominated_points_h = [p for p, flag in zip(dominated_points_h, dominated_flags) if flag]
print(f"Number of dominated points (Three Hump): {len(dominated_points_th)}")
print(f"Number of dominated points (Himmelblau): {len(dominated_points_h)}")
# Plot the Pareto front
f_h_th = [Himmelblau(p) for p in pareto_points_th]
f_t_th = [three_hump_camel(p) for p in pareto_points_th]
f_h_h = [Himmelblau(p) for p in pareto_points_h]
f_t_h = [three_hump_camel(p) for p in pareto_points_h]
f_h_d = [Himmelblau(p) for p in dominated_points_th]
f_t_d = [three_hump_camel(p) for p in dominated_points_th]

plt.figure(figsize=(8, 6))
plt.scatter(f_h_th, f_t_th, c='blue', label='Pareto Front Three Hump Constraint')
plt.scatter(f_h_h, f_t_h, c='green', label='Pareto Front Himmelblau Constraint')
plt.scatter(f_h_d, f_t_d, c='red', label='Dominated Points')
plt.xlabel('Himmelblau Function Value')
plt.ylabel('Three Hump Camel Function Value')
plt.title('Pareto Front')
plt.grid()
plt.legend()
plt.show()
# %%
