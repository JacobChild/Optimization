#HW7.py
#Jacob Child

#%% Import needed packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve

# %% Problem 1: Traveling Salesman Problem (TSP)- (setup)
# TSP setup, 49 points in an area 100 x 100 square, randomly generated
np.random.seed(0)  # For reproducibility
n_points = 49
tsp_points = np.random.rand(n_points, 2) * 100  # Random points in a 100x100 area
# add the [0,0] start point to the beginning of points np array
tsp_points = np.vstack(([0, 0], tsp_points))

# %% Problem 1: TSP- (Greedy Algorithm Approach) Function setup
# Greedy algorithm to find a solution to the TSP
#pseudocode: always select the nearest point (city) to the current point to go to next. I will need a distance function, a function to remove and save points visited, and a function to calculate the total distance of the path. Make all functions vectorized for speed. In my big function I will have three main variables, past_points, current_point, and potential_points. The distance function will pull in the current_point and potential_points and return the distance to each potential point. The remove and save function will take in the current_point and potential_points and remove the current_point from potential_points and add it to past_points. The total distance function will take in past_points and calculate the total distance of the path. I will also need a function to plot the path taken.
def selector(current_pointf, potential_pointsf, past_pointsf):
    """
    Calculate the distance from the current point to all potential points. Select the nearest point and remove it from potential points.
    """
    distances = np.linalg.norm(potential_pointsf - current_pointf, axis=1)
    nearest_index = np.argmin(distances)
    past_pointsf = np.vstack((past_pointsf, potential_pointsf[nearest_index]))
    potential_pointsf = np.delete(potential_pointsf, nearest_index, axis=0)
    current_pointf = past_pointsf[-1]  # Update current point to the last added point
    return current_pointf, potential_pointsf, past_pointsf

def greedy(pointsf):
    """
    Greedy algorithm to solve the TSP.
    """
    current_pointf = pointsf[0]  # Start from the first point
    potential_pointsf = pointsf[1:]  # Remaining points
    past_pointsf = np.array([current_pointf])  # Start with the first point in past points

    while len(potential_pointsf) > 0:
        current_pointf, potential_pointsf, past_pointsf = selector(current_pointf, potential_pointsf, past_pointsf)

    return past_pointsf

def total_distance(past_pointsf):
    """
    Calculate the total distance of the path.
    """
    distances = np.linalg.norm(np.diff(past_pointsf, axis=0), axis=1)
    return np.sum(distances)

def greedy_plotter(pointsf, past_pointsf, total_distancef):
    """
    Plot the path taken by the greedy algorithm.
    """
    plt.figure()
    plt.plot(pointsf[:, 0], pointsf[:, 1], 'ro', label='Points')
    plt.plot(past_pointsf[:, 0], past_pointsf[:, 1], 'b-', label='Path')
    plt.plot(past_pointsf[0, 0], past_pointsf[0, 1], 'go', label='Start Point')
    plt.title('Greedy TSP Path, Total Distance: {:.2f}'.format(total_distancef))
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.show()
# %% Problem 1: TSP- (Greedy Algorithm Approach) Solve
past_points_greedy = greedy(tsp_points)
total_distance_greedy = total_distance(past_points_greedy) #672.8249
print("Greedy Algorithm Total Distance:", total_distance_greedy)
greedy_plotter(tsp_points, past_points_greedy, total_distance_greedy)
# %% Problem 1: TSP- (Genetic Algorithm Approach) Function setup
# Branch and Bound approach to solve the TSP
def total_distance_ga(pts_idx_f, pointsf):
    """
    Calculate the total distance of the path given the indices of the points.
    """
    path_points = pointsf[pts_idx_f]
    path_points = np.vstack((path_points, path_points[0]))  # Close the loop
    distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    return np.sum(distances)

def total_distance_pop_ga(pop_f, pointsf):
    """
    Calculate the total distance of all paths in the population. AI assisted for vectorized speedup
    """
    path_points = pointsf[pop_f]
    first_points = path_points[:, 0, :] [:, np.newaxis, :] #Shape: (npaths, npoints,2)
    closed_paths = np.concatenate((path_points, first_points), axis=1) #Shape: (npaths, npoints+1,2)
    distances = np.linalg.norm(np.diff(closed_paths, axis=1), axis=2) #Shape: (npaths, npoints)
    return np.sum(distances, axis=1) #Shape: (npaths,)

# In this case the design variable/point is going to be an array of indexes that represent the order of the points, including wrapping back around to the first point. 
# Principles of Genetic Algorithm:
# 1. Initialization: Create a population where each individual is a random permutation of the indexes of the points.
# 2. Selection: Select the best individuals to be parents for the next generation. I will use tournament selection.
# 3. Crossover: Create offspring by combining the parents. I will use order crossover.
# 4. Mutation: Randomly change the order of the points in the offspring. I will use swap mutation.
# 5. Replacement: Replace the worst individuals in the population with the offspring. This is built in elitism.
# 6. Termination: Terminate the algorithm when a stopping criterion is met. I will use a maximum number of generations.

def ga_init_pop(pop_size_f, n_points_f):
    """
    Initialize the population. 
    """
    return np.array([np.random.permutation(n_points_f) for _ in range(pop_size_f)])

def ga_selection(pop_f, distances_f, perc_parents_f = 0.5):
    """
    Select the best individuals to be parents for the next generation.
    """
    n_parents_f = int(len(pop_f) * perc_parents_f)
    selected_indices = np.argsort(distances_f)[:n_parents_f]
    return pop_f[selected_indices]

# def ga_crossover(parents_f, perc_offspring_f = 0.5):
#     """
#     Create offspring by ordered crossover.
#     """
#     n_offspring_f = int(len(parents_f) * perc_offspring_f)
#     offspring = np.empty((n_offspring_f, len(parents_f[0])), dtype=int) # Shape: (n_offspring, n_points)
#     for i in range(n_offspring_f):
#         parent1_idx = np.random.randint(len(parents_f))
#         parent2_idx = np.random.randint(len(parents_f))
#         parent1 = parents_f[parent1_idx]
#         parent2 = parents_f[parent2_idx]
        
#         # Order crossover
#         start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
#         offspring[i, start:end] = parent1[start:end]
        
#         # Fill in the rest from parent2
#         fill_idx = np.setdiff1d(np.arange(len(parent1)), offspring[i, start:end])
#         fill_values = [val for val in parent2 if val not in offspring[i]]
#         offspring[i, fill_idx] = fill_values[:len(fill_idx)]
    
#     return offspring
def ga_crossover(parents_f, perc_offspring_f=0.5):
    """
    Create offspring by ordered crossover in a more efficient manner.
    """
    n_offspring_f = int(len(parents_f) * perc_offspring_f)
    n_points = parents_f.shape[1]
    
    # Preallocate offspring array
    offspring = np.empty((n_offspring_f, n_points), dtype=int)
    
    # Randomly select pairs of parents
    parent_indices = np.random.randint(0, len(parents_f), size=(n_offspring_f, 2))
    
    # Generate random crossover points for all offspring
    crossover_points = np.sort(np.random.randint(0, n_points, size=(n_offspring_f, 2)), axis=1)
    
    for i in range(n_offspring_f):
        parent1 = parents_f[parent_indices[i, 0]]
        parent2 = parents_f[parent_indices[i, 1]]
        start, end = crossover_points[i]
        
        # Copy the segment from parent1
        offspring[i, start:end] = parent1[start:end]
        
        # Fill the rest from parent2 in order, skipping duplicates
        parent2_values = [val for val in parent2 if val not in parent1[start:end]]
        fill_idx = np.concatenate((np.arange(0, start), np.arange(end, n_points)))
        offspring[i, fill_idx] = parent2_values
    
    return offspring

def mutation_ga(offspring_f, mutation_rate_f=0.1):
    """
    Randomly change the order of the points in the offspring.
    """
    for i in range(len(offspring_f)):
        if np.random.rand() < mutation_rate_f:
            idx1, idx2 = np.random.choice(len(offspring_f[i]), 2, replace=False)
            offspring_f[i][idx1], offspring_f[i][idx2] = offspring_f[i][idx2], offspring_f[i][idx1]
    return offspring_f

def ga_replacement(pop_f, offspring_f, distances_f):
    """
    Replace the worst individuals in the population with the offspring.
    """
    combined_pop = np.vstack((pop_f, offspring_f))
    combined_distances = np.concatenate((distances_f, total_distance_pop_ga(offspring_f, tsp_points)))
    
    # Select the best individuals to keep in the population
    best_indices = np.argsort(combined_distances)[:len(pop_f)]
    return combined_pop[best_indices]

# def GA(): #TODO finish




#%% Problem 3: Branch and Bound 
# setup
def prb3_func(xf):
    x1, x2, x3, x4, x5 = xf
    return -5.6*x1 -7.0*x2 -7.8*x3 - 4.0*x4 -2.9*x5

def prb3_g1(xf):
    x1, x2, x3, x4, x5 = xf
    return -(0.8*x1 + 5.9*x2 + 3.8*x3 + 1.8*x4 + 0.8*x5) + 8.2
def prb3_g2(xf):
    x1, x2, x3, x4, x5 = xf
    return -(3.5*x1 + 2.1*x2 + 7.8*x3 + 2.2*x4 + 7.9*x5) + 10.2
def prb3_g3(xf):
    x1, x2, x3, x4, x5 = xf
    return -(3.8*x1 + 2.6*x3 + 1.6*x4) + 8.3

constraints_dict = [{'type': 'ineq', 'fun': prb3_g1},
                    {'type': 'ineq', 'fun': prb3_g2},
                   {'type': 'ineq', 'fun': prb3_g3}]

x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])



# %% Problem 3: Branch and Bound
# Solve
bounds_b0 = np.array([[0,1], [0,1], [0,1], [0,1], [0,1]])
res_b0 = minimize(prb3_func, x0, bounds=bounds_b0, constraints=constraints_dict, method='SLSQP')

#b1_1, change x3 to = 0 
bounds_b1_1 = np.array([[0,1], [0,1], [0,0], [0,1], [0,1]])
res_b1_1 = minimize(prb3_func, x0, bounds=bounds_b1_1, constraints=constraints_dict, method='SLSQP')
res_b1_1
#b1_2, keep x3 = 0 and change x4 to = 1 
bounds_b1_2 = np.array([[0,1], [0,1], [0,0], [1,1], [0,1]])
res_b1_2 = minimize(prb3_func, x0, bounds=bounds_b1_2, constraints=constraints_dict, method='SLSQP')
res_b1_2
#b1_3, keep x3 = 0 and x4 = 1 and change x5 = 0
bounds_b1_3 = np.array([[0,1], [0,1], [0,0], [1,1], [0,0]])
res_b1_3 = minimize(prb3_func, x0, bounds=bounds_b1_3, constraints=constraints_dict, method='SLSQP')
res_b1_3
#b1_4, now only x1 unbounded and x2 = 0
bounds_b1_4 = np.array([[0,1], [0,0], [0,0], [1,1], [0,0]])
res_b1_4 = minimize(prb3_func, x0, bounds=bounds_b1_4, constraints=constraints_dict, method='SLSQP')
res_b1_4
# #b1_5, now all x's are bounded, x1 = 0
bounds_b1_5 = np.array([[0,0], [0,0], [0,0], [1,1], [0,0]])
res_b1_5 = minimize(prb3_func, x0, bounds=bounds_b1_5, constraints=constraints_dict, method='SLSQP')
res_b1_5
# #b1_6, now all x's are bounded, x1 = 1
bounds_b1_6 = np.array([[1,1], [0,0], [0,0], [1,1], [0,0]])
res_b1_6 = minimize(prb3_func, x0, bounds=bounds_b1_6, constraints=constraints_dict, method='SLSQP')
res_b1_6
#b1_7, step up and x1 is unbounded, x2 = 1
bounds_b1_7 = np.array([[0,1], [1,1], [0,0], [1,1], [0,0]])
res_b1_7 = minimize(prb3_func, x0, bounds=bounds_b1_7, constraints=constraints_dict, method='SLSQP')
res_b1_7
#b1_8 now x1 = 1
bounds_b1_8 = np.array([[1,1], [1,1], [0,0], [1,1], [0,0]])
res_b1_8 = minimize(prb3_func, x0, bounds=bounds_b1_8, constraints=constraints_dict, method='SLSQP')
res_b1_8 #infeasible
#b1_9 now x1 = 0
bounds_b1_9 = np.array([[0,0], [1,1], [0,0], [1,1], [0,0]])
res_b1_9 = minimize(prb3_func, x0, bounds=bounds_b1_9, constraints=constraints_dict, method='SLSQP')
res_b1_9
#b1_10 step up to x5, and now x5 = 1
bounds_b1_10 = np.array([[0,1], [0,1], [0,0], [1,1], [1,1]])
res_b1_10 = minimize(prb3_func, x0, bounds=bounds_b1_10, constraints=constraints_dict, method='SLSQP')
res_b1_10

# %% Problem 3: Branch and Bound, branch 2
#b2_1, change x3 to = 1
bounds_b2_1 = np.array([[0,1], [0,1], [1,1], [0,1], [0,1]])
res_b2_1 = minimize(prb3_func, x0, bounds=bounds_b2_1, constraints=constraints_dict, method='SLSQP')
res_b2_1
#b2_2, keep x3 = 1 and change x2 to = 1
bounds_b2_2 = np.array([[0,1], [1,1], [1,1], [0,1], [0,1]])
res_b2_2 = minimize(prb3_func, x0, bounds=bounds_b2_2, constraints=constraints_dict, method='SLSQP')
res_b2_2
#b2_3, keep x3 = 1 now, x2 = 0
bounds_b2_3 = np.array([[0,1], [0,0], [1,1], [0,1], [0,1]])
res_b2_3 = minimize(prb3_func, x0, bounds=bounds_b2_3, constraints=constraints_dict, method='SLSQP')
res_b2_3
#b2_4, keep, now x5 = 0 
bounds_b2_4 = np.array([[0,1], [0,0], [1,1], [0,1], [0,0]])
res_b2_4 = minimize(prb3_func, x0, bounds=bounds_b2_4, constraints=constraints_dict, method='SLSQP')
res_b2_4
#b2_5 keep and set x1 = 0
bounds_b2_5 = np.array([[0,0], [0,0], [1,1], [0,1], [0,0]])
res_b2_5 = minimize(prb3_func, x0, bounds=bounds_b2_5, constraints=constraints_dict, method='SLSQP')
res_b2_5
#b2_6 keep, now x4 = 1
bounds_b2_6 = np.array([[0,0], [0,0], [1,1], [1,1], [0,0]])
res_b2_6 = minimize(prb3_func, x0, bounds=bounds_b2_6, constraints=constraints_dict, method='SLSQP')
res_b2_6
#b2_6 check x4 = 0
bounds_b2_7 = np.array([[0,0], [0,0], [1,1], [0,0], [0,0]])
res_b2_7 = minimize(prb3_func, x0, bounds=bounds_b2_7, constraints=constraints_dict, method='SLSQP')
res_b2_7
#b2_8 step up and x1 = 1
bounds_b2_8 = np.array([[1,1], [0,0], [1,1], [0,1], [0,0]])
res_b2_8 = minimize(prb3_func, x0, bounds=bounds_b2_8, constraints=constraints_dict, method='SLSQP')
res_b2_8
#b2_9 step up and x5 = 1
bounds_b2_9 = np.array([[0,1], [0,0], [1,1], [0,1], [1,1]])
res_b2_9 = minimize(prb3_func, x0, bounds=bounds_b2_9, constraints=constraints_dict, method='SLSQP')
res_b2_9

# %%
