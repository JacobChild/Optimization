# %% Problem 1: TSP- (Genetic Algorithm Approach) Function setup
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

def ga_selection(pop_f, distances_f, perc_parents_f = 0.25):
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

def ga_tsp(pointsf, pop_size_f=10, n_generations_f=1000, mutation_rate_f=0.1):
    """
    Genetic Algorithm to solve the TSP.
    """
    best_distances = []
    best_paths = []
    # Initialize population
    pop_f = ga_init_pop(pop_size_f, len(pointsf))
    
    for generation in range(n_generations_f):
        # Calculate distances for the current population
        distances_f = total_distance_pop_ga(pop_f, pointsf)
        
        # Select parents
        parents_f = ga_selection(pop_f, distances_f)
        
        # Create offspring
        offspring_f = ga_crossover(parents_f)
        
        # Mutate offspring
        offspring_f = mutation_ga(offspring_f, mutation_rate_f)
        
        # Replace worst individuals with offspring
        pop_f = ga_replacement(pop_f, offspring_f, distances_f)
        
        # Store the best path and distance of the current generation every 100 generations
        if generation % 2000 == 0:
            distances_f = total_distance_pop_ga(pop_f, pointsf)
            best_idx = np.argmin(distances_f)
            best_path = pop_f[best_idx]
            best_distance = distances_f[best_idx]
            best_paths.append(best_path)
            best_distances.append(best_distance)
            print(f"Generation {generation}: Best Distance = {best_distance:.2f}")
            #plot the best path 
            greedy_plotter(pointsf, pointsf[best_path], best_distance)
        
    # Get the best solution from the final population
    best_idx = np.argmin(total_distance_pop_ga(pop_f, pointsf))
    best_path = pop_f[best_idx]
    best_distance = total_distance_ga(best_path, pointsf)
    
    return best_path, best_distance, best_paths, best_distances

# %% Problem 1: TSP- (Genetic Algorithm Approach) Solve
best_path_ga, best_distance_ga, best_paths_ga, best_distances_ga = ga_tsp(tsp_points, pop_size_f=200, n_generations_f=10000, mutation_rate_f=0.1)

