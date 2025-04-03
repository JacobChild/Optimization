#HW7.py
#Jacob Child

#%% Import needed packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def greedy_plotter(pointsf, past_pointsf, total_distancef, title = 'Greedy'):
    """
    Plot the path taken by the greedy algorithm.
    """
    #wrap the points around to close the loop
    past_pointsf = np.vstack((past_pointsf, past_pointsf[0]))  # Close the loop
    plt.figure()
    plt.plot(pointsf[:, 0], pointsf[:, 1], 'ro', label='Points')
    plt.plot(past_pointsf[:, 0], past_pointsf[:, 1], 'b-', label='Path')
    plt.plot(past_pointsf[0, 0], past_pointsf[0, 1], 'go', label='Start Point')
    plt.title(f'{title} TSP Path, Total Distance: {total_distancef:.2f}')
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

#%% Problem 1: Traveling Salesman Problem (TSP)- 2-Opt Algorithm Approach
# Functions Setup
def two_opt_swap(routeff, index1f, index2f):
    """
    Perform a 2-opt swap on the route between two indices.
    """
    new_route = np.zeros_like(routeff)
    new_route[:index1f] = routeff[:index1f] #keep the first segment
    new_route[index1f:index2f + 1] = np.flip(routeff[index1f:index2f+1])  # Reverse the segment #routeff[index2f:index1f - 1:-1]  # Reverse the segment
    new_route[index2f + 1:] = routeff[index2f + 1:] #keep the last segment
    return new_route

def total_distance_2opt(pts_idx_f, pointsf):
    """
    Calculate the total distance of the path given the indices of the points.
    """
    path_points = pointsf[pts_idx_f]
    path_points = np.vstack((path_points, path_points[0]))  # Close the loop
    distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    return np.sum(distances)

def two_opt(routef, max_iterationsf=1000):
    current_route = routef.copy()
    best_distance = total_distance_2opt(current_route, tsp_points)
    tracker = []
    tracker.append(best_distance)
    iteration = 0
    while iteration < max_iterationsf:
        improved = False
        for i in range(1, len(current_route) - 2):
            for j in range(i + 1, len(current_route)-1):
                new_route = two_opt_swap(current_route, i, j)
                new_distance = total_distance_2opt(new_route, tsp_points)
                tracker.append(new_distance)
                if new_distance < best_distance:
                    best_distance = new_distance
                    current_route = new_route.copy()
                    improved = True
        #update every 100 iterations
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Best Distance = {best_distance:.2f}")
            greedy_plotter(tsp_points, tsp_points[current_route], best_distance, title = '2-Opt')
        if not improved:
            print("No improvement found, stopping. Iterations:", iteration)
            break
        iteration += 1
    # Final plot
    greedy_plotter(tsp_points, tsp_points[current_route], best_distance, title = '2-Opt Final')
    return current_route, best_distance, tracker


#%% Problem 1: Traveling Salesman Problem (TSP)- 2-Opt Algorithm Approach Solve
two_opt_init = np.random.permutation(len(tsp_points)) #randomly permute the points to start with a different route
#find the 0 index, which is the start point, and move it to the front of the array
#roll the array to move the 0 index to the front
zero_idx = np.where(two_opt_init == 0)[0][0]
two_opt_init[zero_idx] = two_opt_init[0]
two_opt_init[0] = 0

best_route_2opt, best_distance_2opt, tracker = two_opt(two_opt_init, max_iterationsf=10)
#convergence plot
plt.figure()
plt.semilogy(tracker, 'r-')
plt.title('Convergence of 2-Opt Algorithm')
plt.xlabel('Inner Loop Iteration')
plt.ylabel('Total Distance')
plt.show()


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
#b1_11 step up to x4, and now x4 = 0
bounds_b1_11 = np.array([[0,1], [0,1], [0,0], [0,0], [0,1]])
res_b1_11 = minimize(prb3_func, x0, bounds=bounds_b1_11, constraints=constraints_dict, method='SLSQP')
res_b1_11
#b1_12 keep and fix x5 = 1
bounds_b1_12 = np.array([[0,1], [0,1], [0,0], [0,0], [1,1]])
res_b1_12 = minimize(prb3_func, x0, bounds=bounds_b1_12, constraints=constraints_dict, method='SLSQP')
res_b1_12
#b1_13 prune and switch x5 = 0 
bounds_b1_13 = np.array([[0,1], [0,1], [0,0], [0,0], [0,0]])
res_b1_13 = minimize(prb3_func, x0, bounds=bounds_b1_13, constraints=constraints_dict, method='SLSQP')
res_b1_13
#b1_14, keep and x2 = 0
bounds_b1_14 = np.array([[0,1], [0,0], [0,0], [0,0], [0,0]])
res_b1_14 = minimize(prb3_func, x0, bounds=bounds_b1_14, constraints=constraints_dict, method='SLSQP')
res_b1_14
#b1_15, reset x2 = 1 
bounds_b1_15 = np.array([[0,1], [1,1], [0,0], [0,0], [0,0]])
res_b1_15 = minimize(prb3_func, x0, bounds=bounds_b1_15, constraints=constraints_dict, method='SLSQP')
res_b1_15
#b1_16 keep and check x1 = 0
bounds_b1_16 = np.array([[0,0], [1,1], [0,0], [0,0], [0,0]])
res_b1_16 = minimize(prb3_func, x0, bounds=bounds_b1_16, constraints=constraints_dict, method='SLSQP')
res_b1_16
#b1_17 switch x1 = 1
bounds_b1_17 = np.array([[1,1], [1,1], [0,0], [0,0], [0,0]])
res_b1_17 = minimize(prb3_func, x0, bounds=bounds_b1_17, constraints=constraints_dict, method='SLSQP')
res_b1_17
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
