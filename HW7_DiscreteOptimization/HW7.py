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
# %% Problem 1: TSP- (Branch and Bound Approach) Function setup
# Branch and Bound approach to solve the TSP
def total_distance_bb(pts_idx_f, pointsf):
    """
    Calculate the total distance of the path given the indices of the points.
    """
    path_points = pointsf[pts_idx_f]
    path_points = np.vstack((path_points, path_points[0]))  # Close the loop
    distances = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    return np.sum(distances)

# In this case the design variable/point is going to be an array of indexes that represent the order of the points, including wrapping back around to the first point. I will do the depth first approach. I will have similar functions to greedy, but adapted to run off of the indexes to tsp_points
# I will also need a function to calculate the lower bound of the path. I will use the minimum spanning tree (MST) to calculate the lower bound. I will also need a function to check if the path is complete. I will also need a function to plot the path taken.
#algorithm pseudocode from the book:
# Let S be the set of indices of the points 
# while branches remain, do: solve relaxed problem for xhat and fhat
# if the relaxed problem is infeasible, then prune the branch and back up tree
# else if xhat_i is included between {0,1} for all i in S, then fbest = min(fbest, fhat), prune this branch and back up tree
#      else branch further 
# end while
# return fbest and xbest
