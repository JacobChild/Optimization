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


# %% Knapsack Problem 
#setup
K = 40 # max hours available
n = len(tasks) # number of tasks
V = np.zeros((n+1, K+1)) # value matrix
S = np.zeros((n+1, K+1), dtype=bool) # selection matrix

#make a value function, given

for i in range(1, n): #AI said n+1
    for k in range(K): #AI said K+1
        if tasks[i,1] > k:
            V[i,k] = V[i-1,k]
        else:
            if tasks[i,2] + V[i-1,k-tasks[i,1]] > V[i-1,k]: #take the task
                V[i,k] = tasks[i,2] + V[i-1,k-tasks[i,1]]
                S[i,k] = True
            else: #don't take the task
                V[i,k] = V[i-1,k]
                S[i,k] = False
                

# %%
