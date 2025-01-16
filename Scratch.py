# Scratch.py 
# Jacob Child
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def foo(x):
    return (2*x[0] - 3)**2 + (x[1]-1)**2 + 0.5*(2*x[1]-x[0]**2)**2


def g(x): #constraint equation
    return -(x[0] + x[1] - 2)


# optimizer start
x0 = [0,0] # initial guess
myconstraints = [{'type':'ineq', 'fun':g}]
res = minimize(foo, x0, constraints = myconstraints)
print(res)

# playing around in the space below
print(foo([0,0])) # 
print(foo([1,1]))
print(foo([1.45,1]))

n1 = 100
x1 = np.linspace(-3, 3, n1) 
n2 = 100
x2 = np.linspace(-3, 3, n2)
# print(x1)
fun_output = np.zeros([n1,n2])
g_output = np.zeros([n1,n2])

# compare time taken for loop vs comprehension

# print(fun_output)

for i in range(n1):
    for j in range(n2):
        fun_output[i,j] = foo([x1[i],x2[j]])
        g_output[i,j] = g([x1[i],x2[j]])
        
# repeat the above but with a comprehension
fun_output2 = np.array([[foo([x1[i],x2[j]]) for j in range(n2)] for i in range(n1)])

print('g is: ', g([res.x[0], res.x[1]]))

plt.figure()
num_lines = 500
# plt.contour(x1,x2,fun_output, num_lines)
# plt.contour(x1,x2,fun_output, [-0.1, 0,1,10,100,1000], linewidths = 2)
plt.contour(x1,x2, np.transpose(fun_output), [-0.1, 0,1,10,100,1000], linewidths = 2, colors = ['r','b']) # this is the same as above but with the line color red
plt.contour(x1,x2, np.transpose(fun_output), 100, linewidths = 2, colors = ['r','b']) # this is the same as above but with the line color red
plt.contourf(x1,x2, np.transpose(g_output), [0, np.max(g_output)], alpha = 0.5) # recommended by copilot, alpha is the transparency
#! #? something made it so we had to transpose the above, and I don't quite understand
#! the way we did the for loop was backwards, so we had to transpose it to make it work -> look into this
# Quick copy new line is shift + alt + down arrow !
plt.plot(res.x[0], res.x[1], 'r*')
plt.colorbar()
plt.show() #without this I can't see it, the program doesn't end until I close the plot