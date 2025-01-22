# From Dr. Salmon HW1 Help
# this is an example program to illustrate a simple optimization problem and make some contours

import numpy as np 
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# objective function definition
def f(x):
    return (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1]-x[0]**2)**2

# constraint definition
def g1(x):
    return -(2*x[0] - x[1]+4)
def g2(x):
    return -((x[0]+2)**2 + x[1]**2 -4) # remember scipy takes non-negative constraints!!! so g(x) >= 0 instead of the usual way: g(x) <= 0

def g(x):
    gvals = np.zeros([len(x)])
    gvals[0] = -(2*x[0] - x[1]+4)
    gvals[1] = -((x[0]+2)**2 + x[1]**2 -4)
    return gvals

#initial guess
x0 = [3,4]
print(f(x0))

# set up bounds and constraints if desired.
thebounds = ((-4,5),(-5,5))
# thebounds = ((-4,-2),(-5,5))
theconstraints = [{'type':'ineq','fun':g1},{'type':'ineq','fun':g2}]
# theconstraints = [{'type':'ineq','fun':g1},{'type':'eq','fun':g2}]
# theconstraints = [{'type':'ineq','fun':g}]

# optimize
# res = minimize(f,x0)
# res = minimize(f,x0,bounds=thebounds)
res = minimize(f,x0,constraints=theconstraints,bounds=thebounds)
# res = minimize(f,x0, constraints=theconstraints, bounds=thebounds, tol=1e-6)

print(res)
print("The optimal point is found at [",res.x[0] ,",", res.x[1])

# design space 
n1 = 100
x1 = np.linspace(-5,5,n1)
n2 = 99
x2 = np.linspace(-5,5,n2)

fun_output = np.zeros([n1,n2])
g1_output = np.zeros([n1,n2])
g2_output = np.zeros([n1,n2])
# h_output = np.zeros([n1,n2])
# g_output = np.zeros([n1,n2])

# generating the function output
for i in range(n1):
    for j in range(n2):
        fun_output[i,j] = f([x1[i],x2[j]])
        g1_output[i,j] = -g1([x1[i],x2[j]])
        g2_output[i,j] = -g2([x1[i],x2[j]])
        # h_output[i,j] = -g2([x1[i],x2[j]])
        # g_output[i,j] = -g([x1[i],x2[j]])

# print(fun_output)

if True:

    # plotting the contour with the constraints and the optimal point
    plt.figure()

    # default contours
    # plt.contour(x1,x2, np.transpose(fun_output),50)

    # blue contour lines
    # plt.contour(x1,x2, np.transpose(fun_output), levels=100, cmap="Blues")  # 'Blues' is a built-in colormap
    plt.contour(x1,x2, np.transpose(fun_output), levels=100, cmap="Blues_r")  # 'Blues' is a built-in colormap


    # # labeling contours
    # contour = plt.contour(x1,x2, np.transpose(fun_output), 50, linewidths=1)
    # plt.clabel(contour, inline=True, fontsize=5, colors='black')


    plt.plot(res.x[0], res.x[1],"r*")
    plt.colorbar()

    #include inequality constraints
    plt.contourf(x1,x2, np.transpose(g1_output),[0, 1000], colors=['red'], alpha=0.3)
    plt.contourf(x1,x2, np.transpose(g2_output),[0, 1000], colors=['red'], alpha=0.3)

    #include equality constraints
    # plt.contour(x1,x2, np.transpose(h_output),[0], colors=['red'])

    plt.title("Usually you will have a caption, but if not, use a title")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    # plt.close()


if False: # make this True to see the 3D projection
    
    ax = plt.figure().add_subplot(projection='3d')

    # # ax.contourf(x1, x2, np.transpose(f), 100, cmap=cm.coolwarm)  # Plot contour curves
    ax.contour(x1, x2, np.transpose(fun_output), 50, cmap='viridis')  # Plot contour curves

    X1, X2 = np.meshgrid(x1, x2)

    ax.plot_surface(X1, X2, np.transpose(fun_output), edgecolor='royalblue', lw=0.1, rstride=5, cstride=5,alpha=0.3)
    ax.contour(X1, X2, np.transpose(fun_output), zdir='z', levels=25, offset=-150, cmap='viridis')
    # ax.contour(X1, X2, np.transpose(f), zdir='x', offset=-4, cmap='coolwarm')
    # ax.contour(X1, X2, np.transpose(f), zdir='y', offset=4, cmap='coolwarm')

    ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-150, 100),
    xlabel='X', ylabel='Y', zlabel='Z')

    plt.show()