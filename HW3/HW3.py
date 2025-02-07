#HW3.py
#Jacob Child
#Feb 5th, 2025

# %%
from LineSearch3 import *

#%% Practice with sphere_func
# func_calls = 0
# update_grad_func(sphere_func)
# x0 = np.array([2.,2.])
# ainit = 2.
# mu1 = 1e-6
# mu2 = 1e-3
# sigma = 2.
# tau = 1e-4 #! is this a good value? I don't know where to start

# xstar, f_xstar = linesearch('conj_grad', x0, ainit, mu1, mu2, sigma, tau, sphere_func, True)

# #%% 1.4a Slanted quadratic function with Beta = 1.5
# func_calls = 0
# #  report both α∗ and x(k=1) = x(k=0) +α∗p)
# def slanted_quad(vf,betaf = 1.5):
#     # f(x1,x2) = x1^2 + x2^2 -beta*x1*x2
#     global func_calls
#     func_calls += 1
#     x1 = vf[0]
#     x2 = vf[1]
#     return x1**2 + x2**2 - betaf*x1*x2

# update_grad_func(slanted_quad)
# x0 = np.array([2.,-6.])
# ainit = 1.1
# mu1 = 1e-6
# mu2 = 1e-3
# sigma = 2.
# tau = 1e-4 #! is this a good value? I don't know where to start

# xstar, f_xstar = linesearch('conj_grad', x0, ainit, mu1, mu2, sigma, tau, slanted_quad, False)
# print("1.4a Min Point: ", xstar) #(-2,2)
# print("function calls: ", func_calls) #6, although this likely includes the initial making the derivative function too

# %%
func_calls = 0
def rosenbrock(vf):
    global func_calls
    func_calls += 1
    x1 = vf[0]
    x2 = vf[1]
    return (1. - x1)**2 + 100. * (x2 - x1**2)**2

update_grad_func(rosenbrock)
x0 = np.array([0.,2.])
ainit = 0.1
mu1 = 1e-6
mu2 = 1e-3
sigma = 2.
tau = 1e-4 #! is this a good value? I don't know where to start

xstar, f_xstar = linesearch('conj_grad', x0, ainit, mu1, mu2, sigma, tau, rosenbrock, True)
print("1.4a Min Point: ", xstar) #(-2,2)
print("function calls: ", func_calls) #6, although this likely includes the initial making the derivative function too

# %%
