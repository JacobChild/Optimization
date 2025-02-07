#HW2_Problems.py
#Jacob Child
#Jan 30th, 2025

# %%
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LineSearch2 import * #If I want LineSearch3 do HW3.LineSearch3

# %% Test 
def sphere_func(vf):
    xf = vf[0]
    yf = vf[1]
    return (xf-7)**2 + yf**2 # I offset it a bit so I can start at 0 and it won't be an issue

update_grad_func(sphere_func) #! run for each problem

p = np.array([-2.,-7.]).T #direction to search ie along the x axis, so y = y we hold y constant 
pt0 = np.array([9.,9.]) #initial point [-4.,10.] is a trouble point, errors at [9,-9] and p = -2,4
phi0 = sphere_func(pt0) #intial output
ainit = 2. #initial step size
phiprime0 = directional_deriv(pt0,p) #I expect this to be -14
mu1 = 1e-4 #sufficient decrease factor, #? come back and explain this
mu2 = .1 #sufficient curvature factor, should be greater than mu1 
sigma = 2. #step size increase factor 
a_1 = 0.
a_2 = ainit 
phi_1 = phi0 
phiprime_1 = phiprime0
amin = bracket_func(pt0, ainit, phi0, phiprime0, mu1, mu2, sigma,p, sphere_func)
#It worked as expected

# %% 1.4a Slanted quadratic function with Beta = 1.5
func_calls = 0
#  report both α∗ and x(k=1) = x(k=0) +α∗p)
def Slantedquad(vf,betaf = 1.5):
    # f(x1,x2) = x1^2 + x2^2 -beta*x1*x2
    global func_calls
    func_calls += 1
    x1 = vf[0]
    x2 = vf[1]
    return x1**2 + x2**2 - betaf*x1*x2

update_grad_func(Slantedquad)
    
p = np.array([-1.,1.])
pt0 = np.array([2., -6.])
phi0 = Slantedquad(pt0)
phiprime0 = directional_deriv(pt0,p)
ainit = 2.
mu1 = 1e-4
mu2 = .1
sigma = 2.
amin = bracket_func(pt0, ainit, phi0, phiprime0,mu1,mu2,sigma,p,Slantedquad)
print("1.4a Min Point: ", amin) #(-2,2)
print("1.4a Alpha: ", alphafunc(amin,pt0,p)) #(4.0)
print("function calls: ", func_calls) #6, although this likely includes the initial making the derivative function too



# %% 1.4b Rosenbrock function
func_calls = 0
def Rosenbrock(vf):
    global func_calls
    func_calls += 1
    x1 = vf[0]
    x2 = vf[1]
    return (1. - x1)**2 + 100. * (x2 - x1**2)**2

update_grad_func(Rosenbrock)

p = np.array([1.,-3.])
pt0 = np.array([0., 2.])
phi0 = Rosenbrock(pt0)
phiprime0 = directional_deriv(pt0,p)
ainit = .2
mu1 = 1e-4
mu2 = .1
sigma = 2.
amin = bracket_func(pt0, ainit, phi0, phiprime0,mu1,mu2,sigma,p,Rosenbrock)
print("1.4b Min Point: ", amin) #(1.578e-30,2)
print("Should be: (0.5408, 0.3775)" )
print("1.4b Alpha: ", alphafunc(amin,pt0,p)) #(1.578e-30, basically 0)
print("function calls: ", func_calls) #209, it jumped away and came back for some reason.

# %% 1.4c Jones Function
func_calls = 0 
def Jones(vf):
    global func_calls
    func_calls += 1
    x1 = vf[0]
    x2 = vf[1]
    return x1**4 + x2**4 - 4*x1**3 -3*x2**3 + 2*x1**2 + 2*x1*x2

update_grad_func(Jones)

p = np.array([1.,2.])
pt0 = np.array([1., 1.])
phi0 = Jones(pt0)
phiprime0 = directional_deriv(pt0,p)
ainit = 2.
mu1 = 1e-4
mu2 = .1
sigma = 2. #must be greater than 1
amin = bracket_func(pt0, ainit, phi0, phiprime0,mu1,mu2,sigma,p,Jones)
print("1.4c Min Point: ", amin) #(1.59375,2.1875)
print("1.4c Alpha: ", alphafunc(amin,pt0,p)) # 0.59375
print("function calls: ", func_calls) #24, including intial likely.

# %% 1.1 and 1.2 check my by hand
def foo(vf):
    return vf[0]**2 + vf[1]**2

x0 = np.array([-2.,2.])
p = np.array([1.,-2.])
update_grad_func(foo)
print(directional_deriv(x0,p))
foo_grad = grad(foo)
update_grad_func(foo_grad)
print(directional_deriv(x0,p))
# %%
