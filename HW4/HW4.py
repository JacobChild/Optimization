#HW4.py
#Jacob Child
#Feb 17th, 2025

#%% Packages
#! .venv\Scripts\Activate.ps1
import jax.numpy as jnp
import jax as jax
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#%% 4.1 Calculate the Jacobian for the 4-Dimensional Rosenbrock function
#%% Needed functions
def rosenbrock(xsf):
    x1,x2,x3,x4 = xsf
    return 100*(x2-x1**2)**2 + (1-x1)**2 + 100*(x3-x2**2)**2 + (1-x2)**2 + 100*(x4 - x3**2)**2 + (1-x3)**2

def complex_step(hf, xf, funcf):
    #how to comprehension t1 = [expression for item in iterable if condition (optional)]
    f_plus = jnp.array([funcf(xf + hf * jnp.eye(xf.size, dtype=complex)[i]) for i in range(xf.size)])
    if jnp.sum(f_plus.imag) == 0:
        print('zero imaginary part')
    Jstar = f_plus.imag / hf.imag 
    return Jstar

def forward_diff(hf, xf, funcf):
    f0 = funcf(xf)
    dx = hf * (1. + jnp.abs(xf))
    f_plus = jnp.array([funcf(xf + dx * jnp.eye(xf.size)[i]) for i in range(xf.size)])
    Jstar = (f_plus - f0) / dx 
    return Jstar

def my_test_func(xf):
    return jnp.sin(xf)

# %% 4.1 continued, calculating the Jacobian
# test 
t0 = jnp.array([jnp.pi], dtype = complex)
th = 1e-30j
tout = complex_step(th, t0, my_test_func)
tout2 = forward_diff(th, t0, my_test_func)
print('Testing complex_step: ', tout[0] == -1)
print('Testing forward_diff: ', tout2[0] == -1)
# actual calcs
x0 = jnp.array([0.5, -0.5, 0.5, 1.0], dtype=complex)#.reshape(4,1)
hs = jnp.logspace(-30,-1, 30) * 1.j
rosen_jacobian_41 = complex_step(th,x0,rosenbrock)
rosen_jacobian_exact = jax.jacobian(rosenbrock)(x0.real)
print(jnp.allclose(rosen_jacobian_41, rosen_jacobian_exact))#I think close enough
#how to comprehension t1 = [expression for item in iterable if condition (optional)]
rosen_jacobian_plural_41_complex = jnp.array([complex_step(hs[i],x0,rosenbrock) for i in range(hs.size)])
complex_errors = jnp.array([jnp.linalg.norm(rosen_jacobian_plural_41_complex[i] - rosen_jacobian_exact) for i in range(rosen_jacobian_plural_41_complex.shape[0])])
rosen_jacobian_plural_41_forward = jnp.array([forward_diff(hs.imag[i],x0,rosenbrock) for i in range(hs.size)])
forward_errors = jnp.array([jnp.linalg.norm(rosen_jacobian_plural_41_forward[i] - rosen_jacobian_exact) for i in range(rosen_jacobian_plural_41_forward.shape[0])])

# %% 4.1 Plot
plt.figure()
plt.loglog(hs.imag[::-1], complex_errors[::-1], label = 'Complex Step', marker='o')
plt.loglog(hs.imag[::-1], forward_errors[::-1], label = 'Foward Diff', marker='o')
plt.gca().invert_xaxis()
plt.xlabel('Step size, h')
plt.ylabel('Relative error (normed)')
plt.title('Jacobian Relative Errors')
plt.legend()

# %% 4.2 Derivatives using AD (algorithmic differentiation)
x = 2.
y = 1.5 
# given lines of code, f = [f1,f2].T, doing forward mode (my choice)
def func_42(xf,yf):
    a = xf **2 + 3* yf **2
    b = jnp.sin ( a )
    c = 3* xf **2 + yf **2
    d = jnp.sin ( c )
    e = xf **2 + yf **2
    g = -0.5* jnp.exp ( - e /2)
    f1 = a + c + g
    h = xf * yf
    f2 = a - g + h 
    return jnp.asarray([f1,f2])

func_42_exact_jacobian = jnp.array(jax.jacobian(func_42, argnums= (0,1))(x,y)) #argnums tells it to do with respect to both inputs

# My way
v1 = x
v2 = y
v3 = v1 **2 + 3* v2 **2
v4 = jnp.sin ( v3 )
v5 = 3* v1 **2 + v2 **2
v6 = jnp.sin ( v5 )
v7 = v1 **2 + v2 **2
v8 = -0.5* jnp.exp ( - v7 /2) #!I'm off somewhere in here
v9 = v3 + v5 + v8
v10 = v1 * v2
v11 = v3 -v8 + v10 

Jac_AD_42 = jnp.array([[v1/2 * jnp.exp(-v7/2) + 8*v1, v2/2 * jnp.exp(-v7/2) + 8*v2], [v2 - v1/2*jnp.exp(-v7/2)+2*v1, v1-v2/2*jnp.exp(-v7/2) + 6*v2]])
print('4.2 Jacobian Check: ', jnp.isclose(func_42_exact_jacobian, Jac_AD_42))

# %% 4.3 Truss Derivatives
from complex_modified_truss import truss as c_truss
from jaxmodified_truss import truss as jm_truss
import scipy as scipy
MyMassTracker = []
TrussFunctionCalls = 0
def MyCallbackFunc(xk):
    massf, _ = jm_truss(xk)
    MyMassTracker.append(massf)
 
def TrussMass_c(Af):
    global TrussFunctionCalls
    # TrussFunctionCalls += 1 
    massf, _ = c_truss(Af)
    return massf

def TrussMass_jm(Af):
    global TrussFunctionCalls
    TrussFunctionCalls += 1 
    massf, _ = jm_truss(Af)
    return massf

def g(Af):
    _, stressf = jm_truss(Af)
    # print('g: ', stressf)
    return yield_stresses_jm - jnp.abs(stressf) 

def g_c(Af):
    _, stressf = c_truss(Af)
    # print('g_c: ', stressf)
    return yield_stresses_c - jnp.abs(stressf.real) + 1.j*jnp.abs(stressf.imag)# + is 0.5 #yield_stresses_c - jnp.abs(stressf) 
# setup
E = 25.0 * 10**3 # psi
E_special = 75.0 * 10**3 # psi
A_min = 0.1 # in^2

yield_stresses_c = jnp.ones(10, dtype = complex) * E
yield_stresses_jm = jnp.ones(10) * E
yield_stresses_c.at[8].set(E_special) #= E_special
yield_stresses_jm.at[8].set(E_special) #= E_special

area_bounds = [(A_min, jnp.inf)] * 10 # in^2
area_bounds = scipy.optimize.Bounds(jnp.ones(10)*0.1, jnp.inf)

# a) the derivatives of mass (objective, my TrussMass_c function) with respect to cross-sectional areas (design vars), dm/dA_i, for i = 1... n_x, finite dif, complex step, and AD
# TrussFunctionCalls = []
def truss_grad_finitediff(Af):
    return forward_diff(1e-3, Af, TrussMass_jm)

def truss_grad_complex(Af):
    return complex_step(1e-30j, Af, TrussMass_c)

def truss_grad_AD(Af):
    return jax.jacobian(TrussMass_jm)(Af)

A0 = jnp.zeros(10) + 5
A0_c = jnp.zeros(10, dtype = complex) + 5

finitediff_dmdA = truss_grad_finitediff(A0)
# print('stop here')
complex_dmdA = truss_grad_complex(A0_c)
AD_dmdA = truss_grad_AD(A0)
act_dmdA = jax.grad(TrussMass_jm)(A0)
# print('ran to here1')
#errors
finitediff_error = jnp.linalg.norm(finitediff_dmdA - act_dmdA) / jnp.linalg.norm(act_dmdA)
complex_error = jnp.linalg.norm(complex_dmdA - act_dmdA) / jnp.linalg.norm(act_dmdA)
AD_error = jnp.linalg.norm(AD_dmdA - act_dmdA) / jnp.linalg.norm(act_dmdA)
print("Finite Difference Relative Error (dmdA):", finitediff_error)
print("Complex Step Relative Error (dmdA):", complex_error)
print("AD Relative Error (dmdA):", AD_error)

# b) the derivatives of stress (constraints) with respect to cross-sectional areas (design vars), dsigma/dA_j, for i = 1...n_g and j = 1...n_x in other words, this is an n_g x n_x matrix (and in this case, n_g = n_x)
def truss_stress_grad_finitediff(Af):
    return forward_diff(1e-3, Af, g)

def truss_stress_grad_complex(Af):
    return complex_step(1e-30j, Af, g_c)

def truss_stress_grad_AD(Af):
    return jax.jacfwd(g)(A0)

finitediff_dsigmadA = truss_stress_grad_finitediff(A0)
# print('ran to here2')
complex_dsigmadA = truss_stress_grad_complex(A0_c)
# print('ran to here3')
AD_dsigmadA = truss_stress_grad_AD(A0)
# print('ran to here4')
act_dsigmadA = jax.jacobian(g)(A0)
# print('ran to here5')
# relative errors
finitediff_stress_error = jnp.linalg.norm(finitediff_dsigmadA - act_dsigmadA) / jnp.linalg.norm(act_dsigmadA)
complex_stress_error = jnp.linalg.norm(complex_dsigmadA - act_dsigmadA) / jnp.linalg.norm(act_dsigmadA)
AD_stress_error = jnp.linalg.norm(AD_dsigmadA - act_dsigmadA) / jnp.linalg.norm(act_dsigmadA) 
print("Finite Difference Stress Relative Error:", finitediff_stress_error)
print("Complex Step Stress Relative Error:", complex_stress_error)
print("AD Stress Relative Error:", AD_stress_error)


#%% 4.4 Truss optimization
MyMassTracker = []
TrussFunctionCalls = 0
# stress_constraints = [{'type':'ineq','fun':g, 'jac': truss_stress_grad_AD}] #TODO, you can add in the jacobian here
x0 = jnp.ones(10) * 10.
mass_ad = jax.jacobian(TrussMass_jm)
stress_ad = jax.jacobian(g)
stress_constraints = [{'type':'ineq','fun':g, 'jac': stress_ad}] 
res = minimize(TrussMass_jm, x0, jac = mass_ad,  constraints=stress_constraints, bounds=area_bounds, tol = 0.01, callback = MyCallbackFunc) #! plug in Jacobian's from the stress inside of constraints

# %% Convergence plotting
plt.plot(MyMassTracker)
plt.title('Mass Convergence')
plt.xlabel('Iterations')
plt.ylabel('Mass')
# %%
