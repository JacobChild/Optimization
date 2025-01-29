#HW2_LineSearch.py
#Jacob Child
#Jan 25th, 2025

# To activate this environment: `.venv\Scripts\Activate.ps1`

#%% Problem 1.3 Statement
#1.3 Develop a line search algorithm (i.e. Algorithm 4.3 and 4.4) that given a direction p, your algorithm returns the optimal point along that line (i.e. the optimal step length along p). Include your code in a zip folder.
# %% My pseudo code and plan: 
# Pick and plot a simple bowl function, this will be used to test and troubleshoot my algorithm
# Code up (by parts!) Algorithm 4.3, which is the bracketing phase of the line search algorithm
#   Try to do this the correct python way with classes and objects, if it is too difficult, start my way (function based) and then adapt to python class and objects. 
# Test the bracketing algorithm on my bowl function
# Repeat the above, but for Algorithm 4.4 the pinpoint function

# %% Import needed packages 
import numpy as np #it looks like I could do import jax.numpy as jnp and that might be faster than numpy even
import matplotlib.pyplot as plt
from jax import grad
# %% My test function, taken from [Wikipedia page test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
# I will do the Sphere function (it has a global min at 0): f(x) = summation from i = 1 to n of x_i ^2 
# its bounds are -inf to inf and n >= 1
def sphere_func(vf):
    xf = vf[0]
    yf = vf[1]
    return (xf-7)**2 + yf**2 # I offset it a bit so I can start at 0 and it won't be an issue

sphere_func_grad = grad(sphere_func)

def directional_deriv(ptf, pf, funcf=sphere_func_grad):
    return np.dot(funcf(ptf),pf) 

def stepper(pt0f, stepf, pf): #initial point, stepsize, direction 
    return pt0f + np.dot(stepf,pf)

p = np.array([1.,0.]).T 

# %% Bracketing Function begin
# Goals/conditions at which the bracket function stops
# 1. The function value at the candidate step is higher than the value at the start of the line search. 
# 2. The step satisfies sufficient decrease, and the slope is positive.
# phi is my function, and phi_0 is really phi(0)
def bracket_func(pt0f, a_init_f, phif_0, phiprimef_0, muf_1, muf_2, sigmaf, pf, solve_funcf = sphere_func):
    # safety checks
    if muf_1 > muf_2 or muf_1 < 0 or muf_2 > 1:
        raise("Something is wrong with the mus, remember 0<mu1<mu2<1 :)")
    
    pt_tracker = np.array([pt0f])
    phi_tracker = np.array([phif_0])
    phiprime_tracker = np.array([phiprimef_0])
    af1 = 0.
    af2 = a_init_f
    phi1f = phif_0
    phiprime1f = phiprimef_0
    first = True 
    i = 0
    while True:
        print(i)
        if i > 0:
            pt_tracker = np.vstack([pt_tracker, stepper(pt_tracker[-1], af2, pf)])
            phi_tracker = np.vstack([phi_tracker, solve_funcf(pt_tracker[-1])])
        
        cond1 = phi_tracker[-1] > (phi_tracker[0] + muf_1*af2*phiprime_tracker[0])
        cond2 = not first and phi_tracker[-1] > phi_tracker[-2]
        if cond1 and cond2: 
            print("call pinpoint @ cond1 and cond2")
            plotter(pt_tracker)
            astar, pt_tracker = pinpoint(pt_tracker, phi_tracker, phiprime_tracker, 'quad', muf_1, muf_2, pf)
            plotter(pt_tracker)
            return astar
        
        phiprime_tracker = np.vstack([phiprime_tracker, directional_deriv(pt_tracker[-1],pf)])
        
        #? why is check3 true?
        print("phiprime2: ", phiprime_tracker[-1])
        print("phiprime0: ", phiprime_tracker[0])
        check3 = np.abs(phiprime_tracker[-1]) <= -mu2*phiprime_tracker[0]
        if check3: 
            print("astar = a2 at check3")
            plotter(pt_tracker)
            astar = pt_tracker[-1]
            return astar
        
        check4 = phiprime_tracker[-1] >= 0 
        if check4: 
            print("call pinpoint @ check4")
            plotter(pt_tracker)
            astar, pt_tracker = pinpoint(pt_tracker, phi_tracker, phiprime_tracker, 'cubic', muf_1, muf_2, pf)
            plotter(pt_tracker)
            return astar
        
        af1 = af2
        af2 = sigmaf * af2 
        first = False 
        i += 1
        
        
# %% pinpoint function
def pinpoint(pttf, phitf, phiptf, methodf, mu1f, mu2f, pf,  solve_func = sphere_func, maxiter = 100): #pt_tracker, phi_tracker, phiprime_tracker
    k = 0 
    
    while k <= maxiter:
        a1f = pttf[-2] 
        a2f = pttf[-1]
        
        if methodf == 'quad': # for quad I don't yet know phiprime2 and phi(pt2) > phi(pt1) 
            a_low = pttf[-2]
            phi_low = phitf[-2]
            a_high = pttf[-1]
            phi_high = pttf[-1]
            ap = (2*a1f * np.abs(phitf[-1] - phitf[-2]) + phiptf[-1]*(a1f**2 - a2f**2)) / (2 * np.abs(phitf[-1] - phitf[-2] + phiptf[-1]*(a1f-a2f)))
            
        if methodf == 'cubic': # I should know phiprime2, and phi(pt2) < phi(pt1) 
            phi_low = phitf[-1]
            a_low = pttf[-1]
            phi_high = phitf[-2]
            a_high = pttf[-2]
            
            beta1 =  phiptf[-2] + phiptf[-1] - 3* (phitf[-2] - phitf[-1]) / distance(a2f,a1f) #(a1f - a2f)
            beta2 = np.sign(a2f - a1f) * np.sqrt(beta1**2 - phiptf[-2]*phiptf[-1])
            ap = a2f - (a2f - a1f) * (phiptf[-1] + beta2 -beta1) / (phiptf[-1] - phiptf[-2] + 2* beta2)
            
        phip = solve_func(stepper(pttf[-2],ap, pf))
        
        cond1 = phip > phitf[0] + mu1f * ap * phiptf[0]
        cond2 = phip > phi_low
        if cond1 or cond2: 
            a_high = ap 
            
        else:
            phiprimep = directional_deriv(stepper(pttf[-2],ap,pf), pf)
            
            cond3 = np.abs(phiprimep) <= -mu2f*phiptf[0]
            cond4 = phiprimep * distance(a_low, a_high) >= 0
            
            if cond3: 
                astar = ap 
                return ap #return ap
            
            elif cond4:
                a_high = a_low
                
            a_low = ap 

        pttf = np.vstack(pttf, ap)
        
        k += 1
    print("uh oh, maxiter reached, returning current ap, pttf")
    return ap, pttf
    
            

        
    
# %%
def finitedif_2D(ptf, af, pf, funcf=sphere_func): 
    pt2f = stepper(ptf, af * .001, pf)
    deltaphif = funcf(pt2f) - funcf(ptf) #y2-y1
    deltaptf = pt2f -ptf #x2 - x1
    gradf = deltaphif / np.linalg.norm(deltaptf) # (y2-y1)/(x2-x1) as an absolute difference
    return gradf 

def distance(pt1f, pt2f):
    deltaptf = pt2f - pt1f 
    return np.linalg.norm(deltaptf)

# %%
#if __name__ == '__main__': #shift everything below here over later
def plotter(ptsf):
    # Do a contour plot of the sphere_func 
    x_examp = np.linspace(-10, 10)
    y_examp = np.linspace(-10, 10)
    X_examp_mesh, Y_examp_mesh = np.meshgrid(x_examp, y_examp)
    Z_examp = sphere_func([X_examp_mesh, Y_examp_mesh])

    # Create the contour plot
    plt.contour(X_examp_mesh, Y_examp_mesh, Z_examp, levels=50, cmap='viridis')

    # Scatter plot the points in ptsf
    num_points = ptsf.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, num_points))

    for i, (pt, color) in enumerate(zip(ptsf, colors)):
        plt.scatter(pt[0], pt[1], color=color, label=f'Point {i}')
        plt.text(pt[0], pt[1], f'P{i}', fontsize=9, ha='right')

    plt.colorbar(label='Function value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour plot of the sphere function')
    p#lt.legend()
    plt.show()


# %% Before actually doing a bracket function, I am going to just run through myself 
p = np.array([1.,0.]).T #direction to search ie along the x axis, so y = y we hold y constant 
pt0 = np.array([-4.,3.]) #initial point
phi0 = sphere_func(pt0) #intial output
ainit = 2. #initial step size
# I want to end up pt1 = [1,2]
pt1 = stepper(pt0, ainit, p)
phiprime0 = directional_deriv(pt0,p) #I expect this to be -14

# that is kind of all of the initial things out of the way, now set some conditions
mu1 = 1e-4 #sufficient decrease factor, #? come back and explain this
mu2 = .1 #sufficient curvature factor, should be greater than mu1 
sigma = 2. #step size increase factor
pt_tracker = np.array(pt0)


# %% begin "Bracket" function 
a_1 = 0.
a_2 = ainit 
phi_1 = phi0 
phiprime_1 = phiprime0
first = True 
i = 1
# %%
amin = bracket_func(np.array([8.,9.]), ainit, phi0, phiprime0, mu1, mu2, sigma,np.array([0.,-1.]))