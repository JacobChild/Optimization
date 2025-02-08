#LineSearch1.py
#Jacob Child
#Jan 30th, 2025

#Needed packages
#! .venv\Scripts\Activate.ps1
import numpy as np #it looks like I could do import jax.numpy as jnp and that might be faster than numpy even
import matplotlib.pyplot as plt
from jax import grad
from scipy.optimize import minimize

#Functions 
def sphere_func(vf):
    return vf[0]**2 + vf[1]**2

def plotter(ptsf, funcf, bounds = (-10,10)):
    # Do a contour plot of the sphere_func 
    x_examp = np.linspace(bounds[0], bounds[1])
    y_examp = np.linspace(bounds[0], bounds[1])
    X_examp_mesh, Y_examp_mesh = np.meshgrid(x_examp, y_examp)
    Z_examp = funcf([X_examp_mesh, Y_examp_mesh])

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
    # plt.legend()
    plt.show()

# Define the global variable
grad_func = None

def update_grad_func(func):
    global grad_func
    grad_func = grad(func)

def directional_deriv(ptf, pf):
    global grad_func
    return np.dot(grad_func(ptf),pf) 

def stepper(pt0f, stepf, pf): #initial point, stepsize, direction 
    return pt0f + np.dot(stepf,pf)

def alphafunc(x1f,x0f,pf):
    with np.errstate(divide='ignore', invalid='ignore'):
        af = np.divide((x1f - x0f), pf, out=np.full_like(x0f, np.nan), where=pf!=0)
    # Find the first non-NaN value
    non_nan_values = af[~np.isnan(af)]
    if len(non_nan_values) == 0:
        raise ValueError("All elements in af are NaN")
    
    first_non_nan = non_nan_values[0]
    
    # Ensure all elements are the same
    if not np.allclose(non_nan_values, first_non_nan, equal_nan=True):
        raise ValueError("Not all elements in af are the same")
    
    return first_non_nan
    

# Bracketing Function begin
# Goals/conditions at which the bracket function stops
# 1. The function value at the candidate step is higher than the value at the start of the line search. 
# 2. The step satisfies sufficient decrease, and the slope is positive.
# phi is my function, and phi_0 is really phi(0)
def bracket_func(pt0f, a_init_f, phif_0, phiprimef_0, muf_1, muf_2, sigmaf, pf, solve_funcf, debugf = False):
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
        # print(i)
        if i > 0:
            pt_tracker = np.vstack([pt_tracker, stepper(pt_tracker[-1], af2, pf)])
            phi_tracker = np.vstack([phi_tracker, solve_funcf(pt_tracker[-1])])
        
        cond1 = phi_tracker[-1] > (phi_tracker[0] + muf_1*af2*phiprime_tracker[0])
        cond2 = not first and phi_tracker[-1] > phi_tracker[-2]
        if cond1 and cond2: 
            if debugf:
                print("call pinpoint @ cond1 and cond2")
                # plotter(pt_tracker,solve_funcf)
            astar, pt_tracker = pinpoint(pt_tracker, phi_tracker, phiprime_tracker, 'quad', muf_1, muf_2, pf, solve_funcf)
            if debugf:
                print("after pinpoint")
                # plotter(pt_tracker,solve_funcf)
            return astar
        if True:#i > 0: #? why not i>0?
            phiprime_tracker = np.vstack([phiprime_tracker, directional_deriv(pt_tracker[-1],pf)])
        
        #? why is check3 true?
        # print("phiprime2: ", phiprime_tracker[-1])
        # print("phiprime0: ", phiprime_tracker[0])
        check3 = np.abs(phiprime_tracker[-1]) <= -muf_2*phiprime_tracker[0]
        if check3: 
            if debugf:
                print("astar = a2 at check3")
                # plotter(pt_tracker,solve_funcf)
            astar = alphafunc(pt_tracker[-1],pt_tracker[-2],pf)
            return astar
        
        check4 = phiprime_tracker[-1] >= 0 and i > 0 #this might force it to take a bad step
        if check4: 
            if debugf:
                print("call pinpoint @ check4")
                # plotter(pt_tracker,solve_funcf)
            astar, pt_tracker = pinpoint(pt_tracker, phi_tracker, phiprime_tracker, 'cubic', muf_1, muf_2, pf, solve_funcf)
            if debugf: 
                print("after pinpoint @ check4")
                # plotter(pt_tracker,solve_funcf)
            return astar
        
        af1 = af2
        af2 = sigmaf * af2 
        first = False 
        i += 1
        
# pinpoint function
def pinpoint(pttf, phitf, phiptf, methodf, mu1f, mu2f, pf,  solve_func, maxiter = 1000): #pt_tracker, phi_tracker, phiprime_tracker
    k = 0 
    #find a_low and a_high 
    if phitf[-1,0] < phitf[-2,0]:
        phi_low = phitf[-1, 0]
        a_low = alphafunc(pttf[-1], pttf[0], pf)
        phi_high = phitf[-2, 0]
        a_high = alphafunc(pttf[-2], pttf[0], pf)
        
    else:
        phi_low = phitf[-2, 0]
        a_low = alphafunc(pttf[-2], pttf[0], pf)
        phi_high = phitf[-1, 0]
        a_high = alphafunc(pttf[-1], pttf[0], pf)
    
    while k <= maxiter:
        
        #simple bisection method
        ap = (a_high + a_low) / 2.
            
        phip = solve_func(stepper(pttf[0],ap, pf))
        
        cond1 = phip > phitf[0,0] + mu1f * ap * phiptf[0,0]
        cond2 = phip > solve_func(stepper(pttf[0],a_low,pf))
        if cond1 or cond2: 
            a_high = ap 
            
        else:
            phiprimep = directional_deriv(stepper(pttf[0],ap,pf), pf)
            
            cond3 = np.abs(phiprimep) <= -mu2f*phiptf[0]
            cond4 = phiprimep * (a_high - a_low) >= 0
            
            if cond3: 
                astar = ap 
                newpoint = stepper(pttf[0],astar,pf)
                pttf = np.vstack([pttf,newpoint])
                return astar, pttf #return ap
            
            elif cond4:
                a_high = a_low
                
            a_low = ap 

        pttf = np.vstack([pttf, stepper(pttf[0],ap,pf)])
        # if directional_deriv(pttf[-1],pf) > 0 and directional_deriv(pttf[-2],pf) > 0:
        #     print("somehow positive grads, returning current ap")
        #     return ap
        k += 1
    print("uh oh, maxiter reached, returning current ap = 0.001, pttf to reset")
    newpoint = stepper(pttf[0],ap,pf)
    pttf = np.vstack([pttf,newpoint])
    # plotter(pttf, solve_func)
    return 0.001, pttf

def linesearch(directionmethod, pt0f, a_init_f, muf_1, muf_2, sigmaf, tauf, solve_funcf, debugf = False):
    # provide different options for the linesearch
    grad_tracker = np.array([grad_func(pt0f)])
    act_pt_tracker = np.array([pt0f])
    act_alpha_tracker = np.array([a_init_f])
    p_tracker = np.array([0.,0.])
    grad_mag_tracker = []
    
    if directionmethod == "steepest_descent":
        k = 0
        mems = 0
        while (np.linalg.norm(grad_tracker[-1], np.inf) > tauf):
            
            p = -grad_tracker[-1] / np.linalg.norm(grad_tracker[-1]) 
            p_tracker = np.vstack([p_tracker,p])
            if k == 0: 
                alphainit = a_init_f
            else:
                alphainit = act_alpha_tracker[-2,0] * grad_tracker[-2,0].T * p_tracker[-2,0] / (grad_tracker[-1,0].T * p_tracker[-1,0])
                
            act_alpha_tracker = np.vstack([act_alpha_tracker,alphainit])
            phi0 = solve_funcf(act_pt_tracker[-1])
            phiprime0 = directional_deriv(act_pt_tracker[-1], p_tracker[-1])
            amin = bracket_func(act_pt_tracker[-1], act_alpha_tracker[-1,0], phi0, phiprime0,muf_1, muf_2, sigmaf, p_tracker[-1], solve_funcf, False)
            new_point = stepper(act_pt_tracker[-1], amin, p_tracker[-1])
            act_alpha_tracker = np.vstack( [act_alpha_tracker, amin] )
            

            if debugf and mems == 99:
                print("finished k=",k)
                print("np.linalg.norm(grad_func(new_point),np.inf): ", np.linalg.norm(grad_func(new_point), np.inf))
                plotter(np.vstack([act_pt_tracker,new_point]), solve_funcf)
                    
            act_pt_tracker = np.vstack([act_pt_tracker,new_point]) #? why did I do the line above originally anyway?
            new_grad = grad_func(new_point)
            grad_tracker = np.vstack([grad_tracker, new_grad])
            # grad_mag_tracker.append(np.linalg.norm(grad_tracker[-1]))
            grad_mag_tracker.append(np.linalg.norm(grad_tracker[-1]))

            k += 1
            mems += 1
            if mems > 100: 
                # clear memory
                print("100 done, mems cleared :)")
                act_alpha_tracker = act_alpha_tracker[-4:-1]
                act_pt_tracker = act_pt_tracker[-4:-1]
                grad_tracker = grad_tracker[-4:-1]
                p_tracker = p_tracker[-4:-1]
                mems = 0
            
        if debugf:
                print("finished k=",k)
                print("np.linalg.norm(grad_func(new_point),np.inf): ", np.linalg.norm(new_grad, np.inf))
                plotter(np.vstack([act_pt_tracker,new_point]), solve_funcf, (-2.1,2.1))
                
        return act_pt_tracker[-1], solve_funcf(act_pt_tracker[-1]), grad_mag_tracker
    
    elif directionmethod == 'conj_grad':
        k = 0
        mems = 0
        reset = False
        while (np.linalg.norm(grad_tracker[-1], np.inf) > tauf):
            
            if k == 0 or reset:
                p = -grad_tracker[-1] / np.linalg.norm(grad_tracker[-1]) 
                p_tracker = np.vstack([p_tracker,p])
                act_alpha_tracker = np.vstack([act_alpha_tracker,a_init_f])
                reset = False
            else:
                # beta = grad_tracker[-1].T * grad_tracker[-1] / (grad_tracker[-2].T * grad_tracker[-2])
                beta = np.dot(grad_tracker[-1].T,grad_tracker[-1]) / np.dot(grad_tracker[-2].T,grad_tracker[-2])
                p_stpst = -grad_tracker[-1] / np.linalg.norm(grad_tracker[-1])
                p = -grad_tracker[-1] / np.linalg.norm(grad_tracker[-1]) + beta * p_tracker[-2]
                p = p / np.linalg.norm(p) 
                reset_cond = np.dot(grad_tracker[-1].T,grad_tracker[-2]) / np.dot(grad_tracker[-1].T,grad_tracker[-1])
                if reset_cond >= 0.1:
                    print("reset_cond hit")
                    reset = True
                if directional_deriv(act_pt_tracker[-1], p_stpst) < 0 and directional_deriv(act_pt_tracker[-1], p) > 0:
                    print("changed p from: ", p)
                    p = p_stpst
                
                p_tracker = np.vstack([p_tracker,p])
                
            phi0 = solve_funcf(act_pt_tracker[-1])
            phiprime0 = directional_deriv(act_pt_tracker[-1], p_tracker[-1])
            
            amin = bracket_func(act_pt_tracker[-1], act_alpha_tracker[-1,0], phi0, phiprime0,muf_1, muf_2, sigmaf, p_tracker[-1], solve_funcf, False)
            
            new_point = stepper(act_pt_tracker[-1], amin, p_tracker[-1])
            act_alpha_tracker = np.vstack( [act_alpha_tracker, amin] )
            act_pt_tracker = np.vstack([act_pt_tracker,new_point]) 
            new_grad = grad_func(new_point)
            grad_tracker = np.vstack([grad_tracker, new_grad])
            grad_mag_tracker.append(np.linalg.norm(grad_tracker[-1]))
            
            k += 1
            mems += 1
            if mems > 50: 
                # clear memory
                print("50 done, mems cleared :)")
                act_alpha_tracker = act_alpha_tracker[-4:-1]
                act_pt_tracker = act_pt_tracker[-4:-1]
                grad_tracker = grad_tracker[-4:-1]
                p_tracker = p_tracker[-4:-1]
                mems = 0
                reset = True
            
        if debugf:
                print("finished k=",k)
                print("np.linalg.norm(grad_func(new_point),np.inf): ", np.linalg.norm(new_grad, np.inf))
                plotter(np.vstack([act_pt_tracker,new_point]), solve_funcf, (-2.1,2.1))
                
        return act_pt_tracker[-1], solve_funcf(act_pt_tracker[-1]), grad_mag_tracker
    
    elif directionmethod == "quasi-newton": 
        k = 0
        mems = 0
        a_init_f = 1.
        reset = False
        I = np.eye(grad_tracker[-1].shape[0])
        Vk_tracker = []
        while (np.linalg.norm(grad_tracker[-1], np.inf) > tauf):
            
            if k == 0 or reset:
                Vk = 1. / np.linalg.norm(grad_tracker) * I
            else:
                #! matrix operations are critical
                s = act_pt_tracker[-1] - act_pt_tracker[-2]
                y = grad_tracker[-1] - grad_tracker[-2]
                # Reshape s and y to column vectors if needed
                s = s.reshape(-1, 1)
                y = y.reshape(-1, 1)
                sigma = 1.0 / (s.T @ y)  # Use matrix multiplication @
                Vk = (I - sigma * (s @ y.T)) @ Vk_tracker[-1] @ (I - sigma * (y @ s.T)) + sigma * (s @ s.T)
                
            Vk_tracker.append(Vk)
            p = np.dot(-Vk, grad_tracker[-1])
            # p = p / np.linalg.norm(p)
            p_stpst = -grad_tracker[-1] / np.linalg.norm(grad_tracker[-1])
            if directional_deriv(act_pt_tracker[-1], p_stpst) < 0 and directional_deriv(act_pt_tracker[-1], p) > 0:
                    print("changed p from: ", p)
                    p = p_stpst
            
            p_tracker = np.vstack([p_tracker,p])
            
            phi0 = solve_funcf(act_pt_tracker[-1])
            phiprime0 = directional_deriv(act_pt_tracker[-1], p_tracker[-1])
            
            amin = bracket_func(act_pt_tracker[-1], a_init_f, phi0, phiprime0,muf_1, muf_2, sigmaf, p_tracker[-1], solve_funcf, False)
            
            new_point = stepper(act_pt_tracker[-1], amin, p_tracker[-1])
            act_alpha_tracker = np.vstack( [act_alpha_tracker, amin] )
            act_pt_tracker = np.vstack([act_pt_tracker,new_point]) 
            new_grad = grad_func(new_point)
            grad_tracker = np.vstack([grad_tracker, new_grad])
            grad_mag_tracker.append(np.linalg.norm(grad_tracker[-1]))
            
            k += 1
            mems += 1
            if mems > 100: 
                # clear memory
                print("100 done, mems cleared :)")
                act_alpha_tracker = act_alpha_tracker[-4:-1]
                act_pt_tracker = act_pt_tracker[-4:-1]
                grad_tracker = grad_tracker[-4:-1]
                p_tracker = p_tracker[-4:-1]
                mems = 0
                reset = True
            
        if debugf:
                print("finished k=",k)
                print("np.linalg.norm(grad_func(new_point),np.inf): ", np.linalg.norm(new_grad, np.inf))
                plotter(np.vstack([act_pt_tracker,new_point]), solve_funcf, (-2.1,2.1))
                
        return act_pt_tracker[-1], solve_funcf(act_pt_tracker[-1]), grad_mag_tracker
    
    elif directionmethod == "scipy":
        #scipy minimize with x0 and solve_funcf
        result = minimize(solve_funcf, pt0f)
        return result.x, result.fun, result
        