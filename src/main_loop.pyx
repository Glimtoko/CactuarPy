import math
import sys
import riemann_solvers as rs

import numpy

cimport numpy
cimport cython


# Data types
DTYPE = numpy.double
ITYPE = numpy.int

ctypedef numpy.double_t DTYPE_t
ctypedef numpy.int_t ITYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def main_loop(
        numpy.ndarray[DTYPE_t, ndim=1] rho,
        numpy.ndarray[DTYPE_t, ndim=1] P,
        numpy.ndarray[DTYPE_t, ndim=1] u,
        numpy.ndarray[DTYPE_t, ndim=1] E,
        numpy.ndarray[DTYPE_t, ndim=1] momentum,
        tend,
        dtmax,
        dx,
        flux_function,
        gamma=1.4,
        model=rs.riemann_iterative
):
    ncells = len(P) - 2
    
    F = [None]*(ncells+2)
    
    # Main loop
    t = 0.0
    step = 1
    while True:
        S = 0
        for i in range(1, ncells+1):
            a = math.sqrt((gamma*P[i])/rho[i])
            S = max(S, a + u[i])
            
        dt = min(dtmax, 0.6*dx/S)
        
        print("step: {0:4d}, t = {1:6.4f}, dt = {2:8.6e}".format(step, t, dt))
                   
        for i in range(1, ncells+2):
            F[i] = flux_function(u[i-1], rho[i-1], P[i-1], u[i], rho[i], P[i], gamma, model)
     
        f = dt/dx
        
        for i in range(1, ncells+1):
            momL = F[i].mom
            momR = F[i+1].mom
            EL = F[i].E
            ER = F[i+1].E
                  
            # Conservative update
            rho[i] += f*(F[i].rho - F[i+1].rho)
            momentum[i] += f*(momL - momR)
            E[i] += f*(EL - ER)
            
            # Primative update
            P[i] = (gamma - 1.0)*(E[i] - 0.5*rho[i]*u[i]*u[i])
            u[i] = momentum[i]/rho[i]
    
        
            if P[i] < 0.0:
                print("\nNegative pressure in cell {}".format(i))
                print("Pressure =",P[i])
                print("Energy =",E[i])
                print("Rho =",rho[i])
                print("u =",u[i])
                sys.exit()
    
            if rho[i] < 0.0:
                print("\nNegative density in cell {}".format(i))
                print("Pressure =",P[i])
                print("Energy =",E[i])
                print("Rho =",rho[i])
                print("u =",u[i])
                sys.exit()         
        
        if t > tend:
            break
        t += dt
        step += 1
        
    return step