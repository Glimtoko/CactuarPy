import math
import sys
import numpy

cimport numpy
cimport cython

from libc.math cimport sqrt


# Data types
DTYPE = numpy.double
ITYPE = numpy.int

ctypedef numpy.double_t DTYPE_t
ctypedef numpy.int_t ITYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def main_loop(
        numpy.ndarray[DTYPE_t, ndim=1] rho,
        numpy.ndarray[DTYPE_t, ndim=1] P,
        numpy.ndarray[DTYPE_t, ndim=1] u,
        numpy.ndarray[DTYPE_t, ndim=1] E,
        numpy.ndarray[DTYPE_t, ndim=1] momentum,
        double tend,
        double dtmax,
        double dx,
        flux_function,
        double gamma=1.4,
        model=riemann_iterative,
        double cfl=0.6
):
    cdef int ncells = len(P) - 2
        
    cdef numpy.ndarray[DTYPE_t, ndim=1] rho_f = numpy.zeros(ncells+2, float)
    cdef numpy.ndarray[DTYPE_t, ndim=1] mom_f = numpy.zeros(ncells+2, float)
    cdef numpy.ndarray[DTYPE_t, ndim=1] E_f = numpy.zeros(ncells+2, float)
       
    # Main loop
    cdef double t = 0.0
    cdef int step = 1
    cdef double S, dt, f, a
    cdef int i
    while True:
        S = 0
        for i in range(1, ncells+1):
            a = sqrt((gamma*P[i])/rho[i])
            S = max(S, a + u[i])
            
        dt = min(dtmax, cfl*dx/S)
        
        print("step: {0:4d}, t = {1:6.4f}, dt = {2:8.6e}".format(step, t, dt))
                   
        for i in range(1, ncells+2):
            rho_f[i], mom_f[i], E_f[i] = flux_function(
                u[i-1], rho[i-1], P[i-1],
                u[i], rho[i], P[i],
                gamma,
                model)
                
        f = dt/dx
        
        for i in range(1, ncells+1):                 
            # Conservative update
            rho[i] += f*(rho_f[i] - rho_f[i+1])
            momentum[i] += f*(mom_f[i] - mom_f[i+1])
            E[i] += f*(E_f[i] - E_f[i+1])
            
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