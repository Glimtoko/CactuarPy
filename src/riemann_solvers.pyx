import math
import sys

#import riemann_sampler as rs
#import riemann_exact as re

cimport cython
from libc.math cimport sqrt


@cython.cdivision(True)
def riemann_iterative(
    double uL,
    double rhoL,
    double PL,
    double uR,
    double rhoR,
    double PR,
    double P0,
    double gamma=1.4
):  
    TOL = 1.0e-6
    MAXITER = 100
    
    cdef double Pold, Pnew, u, d
    Pold = P0
    
    cdef int i = 0
    while True:
        i += 1
        Pnew = get_P(Pold, PL, rhoL, PR, rhoR, uL, uR, gamma)
        if Pnew < 0.0:
            print("Error: P < 0 in Riemann solve")
            sys.exit()
        d = rPc(Pnew, Pold)
        
        if d < TOL or i > MAXITER:
            break
        
        Pold = Pnew
           
    u = 0.5*(uL + uR) + 0.5*(fk(Pnew, PR, rhoR, gamma) - fk(Pnew, PL, rhoL, gamma))
    
    cdef (double, double, double, double) solution = (Pnew, u, 0.0, 0.0)
    
    return solution
   

@cython.cdivision(True)    
def riemann_PVRS1(
    double uL,
    double rhoL,
    double PL,
    double uR,
    double rhoR,
    double PR,
    double P0,
    double gamma=1.4
):  
    aL = sqrt((gamma*PL)/rhoL)
    aR = sqrt((gamma*PR)/rhoR)
    
    rho_bar = 0.5*(rhoL + rhoR)
    a_bar = 0.5*(aL + aR)
    
    Pstar = 0.5*(PL + PR) + 0.5*(uL - uR)*(rho_bar*a_bar)
    ustar = 0.5*(uL + uR) + 0.5*(PL - PR)/(rho_bar*a_bar)
    rhoLstar = rhoL + (uL - ustar)*(rho_bar/a_bar)
    rhoRstar = rhoR + (ustar - uR)*(rho_bar/a_bar)
    
    return Pstar, ustar, rhoLstar, rhoRstar


@cython.cdivision(True)
def riemann_PVRS2(
    double uL,
    double rhoL,
    double PL,
    double uR,
    double rhoR,
    double PR,
    double P0,
    double gamma=1.4
):  
    aL = sqrt((gamma*PL)/rhoL)
    aR = sqrt((gamma*PR)/rhoR)
    
    CL = rhoL * aL
    CR = rhoR * aR
    
    f = 1.0/(CL + CR)
    
    Pstar = f*(CR*PL + CL*PR + CL*CR*(uL - uR))
    ustar = f*(CL*uL + CR*uR + (PL - PR))
    rhoLstar = rhoL + (Pstar - PL)/(aL**2)
    rhoRstar = rhoR + (Pstar - PR)/(aR**2)
    
    return Pstar, ustar, rhoLstar, rhoRstar


@cython.cdivision(True)
def riemann_TRRS(
    double uL,
    double rhoL,
    double PL,
    double uR,
    double rhoR,
    double PR,
    double P0,
    double gamma=1.4
):  
    aL = sqrt((gamma*PL)/rhoL)
    aR = sqrt((gamma*PR)/rhoR)
    
    z = (gamma - 1.0)/(2.0*gamma)
    PLR = (PL/PR)**z
    
    ustar = PLR*uL/aL + uR/aR + 2.0*(PLR - 1.0)/(gamma - 1.0)
    ustar /= (PLR/aL + 1.0/aR)
    
    Pstar = 0.5*(
        PL*(1 + (gamma - 1.0)/(2*aL)*(uL - ustar))**(1.0/z) +
        PR*(1 + (gamma - 1.0)/(2*aR)*(ustar - uR))**(1.0/z)
    )
    
    rhoLstar = rhoL*(Pstar/PL)**(1/gamma)
    rhoRstar = rhoR*(Pstar/PR)**(1/gamma)
    
    return Pstar, ustar, rhoLstar, rhoRstar


@cython.cdivision(True)
def riemann_TSRS(
    double uL,
    double rhoL,
    double PL,
    double uR,
    double rhoR,
    double PR,
    double P0,
    double gamma=1.4
):   
    cdef double f = (gamma - 1.0)/(gamma + 1.0)
    
    cdef double AL = 2.0/((gamma + 1.0)*rhoL)
    cdef double AR = 2.0/((gamma + 1.0)*rhoR)
    cdef double BL = f*PL
    cdef double BR = f*PR
    
    # Initial guess at pressure from PVRS model
    cdef double aL = sqrt((gamma*PL)/rhoL)
    cdef double aR = sqrt((gamma*PR)/rhoR)
    cdef double rho_bar = 0.5*(rhoL + rhoR)
    cdef double a_bar = 0.5*(aL + aR)
    
    cdef double Pguess = 0.5*(PL + PR) + 0.5*(uL - uR)*(rho_bar*a_bar)
    Pguess = max(Pguess, 0.0)
    
    cdef double gL = sqrt(AL/(Pguess + BL))
    cdef double gR = sqrt(AR/(Pguess + BR))
    
    cdef double Pstar = gL*PL + gR*PR - (uR - uL)
    Pstar /= (gL + gR)
    
    cdef double ustar = 0.5*(uL + uR) + 0.5*((Pstar - PR)*gR - (Pstar - PL)*gL)
    
    cdef double rhoLstar = rhoL * ((Pstar/PL + f)/(f*Pstar/PL + 1))
    cdef double rhoRstar = rhoR * ((Pstar/PR + f)/(f*Pstar/PR + 1))
    
    return Pstar, ustar, rhoLstar, rhoRstar
    
    
@cython.cdivision(True)
def solve(
    double uL,
    double rhoL,
    double PL,
    double uR,
    double rhoR,
    double PR,
    double x,
    double t,
    double gamma=1.4,
    model=riemann_iterative
):
    cdef double aL = sqrt((gamma*PL)/rhoL)
    cdef double aR = sqrt((gamma*PR)/rhoR)
    cdef double P0 = 0.5*(PL + PR) - 0.125*(uR - uL)*(rhoR + rhoL)*(aL + aR)
    P0 = max(1.0e-6, P0)

    cdef double Pstar, ustar, rhoLstar, rhoRstar
       
    Pstar, ustar, rhoLstar, rhoRstar = model(uL, rhoL, PL, uR, rhoR, PR, P0, gamma)
    
    cdef double rhoxp, Pxp, uxp
    if model == riemann_iterative:
        rhoxp, Pxp, uxp = sample_approx(Pstar, ustar, 0.0, 0.0, uL, rhoL, PL, uR, rhoR, PR, gamma, exact=True)
    else:
        rhoxp, Pxp, uxp = sample_approx(Pstar, ustar, rhoLstar, rhoRstar, uL, rhoL, PL, uR, rhoR, PR, gamma)
    
    cdef double e = Pxp/((gamma - 1.0)*rhoxp)
    cdef double Exp = rhoxp * (0.5*uxp*uxp + e)
    
    return rhoxp, Pxp, uxp, Exp

