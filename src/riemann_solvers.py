import math
import sys

import riemann_sampler as rs
import riemann_exact as re

class solution():
    def __init__(self, rho, P, u, E):
        self.rho = rho
        self.P = P
        self.u = u
        self.E = E


def riemann_iterative(uL, rhoL, PL, uR, rhoR, PR, P0, gamma=1.4):
    TOL = 1.0e-6
    MAXITER = 100
    
    Pold = P0
    i = 0
    while True:
        i += 1
        Pnew = re.get_P(Pold, PL, rhoL, PR, rhoR, uL, uR, gamma)
        if Pnew < 0.0:
            print("Error: P < 0 in Riemann solve")
            sys.exit()
        d = re.rPc(Pnew, Pold)
        
        if d < TOL or i > MAXITER:
            break
        
        Pold = Pnew
           
    u = 0.5*(uL + uR) + 0.5*(re.fk(Pnew, PR, rhoR, gamma) - re.fk(Pnew, PL, rhoL, gamma))
    
    return (Pnew, u, None, None)
   
    
def riemann_PVRS1(uL, rhoL, PL, uR, rhoR, PR, P0, gamma=1.4):
    aL = math.sqrt((gamma*PL)/rhoL)
    aR = math.sqrt((gamma*PR)/rhoR)
    
    rho_bar = 0.5*(rhoL + rhoR)
    a_bar = 0.5*(aL + aR)
    
    Pstar = 0.5*(PL + PR) + 0.5*(uL - uR)*(rho_bar*a_bar)
    ustar = 0.5*(uL + uR) + 0.5*(PL - PR)/(rho_bar*a_bar)
    rhoLstar = rhoL + (uL - ustar)*(rho_bar/a_bar)
    rhoRstar = rhoR + (ustar - uR)*(rho_bar/a_bar)
    
    return Pstar, ustar, rhoLstar, rhoRstar


def riemann_PVRS2(uL, rhoL, PL, uR, rhoR, PR, P0, gamma=1.4):
    aL = math.sqrt((gamma*PL)/rhoL)
    aR = math.sqrt((gamma*PR)/rhoR)
    
    CL = rhoL * aL
    CR = rhoR * aR
    
    f = 1.0/(CL + CR)
    
    Pstar = f*(CR*PL + CL*PR + CL*CR*(uL - uR))
    ustar = f*(CL*uL + CR*uR + (PL - PR))
    rhoLstar = rhoL + (Pstar - PL)/(aL**2)
    rhoRstar = rhoR + (Pstar - PR)/(aR**2)
    
    return Pstar, ustar, rhoLstar, rhoRstar


def riemann_TRRS(uL, rhoL, PL, uR, rhoR, PR, P0, gamma=1.4):
    aL = math.sqrt((gamma*PL)/rhoL)
    aR = math.sqrt((gamma*PR)/rhoR)
    
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


def riemann_TSRS(uL, rhoL, PL, uR, rhoR, PR, P0, gamma=1.4):   
    f = (gamma - 1.0)/(gamma + 1.0)
    
    AL = 2.0/((gamma + 1.0)*rhoL)
    AR = 2.0/((gamma + 1.0)*rhoR)
    BL = f*PL
    BR = f*PR
    
    # Initial guess at pressure from PVRS model
    Pguess, d1, d2, d3 = riemann_PVRS1(uL, rhoL, PL, uR, rhoR, PR, P0, gamma)
    Pguess = max(Pguess, 0.0)
    
    gL = math.sqrt(AL/(Pguess + BL))
    gR = math.sqrt(AR/(Pguess + BR))
    
    Pstar = gL*PL + gR*PR - (uR - uL)
    Pstar /= (gL + gR)
    
    ustar = 0.5*(uL + uR) + 0.5*((Pstar - PR)*gR - (Pstar - PL)*gL)
    
    rhoLstar = rhoL * ((Pstar/PL + f)/(f*Pstar/PL + 1))
    rhoRstar = rhoR * ((Pstar/PR + f)/(f*Pstar/PR + 1))
    
    return Pstar, ustar, rhoLstar, rhoRstar
    
    

def solve(uL, rhoL, PL, uR, rhoR, PR, x, t, gamma=1.4, model=riemann_iterative):
    aL = math.sqrt((gamma*PL)/rhoL)
    aR = math.sqrt((gamma*PR)/rhoR)
    P0 = 0.5*(PL + PR) - 0.125*(uR - uL)*(rhoR + rhoL)*(aL + aR)
    P0 = max(1.0e-6, P0)
       
    Pstar, ustar, rhoLstar, rhoRstar = model(uL, rhoL, PL, uR, rhoR, PR, P0, gamma)
    if model == riemann_iterative:
        rhoxp, Pxp, uxp = rs.sample_exact(Pstar, ustar, x, t, uL, rhoL, PL, uR, rhoR, PR, gamma)
    else:
        rhoxp, Pxp, uxp = rs.sample_approx(Pstar, ustar, rhoLstar, rhoRstar, uL, rhoL, PL, uR, rhoR, PR, gamma)
    
    e = Pxp/((gamma - 1.0)*rhoxp)
    Exp = rhoxp * (0.5*uxp*uxp + e)
    
    return solution(rhoxp, Pxp, uxp, Exp)


def exact(uL, uR, rhoL, rhoR, PL, PR, t, gamma=1.4, xL=0.0, xR=1.0, diaphram=0.5, dx=0.001):
    # Initial guess at pressure in star region
    aL = math.sqrt((gamma*PL)/rhoL)
    aR = math.sqrt((gamma*PR)/rhoR)
    P0 = 0.5*(PL + PR) - 0.125*(uR - uL)*(rhoR + rhoL)*(aL + aR)
    P0 = max(1.0e-6, P0)
    
    Pstar, ustar, dummy1, dummy2 = riemann_iterative(uL, rhoL, PL, uR, rhoR, PR, P0, gamma)

    rho = []
    u = []
    P = []
    x = []
    E = []
    
    xv = xL
    while xv < xR:
        xp = xv - diaphram
        
        # Density, pressure and velocity from sampling the solution
        rhoxp, Pxp, uxp = rs.sample_exact(Pstar, ustar, xp, t, uL, rhoL, PL, uR, rhoR, PR, gamma)
        
        # Energy from EoS
        exp = Pxp/((gamma - 1.0)*rhoxp)
        
        rho.append(rhoxp)
        P.append(Pxp)
        E.append(exp)
        u.append(uxp)
        x.append(xv)
               
        xv += dx
        
    return x, solution(rho, P, u, E)
