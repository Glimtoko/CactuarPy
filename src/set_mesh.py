import numpy
import sys


def set_mesh(problem, gamma=1.4, L=1.0, x0=0.5, ncells=100, invert=False):
    if problem.upper() == "SOD":
        uL = 0.0
        uR = 0.0    
        rhoL = 1.0
        rhoR = 0.125
        PL = 1.0
        PR = 0.1
        tend = 0.25
        problem = "Sod Shock Tube"
    elif problem.upper() == "SOD20":
        uL = 0.0
        uR = 0.0    
        rhoL = 1.0
        rhoR = 0.125
        PL = 1.0
        PR = 0.1
        tend = 0.2
        problem = "Sod Shock Tube - 0.2s"
    elif problem == "123":
        rhoL = 1.0
        uL = -2.0
        PL = 0.4   
        rhoR = 1.0
        uR = 2.0    
        PR = 0.4
        tend = 0.15
        problem = "'123' Problem"
    elif problem.upper() == "BLAST":
        rhoL = 1.0
        uL = 0.0
        PL = 1000.0   
        rhoR = 1.0
        uR = 0.0    
        PR = 0.01
        tend = 0.012
        problem = "Blast Wave - Left-hand Side"
    elif problem.upper() == "BLAST2":
        rhoL = 1.0
        uL = 0.0
        PL = 0.01
        rhoR = 1.0
        uR = 0.0    
        PR = 100.0
        tend = 0.035
        problem = "Blast Wave - Right-hand Side"     
    elif problem.upper() == "COLLISION":
        rhoL = 5.99924
        uL = 19.5975
        PL = 460.894
        rhoR = 5.99242
        uR = -6.19633
        PR = 46.0950
        tend = 0.035
        problem = "Shock Collision"
    else:
        print("ERFROR: Unexpected problem name: {}".format(problem.upper()))
        sys.exit()
        
    dx = L/ncells   
    
    # Cell boundaries
    cx = numpy.zeros(ncells+2)
    
    # Conserved variables
    rho = numpy.zeros(ncells+2)
    momentum = numpy.zeros(ncells+2)
    E = numpy.zeros(ncells+2)
    
    # Primitive variables
    P = numpy.zeros(ncells+2)
    u = numpy.zeros(ncells+2)
    
    # Set cell centre positions
    for i in range(1, ncells+1):
        cx[i] = (i-0.5)*dx
    cx[0] = -dx
    cx[-1] = L+dx
    
    # Set initial density and pressure fields
    for i in range(1, ncells+1):
        xupper = cx[i] + dx/2.0
        if xupper <= x0:
            rho[i] = rhoL
            momentum[i] = rhoL*uL
            P[i] = PL
            u[i] = uL
            e = P[i]/((gamma - 1.0)*rho[i])
            E[i] = rho[i]*(0.5*u[i]*u[i] + e)
            
        else:
            rho[i] = rhoR
            momentum[i] = rhoR*uR
            P[i] = PR
            u[i] = uR
            e = P[i]/((gamma - 1.0)*rho[i])
            E[i] = rho[i]*(0.5*u[i]*u[i] + e)
    
    # Set boundaries
    rho[0] = rho[1]
    E[0] = E[1]
    momentum[0] = momentum[1]
    P[0] = P[1]
    u[0] = u[1]
    
    rho[-1] = rho[-2]
    E[-1] = E[-2]
    momentum[-1] = momentum[-2]
    P[-1] = P[-2]
    u[-1] = u[-2]

    return rho, momentum, E, P, u, cx, tend, dx